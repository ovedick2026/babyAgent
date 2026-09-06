"""
SynapseMind - 能量扩散激活引擎 (core/activation.py)

将 CONFIRMED 知识子图物化为 rustworkx 带权有向图，支撑阻尼能量扩散推演。

核心铁律：
- 推演只依赖 status='CONFIRMED' 的突触（数据源头由 graph_cortex.get_confirmed_subgraph
  保证），本模块对 PENDING / REJECTED 天然不可见；
- 边权 W_ij = 突触 weight ∈ [0,1]，是扩散公式 E_{t+1}(j) = Σ_i (E_t(i) × W_ij × λ) − θ
  的唯一传导系数；
- relation / evidence 元数据随边负载保留，供 Task 3.2.3 因果路径回溯时
  拼接真实证据链（绝不允许凭空编造推演依据）。

本文件当前实现：
- Task 3.2.1  build_digraph(): CONFIRMED 子图 dict → rustworkx PyDiGraph
- Task 3.2.2  diffuse_energy(): 阻尼能量扩散迭代
              （E_{t+1}(j) = Σ_i E_t(i)·W_ij·λ − θ；λ=0.85，θ=0.1，最大 3 跳；
               scipy 稀疏矩阵向量化，缺依赖时自动降级纯 Python 等价实现；
               低能量剪枝 + 不动点提前止损；同步记录各节点首次激活跳数与
               最大贡献入边，为 3.2.3 因果回溯与证据链拼装奠基）

后续任务将在此文件追加：
- Task 3.2.3  激活序列排序与因果路径回溯 (rank_activations)
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger("synapsemind.activation")

# rustworkx 延迟感知：导入失败不阻塞模块加载（仅真正建图时快速失败），
# 便于在未安装依赖的环境下完成 py_compile / 文档生成等静态流程。
try:
    import rustworkx as rx
except ImportError:  # pragma: no cover - 环境缺依赖时触发
    rx = None


# PyDiGraph 节点负载 = 概念 dict：{id, name, category, description}
# PyDiGraph 边负载   = 突触 dict：{rel_id, weight, relation, evidence}
Subgraph = dict
NodeIndex = Dict[str, int]  # concept_id -> PyDiGraph 节点索引


def build_digraph(subgraph: Subgraph) -> Tuple["rx.PyDiGraph", NodeIndex]:
    """
    Task 3.2.1: 将 graph_cortex.get_confirmed_subgraph 产出的子图 dict
    构建为 rustworkx PyDiGraph 带权有向图。

    构建规则：
      - 节点：按子图 nodes 列表顺序逐一加入（顺序确定 → 索引确定，
        保证同构子图在两次构建间产生完全一致的索引布局，利于缓存与测试）；
      - 边：weight 作为能量扩散的传导系数存于边负载 dict 的 "weight" 键，
        relation / evidence / rel_id 一并保留（Task 3.2.3 回溯证据链直接取用）；
      - 防御性校验：若某条边的端点不在节点集中（正常不会发生，因子图节点
        即由 CONFIRMED 边端点归纳而来），跳过该边并打警告，绝不静默丢整图。

    Args:
        subgraph: {"nodes": [...], "edges": [...]}，见 graph_cortex.get_confirmed_subgraph。

    Returns:
        (graph, node_index) 二元组：
          graph      — rustworkx PyDiGraph，节点负载为概念 dict，边负载为突触 dict；
          node_index — concept_id → PyDiGraph 节点索引 的映射，供扩散引擎
                       定位种子节点与回写激活能量。

    Raises:
        RuntimeError: rustworkx 未安装。
    """
    if rx is None:
        raise RuntimeError("rustworkx 未安装，请先执行 pip install -r requirements.txt")

    graph = rx.PyDiGraph()
    node_index: NodeIndex = {}

    # 1) 概念节点：节点负载 = 概念 dict 本身，回溯因果链时可直接取 name 展示
    for node in subgraph.get("nodes", []):
        idx = graph.add_node(node)
        node_index[node["id"]] = idx

    # 2) 突触边：有向 src → dst，边负载保留全部元数据
    skipped = 0
    for edge in subgraph.get("edges", []):
        src_idx = node_index.get(edge["src_id"])
        dst_idx = node_index.get(edge["dst_id"])
        if src_idx is None or dst_idx is None:
            skipped += 1
            logger.warning("边 %s 端点缺失于节点集，已跳过", edge.get("rel_id", "?"))
            continue
        graph.add_edge(src_idx, dst_idx, {
            "rel_id": edge.get("rel_id", ""),
            "weight": float(edge.get("weight", 0.0)),
            "relation": edge.get("relation", ""),
            "evidence": edge.get("evidence", ""),
        })

    logger.debug(
        "PyDiGraph 构建完成: %d 节点 / %d 边（跳过 %d 条悬空边）",
        graph.num_nodes(), graph.num_edges(), skipped,
    )
    return graph, node_index


def build_brain_digraph(brain_id: str) -> Tuple["rx.PyDiGraph", NodeIndex]:
    """
    便捷入口：按 brain_id 拉取该脑 CONFIRMED 子图并构建 PyDiGraph。

    供 Task 3.2.2 扩散迭代与上层推演路由（Task 4.1.2 / 5.2.2）一键调用；
    内部经 graph_cortex → brain_manager 单例连接，不产生额外句柄。
    """
    from core.graph_cortex import get_brain_confirmed_subgraph  # 延迟导入，避免环依赖

    return build_digraph(get_brain_confirmed_subgraph(brain_id))


# ───────────────────────────────────────────────
# 阻尼能量扩散迭代 (Task 3.2.2)
# ───────────────────────────────────────────────

# 扩散超参（与 Todo 规范严格一致：λ=0.85，θ=0.1，最大 3 跳）
DEFAULT_LAMBDA = 0.85
DEFAULT_THETA = 0.1
DEFAULT_MAX_HOPS = 3
# 单边贡献 / 终态能量低于该值即剪枝，防止数值尘埃无限传播
PRUNE_THRESHOLD = 1e-6


def _extract_edges(graph):
    """跨版本安全地抽取全部 (src_idx, dst_idx, edge_data) 三元组（平行边不丢）。

    首选 rustworkx ≥0.12 的 edge_index_map()（{edge_idx: (u, v, data)}，
    平行边各自独立）；极老版本回退 edge_list()+get_edge_data()（平行边仅取首条）。
    """
    try:
        return list(graph.edge_index_map().values())
    except AttributeError:
        return [(u, v, graph.get_edge_data(u, v)) for u, v in graph.edge_list()]


def _normalize_seeds(seeds) -> Dict[str, float]:
    """种子归一化：dict 直取正权重；list/tuple 视为等权；总能量归一到 1.0。"""
    if isinstance(seeds, dict):
        raw = {cid: float(w) for cid, w in seeds.items() if float(w) > 0}
    else:
        raw = {cid: 1.0 for cid in seeds}
    if not raw:
        return {}
    total = sum(raw.values())
    return {cid: w / total for cid, w in raw.items()}


def diffuse_energy(
    graph,
    node_index: NodeIndex,
    seeds,
    *,
    lam: float = DEFAULT_LAMBDA,
    theta: float = DEFAULT_THETA,
    max_hops: int = DEFAULT_MAX_HOPS,
    prune_threshold: float = PRUNE_THRESHOLD,
) -> dict:
    """
    Task 3.2.2: 阻尼能量扩散迭代。

    公式：E_{t+1}(j) = Σ_i (E_t(i) × W_ij × λ) − θ
      - λ（阻尼系数，默认 0.85）：能量每跨一条突触衰减一次；
      - θ（激活阈值，默认 0.1）：对节点每跳扣除一次，流入不足以抵扣则熄灭；
      - 波传播语义：节点当跳能量全部向下游传导，不与自身残留叠加；
      - 最多传播 max_hops（默认 3）跳。

    剪枝策略（能量低于阈值的传播提前剪枝）：
      - 单边贡献 c = E_t(i)·W_ij·λ ≤ prune_threshold 的边直接跳过；
      - 扣除 θ 后能量 ≤ prune_threshold 的节点置零（含负值熄灭）。

    向量化：scipy/numpy 可用时以 CSR 稀疏矩阵做 e_vec @ W 向量乘；
    缺依赖时自动降级为等价的纯 Python 邻接表迭代（个人级 CONFIRMED
    子图规模 × 3 跳，纯 Python 完全够用），两条路径结果逐位一致。

    Args:
        graph:       build_digraph 产出的 PyDiGraph。
        node_index:  concept_id → 节点索引映射。
        seeds:       {concept_id: 权重}（权重内部归一化，总注入能量恒为 1.0；
                     传 list/tuple 则等权处理）。
        lam/theta/max_hops/prune_threshold: 扩散超参。

    Returns:
        {"energies":     {concept_id: 终态能量(>0)},
         "hop_first":    {concept_id: 首次激活跳数（种子=0）},
         "contributors": {concept_id: {src_id, rel_id, relation, evidence,
                                       contribution, hop}}（每节点仅记贡献最大的一条入边）}
    """

    ids = list(node_index.keys())
    n = len(ids)
    if n == 0:
        return {"energies": {}, "hop_first": {}, "contributors": {}}

    seed_map = _normalize_seeds(seeds)
    if not seed_map:
        raise ValueError("seeds 为空或权重非法，无法启动能量扩散")

    # 邻接表：单边传导系数 c = W_ij·λ 已折算，低于剪枝阈值的边直接不出表
    adj: Dict[int, list] = {}
    for u, v, data in _extract_edges(graph):
        c = float(data.get("weight", 0.0)) * lam
        if c <= prune_threshold:
            continue
        adj.setdefault(u, []).append((v, c, data))

    # scipy/numpy 可用时构建 CSR 稀疏矩阵，传播走 e_vec @ W 向量乘
    np = None
    W = None
    if adj:
        try:
            import numpy as _np
            from scipy.sparse import csr_matrix

            rows, cols, vals = zip(
                *[(u, v, c) for u, lst in adj.items() for v, c, _ in lst]
            )
            W = csr_matrix((vals, (rows, cols)), shape=(n, n))
            np = _np
        except ImportError:
            np = None

    # 初始能量注入（种子 = 第 0 跳）
    energy = [0.0] * n
    hop_first: Dict[str, int] = {}
    contributors: Dict[str, dict] = {}
    for cid, w in seed_map.items():
        idx = node_index.get(cid)
        if idx is None:
            logger.warning("种子 %s 不在 CONFIRMED 子图内，已忽略", cid)
            continue
        energy[idx] = w
        hop_first[cid] = 0

    # 峰值能量追踪（含第 0 跳种子注入）：波传播语义下中间节点把能量传走后
    # 终态归零（energies 仅剩叶节点存活），peaks 刻画"曾经激活多强"，
    # 供 Task 4.1.2 纯图推演对规则条件做 AND 命中判定与结论排序。
    peak = energy[:]

    # 逐跳波传播（最多 max_hops 跳）
    for _hop in range(1, max_hops + 1):
        # 1) 本跳各节点入流：向量乘（scipy）或邻接表累加（纯 Python，带单边剪枝）
        if np is not None:
            e_vec = np.asarray(energy, dtype=float)
            inflow = (e_vec @ W).tolist()
        else:
            inflow = [0.0] * n
            for u, lst in adj.items():
                e = energy[u]
                if e <= prune_threshold:
                    continue
                for v, c, _data in lst:
                    inflow[v] += e * c

        # 2) 贡献者记账：两路径统一走邻接表，保证证据链元数据一致；
        #    同时记录本跳真正发生传导的节点（旧能量随波流走，不再驻留）
        any_active = False
        transmitted = set()
        for u, lst in adj.items():
            e = energy[u]
            if e <= prune_threshold:
                continue
            any_active = True
            transmitted.add(u)
            for v, c, data in lst:
                contrib = e * c
                if contrib <= prune_threshold:
                    continue
                vid = ids[v]
                prev = contributors.get(vid)
                if prev is None or contrib > prev["contribution"]:
                    contributors[vid] = {
                        "src_id": ids[u],
                        "rel_id": data.get("rel_id", ""),
                        "relation": data.get("relation", ""),
                        "evidence": data.get("evidence", ""),
                        "contribution": contrib,
                        "hop": _hop,
                    }
        # 3) 扣除激活阈值 θ 并剪枝，作为下一跳的能量基底。
        #    hop_first（首次激活跳数）在此统一定案：只有 θ 抵扣后仍存活的节点
        #    才算真正"激活"，流入不足以覆盖 θ 的节点视为熄灭，不记激活跳数；
        #    波传播语义补充：已传导节点的旧能量随波流走、不驻留，仅保留新入流
        #    激活；未传导节点（典型为无出边的叶节点）持有能量原样保留 —— 叶节点
        #    收到的激活能量无处传导，须作为终态能量存活到最终报告，不能被后续
        #    跳的 θ 扣除抹掉（不与残留叠加 → 取 max 而非相加）。
        for v in range(n):
            val = inflow[v] - theta
            val = val if val > prune_threshold else 0.0
            if v in transmitted:
                energy[v] = val
            else:
                energy[v] = max(energy[v], val)
            if energy[v] > prune_threshold and ids[v] not in hop_first:
                hop_first[ids[v]] = _hop
            if energy[v] > peak[v]:
                peak[v] = energy[v]

        if not any_active:
            break  # 波前已熄灭，提前止损

    energies = {
        cid: energy[idx]
        for cid, idx in node_index.items()
        if energy[idx] > prune_threshold
    }
    # 峰值能量视图：波传播语义下中间节点把能量传走后终态可能归零（energies
    # 仅剩叶节点存活），peaks 刻画"曾经激活多强"—— 规则断言的 AND 条件节点
    # 可能只是传导枢纽而非叶节点，用峰值而非终态判定命中才不会漏判。
    peaks = {
        cid: peak[idx]
        for cid, idx in node_index.items()
        if peak[idx] > prune_threshold
    }
    # contributors 与 hop_first 对齐：只保留"真正激活过"（θ 抵扣后存活）的节点，
    # 熄灭节点的入边记账一并丢弃，防止 3.2.3 因果回溯把未激活节点拼进证据链
    activated = set(hop_first.keys())
    contributors = {cid: rec for cid, rec in contributors.items() if cid in activated}
    logger.debug("扩散完成: %d/%d 节点存活（峰值视图 %d）", len(energies), n, len(peaks))
    return {"energies": energies, "peaks": peaks, "hop_first": hop_first, "contributors": contributors}


def rank_activations(diffusion_result: dict, top_k=None) -> list:
    """
    Task 3.2.3: 激活序列排序与因果路径回溯。

    排序：将 diffuse_energy 的终态能量按降序产出激活节点序列（top_k 可截断，
          供上层"取最高能结论"直接使用）；
    回溯：对每个激活节点沿其"最大贡献入边"逆向行走至种子（无贡献者记录者即
          种子），拼出 种子 → … → 该节点 的因果链条。链上每条边携带
          rel_id / relation / evidence 真实证据元数据，供 Task 5.2.2 组装
          答案与 Task 6.3.2 打印证据链 —— 绝不允许凭空编造推演依据。

    无环性保证：贡献边 (u→v) 记账于第 hop(v) 跳，而 u 在该跳之前已被激活，
    故沿贡献者链 hop 严格递减，天然无环；另设 seen 集合兜底防御脏数据。

    Args:
        diffusion_result: diffuse_energy() 的返回值
            {"energies", "hop_first", "contributors"}。
        top_k: 仅保留能量最高的前 k 个节点；None 表示全部。

    Returns:
        按终态能量降序排列的列表，每项：
        {
            "id":     概念 id,
            "energy": 终态能量,
            "hop":    首次激活跳数（种子为 0）,
            "chain":  [ # 按传导顺序：种子 → … → 该节点
                {"src_id", "rel_id", "relation", "evidence", "dst_id"},
            ],
        }
    """
    energies = diffusion_result.get("energies", {})
    hop_first = diffusion_result.get("hop_first", {})
    contributors = diffusion_result.get("contributors", {})

    ranked = sorted(energies.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None and top_k >= 0:
        ranked = ranked[:top_k]

    sequence = []
    for cid, energy in ranked:
        chain = []
        seen = set()
        cur = cid
        while cur is not None and cur not in seen:
            seen.add(cur)
            rec = contributors.get(cur)
            if rec is None:
                break  # 无贡献者 → 回溯抵达扩散种子，链条封口
            chain.append({
                "src_id": rec.get("src_id", ""),
                "rel_id": rec.get("rel_id", ""),
                "relation": rec.get("relation", ""),
                "evidence": rec.get("evidence", ""),
                "dst_id": cur,
            })
            cur = rec.get("src_id")
        chain.reverse()  # 逆推所得 → 反转为正序因果链
        sequence.append({
            "id": cid,
            "energy": energy,
            "hop": hop_first.get(cid),
            "chain": chain,
        })
    return sequence


if __name__ == "__main__":
    # 冒烟自测：用 2 节点 1 边的最小子图验证构建逻辑（不依赖 Kùzu）
    logging.basicConfig(level=logging.DEBUG)
    demo = {
        "nodes": [
            {"id": "c_a", "name": "恶寒", "category": "症状", "description": ""},
            {"id": "c_b", "name": "表证", "category": "证型", "description": ""},
        ],
        "edges": [
            {"rel_id": "s_demo", "src_id": "c_a", "dst_id": "c_b",
             "weight": 0.9, "relation": "suggests", "evidence": "示例证据"},
        ],
    }
    g, idx = build_digraph(demo)
    print("nodes =", g.num_nodes(), "edges =", g.num_edges(), "index =", idx)

"""
SynapseMind - 纯图推演执行器 (core/inference.py)

Task 4.1.2: 模式 1（纯手工）闭环推演。

流程：输入线索 → 定位种子概念 → activation 阻尼能量扩散 →
      多前提汇流命中规则断言 → 组装确定性答案。

核心铁律（与 Readme/Todo 对齐）：
- 全程零外部 LLM 调用：推演 100% 由 CONFIRMED 子图上的能量扩散驱动，
  每一条推演依据都能回溯到真实突触的 evidence 文本，绝不编造；
- 只读 CONFIRMED 子图（数据源头 graph_cortex.get_confirmed_subgraph 保证），
  PENDING / REJECTED 突触对本引擎天然不可见；
- 知识不足时明确返回 insufficient 状态与「尚未掌握该知识」话术，
  严禁在无证据情况下给出模棱两可的猜测。

规则断言的图物化与命中判定：
- "A + B 导致 C" 型断言录入后物化为 A→C、B→C 两条 CONFIRMED 突触
  （物化动作属路由层职责，本模块只消费图）；
- 命中判定采用「多前提汇流」：候选结论节点若存在 ≥ min_fanin 条
  入边、且每条入边的源节点峰值能量均 ≥ min_peak，则视为断言命中——
  峰值（而非终态）取自 Task 3.2.2 的 peaks 记录，因为波传播语义下
  前提节点把能量传走后终态可能归零，用终态判定会漏判。

本文件实现：
- locate_seeds():        线索文本 → 种子概念（精确名优先，包含匹配次之）
- _reason_on_subgraph(): 纯函数推演核心（不依赖 Kùzu，便于离线测试）
- pure_reason():         brain_id 入口薄封装（拉取 CONFIRMED 子图后调核心）
"""

import logging
from typing import Dict, List, Optional

from core.activation import (
    build_digraph,
    diffuse_energy,
    rank_activations,
)

logger = logging.getLogger("synapsemind.inference")

# ───────────────────────────────────────────────
# 推演判定阈值
# ───────────────────────────────────────────────

# 前提节点峰值能量 ≥ 该值才计入「已激活前提」
# （种子峰值=1.0；单跳 0.9 权边到达约 0.66；取 0.25 容忍 2~3 跳衰减）
DEFAULT_MIN_PEAK = 0.25
# 汇流入边数 ≥ 该值才判定为规则断言命中（2 = 典型 "A+B⇒C"）
DEFAULT_MIN_FANIN = 2
# 激活序列最多返回的节点数（防止全图倾倒）
DEFAULT_TOP_K = 8


def locate_seeds(subgraph: dict, clue: str) -> Dict[str, float]:
    """
    线索定位：把用户输入的自然语言线索映射到 CONFIRMED 子图中的种子概念。

    匹配优先级（与「确定性优先」原则一致，越靠前越可信）：
      1. 精确同名（可命中多个重名概念，全部入选且等权）；
      2. 双向包含匹配（线索包含概念名 / 概念名包含线索），
         按名称长度降序取最特异的前 3 个，能量均分。

    Args:
        subgraph: get_confirmed_subgraph 产出的 {"nodes": [...], "edges": [...]}。
        clue:     用户线索文本（自动 strip）。

    Returns:
        {concept_id: 归一化前权重}；无任何命中返回空 dict（调用方据此
        走 insufficient 分支，绝不静默挑一个凑数）。
    """
    clue = (clue or "").strip()
    if not clue:
        return {}

    nodes = subgraph.get("nodes", [])

    exact = [n for n in nodes if n.get("name", "") == clue]
    if exact:
        return {n["id"]: 1.0 for n in exact}

    partial = [
        n for n in nodes
        if n.get("name", "") and (n["name"] in clue or clue in n["name"])
    ]
    # 名称更长者更特异（如「风寒表证」优先于「表证」），最多取 3 个均分能量
    partial.sort(key=lambda n: len(n["name"]), reverse=True)
    partial = partial[:3]
    if not partial:
        return {}
    w = 1.0 / len(partial)
    return {n["id"]: w for n in partial}


def _reason_on_subgraph(
    subgraph: dict,
    clue: str,
    *,
    min_peak: float = DEFAULT_MIN_PEAK,
    min_fanin: int = DEFAULT_MIN_FANIN,
    top_k: int = DEFAULT_TOP_K,
    extra_seeds: Optional[Dict[str, float]] = None,
) -> dict:
    """
    纯函数推演核心：在给定 CONFIRMED 子图上完成完整闭环。

    extra_seeds（升级B 增量扩散）：上一轮高能节点经衰减后的增量种子
    {concept_id: weight}，与本轮线索定位种子合并后统一重扩散。

    与 Kùzu / brain_manager 完全解耦，离线单测只需构造 subgraph dict。

    Returns:
        {
          "status": "ok" | "insufficient",
          "message": 状态说明（insufficient 时为固定话术），
          "clue":   原始线索,
          "seeds":  {concept_id: weight},
          "answer": None | {
              "conclusion": 结论概念 dict,
              "energy":     结论终态能量,
              "peak":       结论峰值能量,
              "hop":        结论首次激活跳数,
              "chain":      [种子→…→结论 的真实突触证据链],
              "assertion":  None | {"conditions": [前提概念 dict...],
                                    "evidence": [各前提边 evidence...]},
          },
          "activations": [按能量降序的激活节点列表（≤top_k），
                          每项 {id, name, energy, hop, chain}],
        }
    """
    clue = (clue or "").strip()
    seeds = locate_seeds(subgraph, clue)

    # 升级B 增量扩散：extra_seeds（上轮高能节点经衰减后的增量种子）并入
    # 本轮线索定位种子；同节点冲突取较大权重，非法/越界条目直接丢弃。
    # 并入发生在 insufficient 判定之前——纯追问（线索不命中概念）也能
    # 依靠上轮余温继续扩散（「还有呢」语义）。
    if extra_seeds:
        for cid, w in extra_seeds.items():
            try:
                w = float(w)
            except (TypeError, ValueError):
                continue
            if 0.0 < w <= 1.0:
                seeds[cid] = max(seeds.get(cid, 0.0), w)

    if not seeds:
        return {
            "status": "insufficient",
            "message": "尚未掌握该知识：线索未命中任何已确认概念",
            "clue": clue,
            "seeds": {},
            "answer": None,
            "activations": [],
        }

    graph, node_index = build_digraph(subgraph)

    # 图中真实存在的种子才参与扩散（locate_seeds 本就产自子图，防御性过滤）
    valid_seeds = {cid: w for cid, w in seeds.items() if cid in node_index}
    if not valid_seeds:
        return {
            "status": "insufficient",
            "message": "尚未掌握该知识：种子概念不在已确认子图内",
            "clue": clue,
            "seeds": {},
            "answer": None,
            "activations": [],
        }

    diffusion = diffuse_energy(graph, node_index, valid_seeds)
    peaks: Dict[str, float] = diffusion.get("peaks", {})
    energies: Dict[str, float] = diffusion.get("energies", {})
    hop_first: Dict[str, int] = diffusion.get("hop_first", {})
    contributors: Dict[str, dict] = diffusion.get("contributors", {})

    if not peaks:
        return {
            "status": "insufficient",
            "message": "尚未掌握该知识：已确认图谱中无自线索出发的激活路径",
            "clue": clue,
            "seeds": valid_seeds,
            "answer": None,
            "activations": [],
        }

    name_by_id = {n["id"]: n for n in subgraph.get("nodes", [])}

    def _chain_of(cid: str) -> List[dict]:
        """沿 contributors 逐跳回溯最大贡献入边，拼出 种子→…→cid 正序证据链。"""
        chain: List[dict] = []
        seen: set = set()
        cur = cid
        while cur is not None and cur not in seen:
            seen.add(cur)
            rec = contributors.get(cur)
            if rec is None:
                break  # 抵达扩散种子，链条封口
            chain.append({
                "src_id": rec.get("src_id", ""),
                "rel_id": rec.get("rel_id", ""),
                "relation": rec.get("relation", ""),
                "evidence": rec.get("evidence", ""),
                "dst_id": cur,
            })
            cur = rec.get("src_id")
        chain.reverse()
        return chain

    # 激活序列（含因果链）：以「峰值能量」为序——波传播语义下中间枢纽把能量
    # 传走后终态归零，若按终态排序会把恰好充当规则结论的枢纽节点整支漏掉
    # （如 "A+B⇒C" 中的 C 把能量继续传给下游时终态为 0）。
    activated_ids = sorted(peaks, key=lambda c: peaks[c], reverse=True)
    activations = [
        {
            "id": cid,
            "name": name_by_id.get(cid, {}).get("name", cid),
            "energy": energies.get(cid, 0.0),
            "peak": peaks.get(cid, 0.0),
            "hop": hop_first.get(cid),
            "chain": _chain_of(cid),
        }
        for cid in activated_ids[:top_k]
    ]

    # ── 规则断言命中判定（多前提汇流）──
    # 候选结论 = 峰值达标且非种子的曾激活节点；按「(激活前提数, 峰值)」降序取最优。
    # 前提判定：种子恒视为已激活前提（用户亲手给出的线索），其余节点看峰值。
    best: Optional[dict] = None
    for cid in activated_ids:
        if cid in valid_seeds:
            continue  # 结论不能是种子自身
        p = peaks.get(cid, 0.0)
        if p < min_peak:
            continue

        # 收集该节点的全部「已激活前提」入边
        # （rustworkx incoming_edges 返回 (src_idx, dst_idx, edge_data)）
        idx = node_index[cid]
        fired: List[str] = []
        evidences: List[str] = []
        try:
            incoming = graph.incoming_edges(idx)
        except AttributeError:  # 极老版本回退线性扫描
            incoming = [
                (u, v, graph.get_edge_data(u, v))
                for u, v in graph.edge_list() if v == idx
            ]
        for src_idx, _dst_idx, edata in incoming:
            src_id = graph.get_node_data(src_idx)["id"]
            if src_id in valid_seeds or peaks.get(src_id, 0.0) >= min_peak:
                if src_id not in fired:  # 平行边同源只记一次前提
                    fired.append(src_id)
                evidences.append(edata.get("evidence", ""))

        if len(fired) >= min_fanin:
            score = (len(fired), p)
            if best is None or score > best["_score"]:
                best = {"_score": score, "cid": cid, "fired": fired, "evidences": evidences}

    # ── 组装最终答案 ──
    # 结论优先取断言命中者（多前提汇流是最强确定性证据）；无断言命中时
    # 退化为峰值最高的非种子激活节点（单链传导推演）；若除种子外无任何
    # 被传导激活的节点（hop≥1），视为知识不足，绝不把线索原样当答案退还。
    if best is not None:
        conclusion_id = best["cid"]
    else:
        derived = [cid for cid in activated_ids
                   if cid not in valid_seeds and hop_first.get(cid, 0) >= 1]
        if not derived:
            return {
                "status": "insufficient",
                "message": "尚未掌握该知识：已确认图谱中无自线索出发的激活路径",
                "clue": clue,
                "seeds": valid_seeds,
                "answer": None,
                "activations": activations,
            }
        conclusion_id = derived[0]

    answer = {
        "conclusion": name_by_id.get(conclusion_id, {"id": conclusion_id}),
        "energy": energies.get(conclusion_id, 0.0),
        "peak": peaks.get(conclusion_id, 0.0),
        "hop": hop_first.get(conclusion_id),
        "chain": _chain_of(conclusion_id),
        "assertion": None,
    }
    if best is not None:
        answer["assertion"] = {
            "conditions": [name_by_id[c] for c in best["fired"]],
            "evidence": [e for e in best["evidences"] if e],
        }

    logger.info(
        "纯图推演完成: clue=%r seeds=%d 结论=%s (peak=%.3f, assertion=%s)",
        clue, len(valid_seeds), conclusion_id,
        answer["peak"], best is not None,
    )
    return {
        "status": "ok",
        "message": "ok",
        "clue": clue,
        "seeds": valid_seeds,
        "answer": answer,
        "activations": activations,
    }


def pure_reason(
    brain_id: str,
    clue: str,
    *,
    min_peak: float = DEFAULT_MIN_PEAK,
    min_fanin: int = DEFAULT_MIN_FANIN,
    top_k: int = DEFAULT_TOP_K,
    extra_seeds: Optional[Dict[str, float]] = None,
) -> dict:
    """
    brain_id 入口薄封装：拉取该脑 CONFIRMED 子图后执行纯函数推演核心。

    extra_seeds（升级B 增量扩散）：透传至推演核心，供多轮对话将
    上轮高能节点衰减后并入本轮种子。

    供上层路由（模式 1 API / OpenAI 兼容端点的 manual 分支）直接调用；
    任何 Kùzu 异常向上抛出，由调用方决定 5xx 语义，本层不吞错。
    """
    from core.graph_cortex import get_brain_confirmed_subgraph  # 延迟导入防环

    subgraph = get_brain_confirmed_subgraph(brain_id)
    return _reason_on_subgraph(
        subgraph, clue, min_peak=min_peak, min_fanin=min_fanin, top_k=top_k,
        extra_seeds=extra_seeds,
    )


if __name__ == "__main__":
    # 冒烟自测：A+B⇒C 断言物化为双入边，验证多前提汇流命中
    import logging as _logging
    _logging.basicConfig(level=logging.INFO)

    demo = {
        "nodes": [
            {"id": "a", "name": "恶寒", "category": "症状", "description": ""},
            {"id": "b", "name": "发热", "category": "症状", "description": ""},
            {"id": "c", "name": "表实证", "category": "证型", "description": ""},
            {"id": "d", "name": "咳嗽", "category": "症状", "description": ""},
        ],
        "edges": [
            {"rel_id": "s1", "src_id": "a", "dst_id": "c", "weight": 0.9,
             "relation": "rule", "evidence": "《伤寒论》第1条"},
            {"rel_id": "s2", "src_id": "b", "dst_id": "c", "weight": 0.9,
             "relation": "rule", "evidence": "《伤寒论》第1条"},
            {"rel_id": "s3", "src_id": "c", "dst_id": "d", "weight": 0.6,
             "relation": "may_cause", "evidence": "肺气不宣"},
        ],
    }
    import json
    result = _reason_on_subgraph(demo, "恶寒 发热")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # ── 升级B 增量扩散自检 ──
    # 1) 线索不命中但携带 extra_seeds（上轮余温）→ 不走 insufficient，继续扩散
    carry = _reason_on_subgraph(demo, "", extra_seeds={"c": 0.5})
    assert carry["status"] == "ok" and carry["answer"] is not None, carry
    assert carry["activations"], "余温种子应产出激活序列"
    print("extra_seeds 纯追问续扩散 ✓  激活:",
          [a["name"] for a in carry["activations"]])

    # 2) 无 extra_seeds 且线索不命中 → 仍走 insufficient（原语义无回归）
    bare = _reason_on_subgraph(demo, "不存在的概念")
    assert bare["status"] == "insufficient" and bare["activations"] == [], bare
    print("无线索无余温 insufficient 原语义 ✓")

    # 3) 非法/越界 extra_seeds 条目静默丢弃，不炸不污染
    dirty = _reason_on_subgraph(demo, "恶寒 发热",
                                extra_seeds={"ghost": "abc", "c": 9.9, "d": -0.5})
    assert dirty["status"] == "ok", dirty
    print("非法 extra_seeds 净化 ✓")

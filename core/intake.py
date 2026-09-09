"""
SynapseMind - 待收纳池：提案落库与人工处置 (core/intake.py)

Task 4.2.3: 辅助模式（模式 2）代谢链路的落库与处置环节。

链路定位：
    速读秘书提取提案 (metabolism, Task 4.2.2)
      → 产出为「概念名」形态（source/target 是名字字符串，无 id）
    本模块 commit_proposals：
      概念名 → 查找/建档获得 id（复用优先，缺失新建）
      → 写入 PENDING 突触（绝不直接 CONFIRMED）
    人工处置（本模块其余函数，仅登录用户可触达）：
      list_pending / edit_proposal / confirm_proposals /
      reject_proposals / delete_proposals

核心铁律（与 Readme/Todo 对齐，升级A/升级C 修订）：
- 提案落库默认 PENDING：LLM 产出是「待收纳」候选；CONFIRMED 仅两条
  合法通道 —— ①人工处置路径；②自动收纳白名单（升级A：weight ≥ 阈值
  且 evidence 充分的高置信提案落库即转正，并按比例抽样入复审队列
  供人工抽检，低置信提案绝不自动转正）；
- 升级C 来源收窄：白名单自动转正仅限 intake（速读秘书）通道；对话
  回流（knowledge_source=dialogue）即使高置信也一律 PENDING —— 口语
  歧义噪声大，转正权完全归人工处置，通道外的自动流程仍不得越权；
- 概念名解析绝不自动合并：同名多概念时取字典序第一个复用，
  并在返回中标注 reused 供前端弱提示（绝不合并、绝不改名）；
- 单条失败不拖垮整批：逐条独立落库，失败项连同原因一并返回；
- 每次成功写操作后触发 sync_to_storage 原子刷盘，
  刷盘失败仅告警不阻断（本地工作副本不受影响，/data 保留旧镜像）。
"""

import logging
import random
from typing import Iterable, List, Optional, Tuple, Union

from config import get_auto_confirm_config
from storage import persistence as _persistence
from core.graph_cortex import (
    KNOWLEDGE_SOURCES,
    add_concept,
    add_synapse,
    confirm_synapse,
    delete_synapse,
    get_confirmed_edge,
    get_synapse,
    is_book_class,
    list_concepts,
    reject_synapse,
    update_synapse,
)
from core.intake_store import enqueue_auto_review

# 抽样判定用独立随机实例：不污染全局 random 状态（避免扰动其他模块的随机序列）
_sampler = random.Random()

logger = logging.getLogger("synapsemind.intake")


def _flush() -> None:
    """写操作后的统一刷盘钩子：失败仅告警（绝不阻断处置主流程）。"""
    try:
        result = _persistence.sync_to_storage()
        logger.debug("intake 写操作后刷盘完成: %s", result)
    except Exception as exc:  # 刷盘异常不影响内存态正确性
        logger.warning("intake 写操作后刷盘失败（/data 保留旧镜像）: %s", exc)


# ───────────────────────────────────────────────
# 概念名解析：查找复用优先，缺失才建档
# ───────────────────────────────────────────────

def _find_or_create_concept(conn, name: str, category: str = "general",
                            knowledge_source: str = "intake") -> Tuple[str, bool]:
    """
    按概念名精确查找；命中则复用（返回既有 id），缺失则新建（返回新 id）。

    同名多概念的取舍：list_concepts 按 name 字典序稳定排序，取第一个 ——
    这不是「合并」而是「挂靠」：速读秘书只知道名字，不知道病机细节，
    概念身份主权始终归人；前端可依据 reused 标注弹出弱提示让用户改挂。

    来源归属（升级C）：仅新建概念继承本次落库的 knowledge_source；
    复用既有概念绝不改写其原归属（来源是档案出身，不是最近访客）。

    Returns:
        (concept_id, created) 二元组；created=True 表示本次新建。
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("概念名不得为空")
    if len(name) > 128:
        raise ValueError(f"概念名超长（>{128} 字）")

    exact = [c for c in list_concepts(conn, keyword=name) if c["name"] == name]
    if exact:
        return exact[0]["id"], False
    return add_concept(conn, name=name, category=category,
                       knowledge_source=knowledge_source), True


def _resolve_concept(conn, name: str, cache: dict,
                     knowledge_source: str = "intake") -> Tuple[str, bool]:
    """带批内缓存的概念解析：同一批提案中同名概念只建/查一次。"""
    if name in cache:
        return cache[name], False
    cid, created = _find_or_create_concept(
        conn, name, knowledge_source=knowledge_source)
    cache[name] = cid
    return cid, created


# ───────────────────────────────────────────────
# 提案批量落库 (Task 4.2.3a)
# ───────────────────────────────────────────────

def commit_proposals(conn, proposals: List[dict], sqlite_conn=None,
                     knowledge_source: str = "intake") -> dict:
    """
    将提案批量落库：默认 PENDING，高置信走自动收纳白名单（仅限 intake）。

    处理规则：
      - 逐条独立落库：单条失败（空字段/数值非法/写库异常）仅记入 failed，
        不拖垮整批；
      - 升级A 半自主收纳（仅 intake 通道）：weight ≥ 阈值 且 evidence 字数
        ≥ 下限 的「高置信」提案落库即自动转 CONFIRMED（白名单通道，直接
        进入可推演知识），并按抽样比例登记入复审队列供人工抽检；低置信
        提案一律保持 PENDING 等人工确认 —— 自动通道绝不触碰低置信提案；
      - 升级C 对话回流（knowledge_source="dialogue"）：一律 PENDING，绝不
        自动转正、绝不入复审抽样 —— 口语歧义与对话噪声大，转正权完全归
        人工处置（白名单是 intake 专属通道，dialogue 没有白名单）；
      - 升级D 来源分级审查（仅 dialogue 通道）：提案若命中既有 CONFIRMED
        书籍类突触（同有向端点对 + 同关系），说明对话试图修改投喂/存量
        权威知识 → 入池前直接拒绝并计入 refused（书籍类禁止对话修改，
        仅限人工手动修改）；经验类（manual/dialogue）命中照常入池待
        人工裁决。被拒提案已解析出的新概念节点保留（概念是中性实体，
        其 knowledge_source=dialogue 属实记录对话提及，不构成知识修改）；
      - 概念解析：批内同名缓存复用；跨批次靠库内同名精确匹配复用；
        两级都未命中才真正新建（category 统一 general，人工可后改；
        新建概念继承本次落库的 knowledge_source，复用概念不改归属）；
      - weight 越界值在此 clamp 进 [0,1]（与 metabolism 层净化双保险）。

    Args:
        conn:        Kùzu Connection。
        proposals:   [{source, target, relation, weight, evidence}, ...]
        sqlite_conn: 可选 SQLite 连接（brain.db），提供时自动收纳提案
                     才能按比例写入复审队列；None 时跳过入队（图谱不受影响）。
        knowledge_source: 知识来源标记（合法值见 graph_cortex.KNOWLEDGE_
                     SOURCES）。速读秘书链路默认 intake；对话回流链路显式传
                     dialogue（升级C）。

    Returns:
        {"total": 提案总数,
         "committed": [{rel_id, src_id, src_name, dst_id, dst_name,
                        src_reused, dst_reused, relation, weight, evidence,
                        knowledge_source, auto_confirmed}, ...],
         "failed":    [{index, error}, ...],
         "refused":   [{index, src_name, dst_name, relation,
                        existing_rel_id, existing_source}, ...]（仅升级D
                        dialogue 通道审查产生，其他通道恒空）,
         "concepts_created": 本次新建概念数,
         "concepts_reused":  本次复用既有概念数,
         "auto_confirmed":   高置信自动转正条数（仅 intake 通道可能非零）}
    """
    if knowledge_source not in KNOWLEDGE_SOURCES:
        raise ValueError(
            f"非法 knowledge_source={knowledge_source!r}，仅允许 {KNOWLEDGE_SOURCES}")
    auto_cfg = get_auto_confirm_config()
    review_batch: list[dict] = []  # 命中抽样的自动收纳提案（循环后统一入队）
    summary: dict = {
        "total": len(proposals),
        "committed": [],
        "failed": [],
        "refused": [],
        "concepts_created": 0,
        "concepts_reused": 0,
        "auto_confirmed": 0,
    }
    cache: dict = {}  # 概念名 → id（批内缓存）

    for i, p in enumerate(proposals):
        try:
            if not isinstance(p, dict):
                raise ValueError("提案非对象结构")
            src_name = str(p.get("source") or "").strip()
            dst_name = str(p.get("target") or "").strip()
            relation = str(p.get("relation") or "").strip()
            if not src_name or not dst_name or not relation:
                raise ValueError("source/target/relation 存在空值")
            try:
                weight = min(1.0, max(0.0, float(p.get("weight", 0.5))))
            except (TypeError, ValueError):
                raise ValueError(f"weight 非数值: {p.get('weight')!r}")
            evidence = str(p.get("evidence") or "")

            src_id, src_new = _resolve_concept(
                conn, src_name, cache, knowledge_source=knowledge_source)
            dst_id, dst_new = _resolve_concept(
                conn, dst_name, cache, knowledge_source=knowledge_source)
            summary["concepts_created"] += int(src_new) + int(dst_new)
            summary["concepts_reused"] += int(not src_new) + int(not dst_new)

            # 升级D 来源分级审查（仅 dialogue 通道）：对话试图修改书籍类
            # 既有知识（同有向端点对 + 同关系的 CONFIRMED 突触）→ 入池前
            # 拒绝（书籍类禁止对话修改，仅限人工手动修改）；经验类命中
            # 照常入池。被拒提案已解析出的概念节点保留（中性实体，如实
            # 记录对话提及）。
            if knowledge_source == "dialogue":
                existing = get_confirmed_edge(conn, src_id, dst_id, relation)
                if existing is not None and is_book_class(existing["knowledge_source"]):
                    logger.info(
                        "升级D 审查拒绝（对话修改书籍类知识）: %s -[%s]-> %s "
                        "命中既有 %s（knowledge_source=%s）",
                        src_name, relation, dst_name,
                        existing["rel_id"], existing["knowledge_source"])
                    summary["refused"].append({
                        "index": i,
                        "src_name": src_name, "dst_name": dst_name,
                        "relation": relation,
                        "existing_rel_id": existing["rel_id"],
                        "existing_source": existing["knowledge_source"],
                    })
                    continue

            # 升级A 判定：高置信白名单 → 落库即 CONFIRMED；其余默认 PENDING。
            # 升级C 收窄：白名单仅限 intake 通道 —— dialogue/manual/legacy 来源
            # 即使 weight 再高也一律 PENDING（转正权归人工，自动转正是速读秘书专属）。
            high_confidence = (
                knowledge_source == "intake"
                and weight >= auto_cfg["weight_threshold"]
                and len(evidence.strip()) >= auto_cfg["min_evidence_len"]
            )
            rel_id = add_synapse(
                conn, src_id, dst_id, relation,
                weight=weight, evidence=evidence,
                status="CONFIRMED" if high_confidence else "PENDING",
                knowledge_source=knowledge_source,
            )
            summary["committed"].append({
                "rel_id": rel_id,
                "src_id": src_id, "src_name": src_name, "src_reused": not src_new,
                "dst_id": dst_id, "dst_name": dst_name, "dst_reused": not dst_new,
                "relation": relation, "weight": weight, "evidence": evidence,
                "knowledge_source": knowledge_source,
                "auto_confirmed": high_confidence,
            })
            if high_confidence:
                summary["auto_confirmed"] += 1
                # 抽样复审：按比例随机挑部分自动收纳提案入队（sqlite 可用才登记）
                if sqlite_conn is not None and _sampler.random() < auto_cfg["sample_rate"]:
                    review_batch.append({
                        "rel_id": rel_id,
                        "src_name": src_name, "dst_name": dst_name,
                        "relation": relation, "weight": weight, "evidence": evidence,
                    })
        except Exception as exc:
            logger.warning("提案 #%d 落库失败: %s", i, exc)
            summary["failed"].append({"index": i, "error": str(exc)})

    if review_batch:
        try:
            enqueued = enqueue_auto_review(sqlite_conn, review_batch)
            summary["review_enqueued"] = enqueued
        except Exception as exc:  # 复审入队失败不回头撤销转正：仅告警
            logger.warning("复审队列入队失败（%d 条丢弃）: %s", len(review_batch), exc)

    logger.info(
        "提案落库完成: %d/%d 成功（新建概念 %d / 复用概念 %d；高置信自动转正 %d）",
        len(summary["committed"]), summary["total"],
        summary["concepts_created"], summary["concepts_reused"],
        summary["auto_confirmed"],
    )
    _flush()
    return summary


# ───────────────────────────────────────────────
# 待收纳池查询与人工处置 (Task 4.2.3b)
# ───────────────────────────────────────────────

def list_pending(conn) -> list[dict]:
    """
    列出待收纳池全部 PENDING 突触（含端点概念名，供审核面板直读直显）。

    升级D：附 knowledge_source，前端据此渲染来源徽标（书籍/经验/手工/
    迁移），审核者可一眼分辨提案出身再落裁决。

    Returns:
        [{rel_id, src_id, src_name, dst_id, dst_name,
          weight, relation, evidence, knowledge_source}, ...]，
        按 rel_id 稳定排序。
    """
    result = conn.execute(
        "MATCH (a:Concept)-[s:SYNAPSE]->(b:Concept) WHERE s.status = 'PENDING' "
        "RETURN s.rel_id, a.id, a.name, b.id, b.name, "
        "s.weight, s.relation, s.evidence, s.knowledge_source ORDER BY s.rel_id"
    )
    pending: list[dict] = []
    while result.has_next():
        row = result.get_next()
        pending.append({
            "rel_id": row[0],
            "src_id": row[1], "src_name": row[2],
            "dst_id": row[3], "dst_name": row[4],
            "weight": row[5], "relation": row[6], "evidence": row[7],
            "knowledge_source": row[8],
        })
    return pending


def _as_rel_id_list(rel_ids: Union[str, Iterable[str]]) -> List[str]:
    """入参归一：单个 rel_id 字符串或任意可迭代 → 列表（去重保序）。"""
    if isinstance(rel_ids, str):
        return [rel_ids]
    seen, ordered = set(), []
    for rid in rel_ids:
        if rid and rid not in seen:
            seen.add(rid)
            ordered.append(rid)
    return ordered


def edit_proposal(
    conn,
    rel_id: str,
    weight: Optional[float] = None,
    relation: Optional[str] = None,
    evidence: Optional[str] = None,
) -> bool:
    """
    单条编辑待收纳提案（滑块调权 / 改关系标签 / 改证据）。

    状态闸口：仅 PENDING 可编辑 —— CONFIRMED 是已确认知识，改写它
    会破坏「确认即冻结」语义（补救路径：删除后重新提案）；REJECTED 已终态。

    Returns:
        True=已编辑；False=提案不存在。

    Raises:
        ValueError: 目标非 PENDING 状态或字段非法。
    """
    syn = get_synapse(conn, rel_id)
    if syn is None:
        return False
    if syn["status"] != "PENDING":
        raise ValueError(f"仅 PENDING 提案可编辑，当前状态 {syn['status']}")
    ok = update_synapse(conn, rel_id, weight=weight, relation=relation, evidence=evidence)
    if ok:
        _flush()
    return ok


def confirm_proposals(conn, rel_ids: Union[str, Iterable[str]]) -> dict:
    """
    人工确认提案（单条/批量）：PENDING → CONFIRMED，即时进入可推演知识。

    Returns:
        {"succeeded": [已确认 rel_id, ...], "failed": [{rel_id, error}, ...]}
    """
    succeeded, failed = [], []
    for rid in _as_rel_id_list(rel_ids):
        try:
            if confirm_synapse(conn, rid):
                succeeded.append(rid)
            else:
                failed.append({"rel_id": rid, "error": "提案不存在"})
        except Exception as exc:
            failed.append({"rel_id": rid, "error": str(exc)})
    logger.info("人工确认提案: %d 成功 / %d 失败", len(succeeded), len(failed))
    if succeeded:
        _flush()
    return {"succeeded": succeeded, "failed": failed}


def reject_proposals(conn, rel_ids: Union[str, Iterable[str]]) -> dict:
    """
    人工否决提案（单条/批量）：PENDING → REJECTED。

    默认保留 REJECTED 审计痕迹（谁被否决过也是知识）；
    需要物理清除时另行调用 delete_proposals。
    """
    succeeded, failed = [], []
    for rid in _as_rel_id_list(rel_ids):
        try:
            if reject_synapse(conn, rid):
                succeeded.append(rid)
            else:
                failed.append({"rel_id": rid, "error": "提案不存在"})
        except Exception as exc:
            failed.append({"rel_id": rid, "error": str(exc)})
    logger.info("人工否决提案: %d 成功 / %d 失败", len(succeeded), len(failed))
    if succeeded:
        _flush()
    return {"succeeded": succeeded, "failed": failed}


def delete_proposals(conn, rel_ids: Union[str, Iterable[str]]) -> dict:
    """
    物理删除提案（单条/批量）：任何状态的突触均可删，端点概念保留。

    典型场景：「拒绝并删除」按钮（否决不留痕）与错误录入清理。
    """
    succeeded, failed = [], []
    for rid in _as_rel_id_list(rel_ids):
        try:
            if delete_synapse(conn, rid):
                succeeded.append(rid)
            else:
                failed.append({"rel_id": rid, "error": "提案不存在"})
        except Exception as exc:
            failed.append({"rel_id": rid, "error": str(exc)})
    logger.info("物理删除提案: %d 成功 / %d 失败", len(succeeded), len(failed))
    if succeeded:
        _flush()
    return {"succeeded": succeeded, "failed": failed}


if __name__ == "__main__":
    # 冒烟自检（独立临时运行时目录，不污染真实数据）：
    # 提案落库 → 同名复用/缺失新建 → 编辑调权 → 确认进入 CONFIRMED 子图
    # → 状态闸口拦截 → 否决 → 删除的全链路断言。
    import logging as _logging
    import shutil

    _logging.basicConfig(level=logging.INFO)

    # RUNTIME_DIR 在 persistence 模块导入时即已定型（此处 env setdefault 为时已晚），
    # 因此直接清除测试脑目录，保证每次自检都从干净状态起步。
    shutil.rmtree(_persistence.RUNTIME_DIR / "_intake_selftest", ignore_errors=True)

    from core.brain_manager import get_or_create
    from core.graph_cortex import get_confirmed_subgraph, init_schema

    session = get_or_create("_intake_selftest")
    conn = session.get_kuzu()
    init_schema(conn)

    # 预置一个既有概念，验证「同名复用」分支
    existing_id = add_concept(conn, name="恶寒", category="症状")

    proposals = [
        {"source": "恶寒", "target": "表实证", "relation": "suggests",
         "weight": 0.9, "evidence": "《伤寒论》第1条"},
        {"source": "表实证", "target": "咳嗽", "relation": "may_cause",
         "weight": 0.6, "evidence": "肺气不宣则咳"},
        {"source": "", "target": "空源头", "relation": "causes", "weight": 0.5},
    ]
    summary = commit_proposals(conn, proposals)
    assert summary["total"] == 3 and len(summary["committed"]) == 2, summary
    assert summary["failed"][0]["index"] == 2, "空源头提案应失败"
    assert summary["concepts_created"] == 2, "表实证/咳嗽 应新建"
    first = summary["committed"][0]
    assert first["src_id"] == existing_id and first["src_reused"], "恶寒 应复用既有概念"

    pending = list_pending(conn)
    assert len(pending) == 2, "应恰有 2 条 PENDING"

    rid_confirm, rid_edit = pending[0]["rel_id"], pending[1]["rel_id"]
    assert edit_proposal(conn, rid_edit, weight=0.8), "滑块调权应成功"
    res = confirm_proposals(conn, [rid_confirm, rid_edit])
    assert len(res["succeeded"]) == 2
    sub = get_confirmed_subgraph(conn)
    assert len(sub["edges"]) == 2, "确认后 2 条边应进入 CONFIRMED 子图"

    try:
        edit_proposal(conn, rid_confirm, weight=0.1)
        raise AssertionError("CONFIRMED 编辑应被拒绝")
    except ValueError:
        pass

    # 否决/删除需从 PENDING 出发（CONFIRMED 是终态，状态机会拦截）：
    # 再落一条全新提案，保持 PENDING 专供否决与物理删除测试。
    extra = commit_proposals(conn, [{
        "source": "咳嗽", "target": "肺气不宣", "relation": "suggests",
        "weight": 0.7, "evidence": " cough → lung qi constrained",
    }])
    assert len(extra["committed"]) == 1, "补充提案应成功（咳嗽 概念已存在应复用）"
    rid_reject = extra["committed"][0]["rel_id"]

    res = reject_proposals(conn, rid_reject)
    assert res["succeeded"] == [rid_reject]
    assert get_synapse(conn, rid_reject)["status"] == "REJECTED", "否决后保留 REJECTED 审计痕迹"

    res = delete_proposals(conn, rid_reject)
    assert res["succeeded"] == [rid_reject]
    assert get_synapse(conn, rid_reject) is None, "删除后应不可见"
    # 2 条 CONFIRMED 边不受否决/删除影响
    assert len(get_confirmed_subgraph(conn)["edges"]) == 2

    # ── 升级C：dialogue 对话回流通道 ──
    # 裁定：白名单自动转正仅限 intake；dialogue 即使高置信（weight=1.0 且
    # 证据充分）也一律 PENDING，转正权完全归人工；落库边与新建概念均带
    # knowledge_source=dialogue 溯源标记。
    dlg = commit_proposals(conn, [{
        "source": "自汗", "target": "卫气不固", "relation": "提示",
        "weight": 1.0, "evidence": "用户在对话中纠正：久病自汗属卫气不固",
    }], knowledge_source="dialogue")
    assert len(dlg["committed"]) == 1 and dlg["auto_confirmed"] == 0, dlg
    dlg_edge = dlg["committed"][0]
    assert dlg_edge["knowledge_source"] == "dialogue", "落库边应带对话回流标记"
    dlg_syn = get_synapse(conn, dlg_edge["rel_id"])
    assert dlg_syn["status"] == "PENDING", "对话回流提案绝不自动转正"
    assert dlg_syn["knowledge_source"] == "dialogue"
    dlg_pending = [p for p in list_pending(conn) if p["rel_id"] == dlg_edge["rel_id"]]
    assert dlg_pending, "对话回流提案应出现在待收纳池等人工处置"
    # 新建概念（自汗/卫气不固）继承 dialogue 归属；此前 intake 新建的概念不受影响
    from core.graph_cortex import get_concept
    assert get_concept(conn, dlg_edge["src_id"])["knowledge_source"] == "dialogue"
    assert get_concept(conn, dlg_edge["dst_id"])["knowledge_source"] == "dialogue"
    # 复用概念不改归属：dialogue 提案挂靠既有「恶寒」（intake 之前新建）不回写
    dlg2 = commit_proposals(conn, [{
        "source": "恶寒", "target": "里虚寒", "relation": "提示",
        "weight": 0.5, "evidence": "用户追问中提及",
    }], knowledge_source="dialogue")
    assert get_concept(conn, dlg2["committed"][0]["src_id"])["knowledge_source"] != "dialogue", \
        "复用概念绝不改写原归属"
    # 非法来源入口拦截
    try:
        commit_proposals(conn, proposals[:1], knowledge_source="hacker")
        raise AssertionError("非法 knowledge_source 应被拦截")
    except ValueError:
        pass

    # ── 升级D：来源分级审查（仅 dialogue 通道） ──
    # 前置：手造一条书籍类 CONFIRMED 边（intake 亲笔，绕开白名单阈值不确定性）；
    # 注意：name 允许同脑重复（铁律），既有「恶寒」必须按名解析复用，
    # 绝不能 add_concept 再造一个同名词概念（否则审查端点对错位不命中）。
    b_a = add_concept(conn, name="太阳伤寒", category="证型")
    b_b = next(c["id"] for c in list_concepts(conn) if c["name"] == "恶寒")
    book_rid = add_synapse(conn, b_a, b_b, "表现为", weight=0.9,
                           evidence="《伤寒论》第3条", status="CONFIRMED",
                           knowledge_source="intake")
    # ① 书籍类命中 → 入池前拒绝：committed 空、refused 记录既有边溯源
    d_refuse = commit_proposals(conn, [{
        "source": "太阳伤寒", "target": "恶寒", "relation": "表现为",
        "weight": 0.9, "evidence": "用户对话中试图改口：太阳伤寒不恶寒",
    }], knowledge_source="dialogue")
    assert d_refuse["committed"] == [] and len(d_refuse["refused"]) == 1, d_refuse
    _r = d_refuse["refused"][0]
    assert _r["existing_rel_id"] == book_rid and _r["existing_source"] == "intake", _r
    assert get_synapse(conn, book_rid)["knowledge_source"] == "intake", \
        "书籍类知识归属不得被对话改写"
    # ② 经验类命中照常入池：手造 manual CONFIRMED 边（用户亲笔经验）
    m_a = add_concept(conn, name="动则汗出", category="症状")
    m_b = add_concept(conn, name="气虚", category="病机")
    add_synapse(conn, m_a, m_b, "提示", weight=0.8, evidence="人工手录经验",
                status="CONFIRMED", knowledge_source="manual")
    d_exp = commit_proposals(conn, [{
        "source": "动则汗出", "target": "气虚", "relation": "提示",
        "weight": 0.7, "evidence": "对话补充：动则汗出亦提示气虚",
    }], knowledge_source="dialogue")
    assert d_exp["refused"] == [] and len(d_exp["committed"]) == 1, d_exp
    assert get_synapse(conn, d_exp["committed"][0]["rel_id"])["status"] == "PENDING"
    # ③ 其他通道免审：intake 低置信提案命中同一书籍类 CONFIRMED 边 → 照常 PENDING
    d_other = commit_proposals(conn, [{
        "source": "太阳伤寒", "target": "恶寒", "relation": "表现为",
        "weight": 0.5, "evidence": "低置信速读提案免审对照",
    }])
    assert d_other["refused"] == [] and len(d_other["committed"]) == 1, d_other
    assert get_synapse(conn, d_other["committed"][0]["rel_id"])["status"] == "PENDING"
    print("升级D 来源分级审查 ✓  书籍类拒改 / 经验类放行 / 其他通道免审")

    print("intake 冒烟自检全部通过 ✓")

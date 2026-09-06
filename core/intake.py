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

核心铁律（与 Readme/Todo 对齐）：
- 提案落库一律 PENDING：LLM 产出永远只是「待收纳」候选，
  CONFIRMED 仅可由人工处置路径触发，任何自动流程不得越权；
- 概念名解析绝不自动合并：同名多概念时取字典序第一个复用，
  并在返回中标注 reused 供前端弱提示（绝不合并、绝不改名）；
- 单条失败不拖垮整批：逐条独立落库，失败项连同原因一并返回；
- 每次成功写操作后触发 sync_to_storage 原子刷盘，
  刷盘失败仅告警不阻断（本地工作副本不受影响，/data 保留旧镜像）。
"""

import logging
from typing import Iterable, List, Optional, Tuple, Union

from storage import persistence as _persistence
from core.graph_cortex import (
    add_concept,
    add_synapse,
    confirm_synapse,
    delete_synapse,
    get_synapse,
    list_concepts,
    reject_synapse,
    update_synapse,
)

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

def _find_or_create_concept(conn, name: str, category: str = "general") -> Tuple[str, bool]:
    """
    按概念名精确查找；命中则复用（返回既有 id），缺失则新建（返回新 id）。

    同名多概念的取舍：list_concepts 按 name 字典序稳定排序，取第一个 ——
    这不是「合并」而是「挂靠」：速读秘书只知道名字，不知道病机细节，
    概念身份主权始终归人；前端可依据 reused 标注弹出弱提示让用户改挂。

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
    return add_concept(conn, name=name, category=category), True


def _resolve_concept(conn, name: str, cache: dict) -> Tuple[str, bool]:
    """带批内缓存的概念解析：同一批提案中同名概念只建/查一次。"""
    if name in cache:
        return cache[name], False
    cid, created = _find_or_create_concept(conn, name)
    cache[name] = cid
    return cid, created


# ───────────────────────────────────────────────
# 提案批量落库 (Task 4.2.3a)
# ───────────────────────────────────────────────

def commit_proposals(conn, proposals: List[dict]) -> dict:
    """
    将速读秘书产出的提案批量落库为 PENDING 突触。

    处理规则：
      - 逐条独立落库：单条失败（空字段/数值非法/写库异常）仅记入 failed，
        不拖垮整批 —— 但落库后全部处于 PENDING，必须经人工确认才生效；
      - 概念解析：批内同名缓存复用；跨批次靠库内同名精确匹配复用；
        两级都未命中才真正新建（category 统一 general，人工可后改）；
      - weight 越界值在此 clamp 进 [0,1]（与 metabolism 层净化双保险）。

    Args:
        conn:      Kùzu Connection。
        proposals: [{source, target, relation, weight, evidence}, ...]

    Returns:
        {"total": 提案总数,
         "committed": [{rel_id, src_id, src_name, dst_id, dst_name,
                        src_reused, dst_reused, relation, weight, evidence}, ...],
         "failed":    [{index, error}, ...],
         "concepts_created": 本次新建概念数,
         "concepts_reused":  本次复用既有概念数}
    """
    summary: dict = {
        "total": len(proposals),
        "committed": [],
        "failed": [],
        "concepts_created": 0,
        "concepts_reused": 0,
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

            src_id, src_new = _resolve_concept(conn, src_name, cache)
            dst_id, dst_new = _resolve_concept(conn, dst_name, cache)
            summary["concepts_created"] += int(src_new) + int(dst_new)
            summary["concepts_reused"] += int(not src_new) + int(not dst_new)

            rel_id = add_synapse(
                conn, src_id, dst_id, relation,
                weight=weight, evidence=evidence,
                status="PENDING",  # 铁律：提案一律待收纳，人工确认才转正
            )
            summary["committed"].append({
                "rel_id": rel_id,
                "src_id": src_id, "src_name": src_name, "src_reused": not src_new,
                "dst_id": dst_id, "dst_name": dst_name, "dst_reused": not dst_new,
                "relation": relation, "weight": weight, "evidence": evidence,
            })
        except Exception as exc:
            logger.warning("提案 #%d 落库失败: %s", i, exc)
            summary["failed"].append({"index": i, "error": str(exc)})

    logger.info(
        "提案落库完成: %d/%d 成功（新建概念 %d / 复用概念 %d）",
        len(summary["committed"]), summary["total"],
        summary["concepts_created"], summary["concepts_reused"],
    )
    _flush()
    return summary


# ───────────────────────────────────────────────
# 待收纳池查询与人工处置 (Task 4.2.3b)
# ───────────────────────────────────────────────

def list_pending(conn) -> list[dict]:
    """
    列出待收纳池全部 PENDING 突触（含端点概念名，供审核面板直读直显）。

    Returns:
        [{rel_id, src_id, src_name, dst_id, dst_name,
          weight, relation, evidence}, ...]，按 rel_id 稳定排序。
    """
    result = conn.execute(
        "MATCH (a:Concept)-[s:SYNAPSE]->(b:Concept) WHERE s.status = 'PENDING' "
        "RETURN s.rel_id, a.id, a.name, b.id, b.name, "
        "s.weight, s.relation, s.evidence ORDER BY s.rel_id"
    )
    pending: list[dict] = []
    while result.has_next():
        row = result.get_next()
        pending.append({
            "rel_id": row[0],
            "src_id": row[1], "src_name": row[2],
            "dst_id": row[3], "dst_name": row[4],
            "weight": row[5], "relation": row[6], "evidence": row[7],
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

    print("intake 冒烟自检全部通过 ✓")

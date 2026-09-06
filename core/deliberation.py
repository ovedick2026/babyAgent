"""
SynapseMind - 增强模式分步人工判定状态机 (core/deliberation.py)

Task 4.3.2: 模式 3（AI 增强）第二环节 —— 会话级判定状态机与结论固化。

链路定位：
    Task 4.3.1 plan_for_brain() 产出分步辨证计划
      → create_session()     计划固化为判定会话，逐步骤跟踪状态
      → execute_step()       LLM 结合 CONFIRMED 图谱 + 已确认断言产出候选
      → judge_step()         人工裁决：采纳 ADOPTED / 修改 MODIFIED / 否决 REJECTED
      → finalize_session()   全部步骤终态后，固化最终结论 + 完整证据链

核心铁律：
- 候选与结论都只是「提案」，CONFIRMED 与否永远由人工裁决触发，无任何自动入库；
- 证据链由已确认步骤确定性拼装，杜绝二次加工失真，逐条可回溯到步骤序号；
- 结论文本仅以已确认断言为唯一素材，产出为空即整体报错、会话原样保留可重试，
  绝不部分采用 —— 与速读秘书（Task 4.2.2）、规划器（Task 4.3.1）同一净化哲学；
- 终态不可撤销：判定有误就否决该步或另建会话，严禁暗中改写历史。
"""

import json
import logging
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from core.metabolism import MetabolismError, chat_completion
from core.reasoner import (
    MAX_CLUES_PER_STEP,
    MAX_CLUE_LEN,
    MAX_GOAL_LEN,
    MAX_STEPS,
    build_graph_context,
    plan_steps,
)

logger = logging.getLogger("synapsemind.deliberation")

# 步骤状态机：READY(未执行) → PENDING(候选待判定) → ADOPTED/MODIFIED/REJECTED(终态，仅人工触发)
S_READY, S_PENDING = "READY", "PENDING"
S_ADOPTED, S_MODIFIED, S_REJECTED = "ADOPTED", "MODIFIED", "REJECTED"
_TERMINAL = {S_ADOPTED, S_MODIFIED, S_REJECTED}
_CONFIRMED = {S_ADOPTED, S_MODIFIED}

# 会话状态：ACTIVE → FINALIZED（结论固化后整体只读，仅可查询/关闭）
SES_ACTIVE, SES_FINALIZED = "ACTIVE", "FINALIZED"

MAX_CANDIDATE_LEN = 2000    # 单步候选 / 人工修订文本上限
MAX_CONCLUSION_LEN = 4000   # 最终结论文本上限

_EXECUTOR_SYSTEM_PROMPT = (
    "你是 SynapseMind 的辨证执行者。依据「已知图谱知识」「已确认断言」与「所需线索」，"
    "对当前步骤目标给出简明、可判定的意见。只依据给定知识推理，禁止臆造不存在的概念。"
    "直接输出意见文本（3 句以内），禁止 JSON、禁止寒暄。"
)

_CONCLUDER_SYSTEM_PROMPT = (
    "你是 SynapseMind 的结论整理者。仅依据给定的「已确认断言」，为用户原始诉求撰写最终结论。"
    "结论必须逐条落实断言，禁止引入断言之外的任何信息。直接输出结论文本，禁止寒暄。"
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class DeliberationError(RuntimeError):
    """判定会话流程错误（会话不存在 / 状态非法 / LLM 产出失败等）。"""


def _strip_fences(text: str) -> str:
    """剥除 markdown 代码围栏与首尾空白。"""
    return _FENCE_RE.sub("", (text or "").strip()).strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# 会话注册表：进程内 session_id → 会话字典（与 brain_manager 注册表同风格）
_sessions: dict = {}
_lock = threading.RLock()


# ───────────────────────────────────────────────
# 会话生命周期
# ───────────────────────────────────────────────

def _normalize_steps(steps) -> List[dict]:
    """计划步骤净化：白名单字段、逐条校验，非法步骤整条丢弃；兼容 plan_steps 产物。"""
    if isinstance(steps, dict):
        steps = steps.get("steps")
    normalized: List[dict] = []
    for item in list(steps or [])[:MAX_STEPS]:
        if not isinstance(item, dict):
            continue
        goal, clues = item.get("goal"), item.get("clues")
        if not isinstance(goal, str) or not goal.strip():
            continue
        if not isinstance(clues, list):
            continue
        clean = [c.strip()[:MAX_CLUE_LEN] for c in clues
                 if isinstance(c, str) and c.strip()][:MAX_CLUES_PER_STEP]
        if not clean:
            continue  # 无线索的步骤不可执行，整条剔除
        normalized.append({"goal": goal.strip()[:MAX_GOAL_LEN], "clues": clean})
    if not normalized:
        raise ValueError("计划中不含任何有效步骤，无法创建判定会话")
    return normalized


def create_session(brain_id: str, clue: str, steps, *, graph_context: Optional[dict] = None) -> str:
    """把 Task 4.3.1 的计划固化为判定会话，返回 session_id。"""
    if not isinstance(brain_id, str) or not brain_id.strip():
        raise ValueError("brain_id 不得为空")
    if not isinstance(clue, str) or not clue.strip():
        raise ValueError("用户诉求不得为空")
    normalized = _normalize_steps(steps)

    session_id = uuid.uuid4().hex
    sess = {
        "session_id": session_id,
        "brain_id": brain_id.strip(),
        "clue": clue.strip(),
        "status": SES_ACTIVE,
        "created_at": _now(),
        "graph_context": graph_context or {"nodes": [], "edges": []},
        "steps": [
            {"index": i, "goal": s["goal"], "clues": s["clues"],
             "status": S_READY, "candidate": None, "final_text": None,
             "judged_at": None}
            for i, s in enumerate(normalized)
        ],
        "result": None,
    }
    with _lock:
        _sessions[session_id] = sess
    logger.info("创建判定会话 %s: brain=%s 步骤=%d", session_id, brain_id, len(normalized))
    return session_id


def start_for_brain(brain_id: str, clue: str, *, timeout: float = 60.0) -> str:
    """便捷入口：规划（4.3.1）→ 建会话（4.3.2）一气呵成，返回 session_id。"""
    plan = plan_steps(clue, build_graph_context(brain_id, clue), timeout=timeout)
    return create_session(brain_id, clue, plan["steps"])


def _get_locked(session_id: str) -> dict:
    sess = _sessions.get(session_id)
    if sess is None:
        raise DeliberationError(f"判定会话不存在: {session_id}")
    return sess


def get_session(session_id: str) -> dict:
    with _lock:
        return _get_locked(session_id)


def list_sessions(brain_id: Optional[str] = None) -> List[dict]:
    """列出会话摘要（可按 brain 过滤），供前端会话列表使用。"""
    with _lock:
        return [
            {"session_id": s["session_id"], "brain_id": s["brain_id"],
             "clue": s["clue"], "status": s["status"],
             "resolved": sum(1 for st in s["steps"] if st["status"] in _TERMINAL),
             "total": len(s["steps"])}
            for s in _sessions.values() if brain_id in (None, s["brain_id"])
        ]


def next_pending(session_id: str) -> Optional[int]:
    """返回第一个待处理（READY/PENDING）步骤的下标；全部终态时返回 None。"""
    with _lock:
        sess = _get_locked(session_id)
        for st in sess["steps"]:
            if st["status"] in (S_READY, S_PENDING):
                return st["index"]
        return None


def close_session(session_id: str) -> None:
    """关闭并移除会话（结论固化与否均可主动清理）。"""
    with _lock:
        if _sessions.pop(session_id, None) is not None:
            logger.info("关闭并移除判定会话 %s", session_id)


# ───────────────────────────────────────────────
# 步骤执行与人工判定
# ───────────────────────────────────────────────

def _format_knowledge(ctx: dict) -> dict:
    """CONFIRMED 子图 → LLM 友好的紧凑摘要（与规划器同款）。"""
    id2name = {n["id"]: n.get("name", "") for n in ctx.get("nodes", [])}
    relations = []
    for e in ctx.get("edges", []):
        src = id2name.get(e.get("src_id"), e.get("src_id", ""))
        dst = id2name.get(e.get("dst_id"), e.get("dst_id", ""))
        relations.append(
            f'{src} --{e.get("relation", "")}({float(e.get("weight", 0)):.2f})--> {dst}'
        )
    return {
        "概念": [f'{n.get("name", "")}({n.get("category", "")})' for n in ctx.get("nodes", [])],
        "已确认关系": relations,
    }


def _confirmed_assertions(sess: dict) -> List[dict]:
    """已确认（采纳/修改）步骤 → 断言列表（确定性，顺序即步骤序）。"""
    return [
        {"步骤": st["index"] + 1, "目标": st["goal"], "断言": st["final_text"]}
        for st in sess["steps"] if st["status"] in _CONFIRMED
    ]


def execute_step(session_id: str, step_index: int, *, timeout: float = 60.0) -> dict:
    """为指定步骤产出候选意见（LLM + CONFIRMED 图谱 + 已确认断言），步骤转 PENDING。"""
    with _lock:
        sess = _get_locked(session_id)
        if sess["status"] != SES_ACTIVE:
            raise DeliberationError("会话已固化结论，禁止再执行步骤")
        if not 0 <= step_index < len(sess["steps"]):
            raise ValueError(f"步骤下标越界: {step_index}")
        st = sess["steps"][step_index]
        if st["status"] in _TERMINAL:
            raise DeliberationError(f"步骤 {step_index} 已终态（{st['status']}），不可重复执行")
        snapshot = {
            "clue": sess["clue"],
            "goal": st["goal"],
            "clues": list(st["clues"]),
            "knowledge": _format_knowledge(sess["graph_context"]),
            "assertions": _confirmed_assertions(sess),
        }

    payload = json.dumps({
        "用户诉求": snapshot["clue"],
        "当前步骤": {"目标": snapshot["goal"], "所需线索": snapshot["clues"]},
        "已知图谱知识": snapshot["knowledge"],
        "已确认断言": snapshot["assertions"] or "（暂无，本步为首个判定步骤）",
    }, ensure_ascii=False)
    try:
        raw = chat_completion(
            [{"role": "system", "content": _EXECUTOR_SYSTEM_PROMPT},
             {"role": "user", "content": payload}],
            timeout=timeout,
        )
    except MetabolismError as exc:
        raise DeliberationError(f"步骤候选产出失败: {exc}") from exc

    candidate = _strip_fences(raw)
    if not candidate:
        logger.error("步骤执行返回空候选，原始返回已记录: %r", raw)
        raise DeliberationError("步骤候选产出为空，会话未受影响，请重试")

    with _lock:
        sess = _get_locked(session_id)
        st = sess["steps"][step_index]
        if sess["status"] != SES_ACTIVE or st["status"] in _TERMINAL:
            raise DeliberationError("会话/步骤状态在执行期间发生变化，本次候选作废")
        st["candidate"] = candidate[:MAX_CANDIDATE_LEN]
        st["status"] = S_PENDING
    logger.info("会话 %s 步骤 %d 候选已产出", session_id, step_index)
    return st


def judge_step(session_id: str, step_index: int, action: str, *, revised: Optional[str] = None) -> dict:
    """
    人工裁决指定步骤（唯一合法的终态迁移入口）。

    action:
        adopt  — 采纳候选原文；
        modify — 采纳人工修订文本（必须提供非空 revised）；
        reject — 否决该步（不进入结论与证据链）。
    """
    action = (action or "").strip().lower()
    if action not in ("adopt", "modify", "reject"):
        raise ValueError("action 必须为 adopt / modify / reject")

    with _lock:
        sess = _get_locked(session_id)
        if sess["status"] != SES_ACTIVE:
            raise DeliberationError("会话已固化结论，禁止再判定")
        if not 0 <= step_index < len(sess["steps"]):
            raise ValueError(f"步骤下标越界: {step_index}")
        st = sess["steps"][step_index]
        if st["status"] in _TERMINAL:
            raise ValueError(f"步骤 {step_index} 已终态（{st['status']}），判定不可撤销")

        if action == "adopt":
            if not st["candidate"]:
                raise DeliberationError("该步骤尚无候选可采纳，请先 execute_step")
            st["final_text"] = st["candidate"]
            st["status"] = S_ADOPTED
        elif action == "modify":
            if not isinstance(revised, str) or not revised.strip():
                raise ValueError("modify 动作必须提供非空 revised 文本")
            st["final_text"] = revised.strip()[:MAX_CANDIDATE_LEN]
            st["status"] = S_MODIFIED
        else:  # reject
            st["final_text"] = None
            st["status"] = S_REJECTED
        st["judged_at"] = _now()
        logger.info("会话 %s 步骤 %d 人工判定 → %s", session_id, step_index, st["status"])
        return st


# ───────────────────────────────────────────────
# 结论固化
# ───────────────────────────────────────────────

def finalize_session(session_id: str, *, timeout: float = 90.0) -> dict:
    """
    全部步骤终态后固化结论：证据链确定性拼装自已确认步骤，
    结论文本由 LLM 仅依据断言合成；失败时会话保持 ACTIVE 可重试。
    """
    with _lock:
        sess = _get_locked(session_id)
        if sess["status"] == SES_FINALIZED:
            return sess["result"]  # 幂等：重复固化直接返回既有结论
        unresolved = [st["index"] for st in sess["steps"] if st["status"] not in _TERMINAL]
        if unresolved:
            raise DeliberationError(f"尚有未判定步骤 {unresolved}，全部终态后才能固化结论")
        assertions = _confirmed_assertions(sess)
        if not assertions:
            raise DeliberationError("所有步骤均被否决，无已确认断言，无法产出结论")
        snapshot = {"clue": sess["clue"], "assertions": assertions}

    payload = json.dumps(
        {"用户诉求": snapshot["clue"], "已确认断言": snapshot["assertions"]},
        ensure_ascii=False,
    )
    try:
        raw = chat_completion(
            [{"role": "system", "content": _CONCLUDER_SYSTEM_PROMPT},
             {"role": "user", "content": payload}],
            timeout=timeout,
        )
    except MetabolismError as exc:
        raise DeliberationError(f"结论固化失败（会话保留可重试）: {exc}") from exc

    conclusion = _strip_fences(raw)
    if not conclusion:
        logger.error("结论固化的 LLM 返回为空，会话保持可重试，原始返回: %r", raw)
        raise DeliberationError("结论产出为空，会话未受影响，可重试固化")

    with _lock:
        sess = _get_locked(session_id)
        if sess["status"] == SES_FINALIZED:  # 并发双跑：后到者复用先到结果
            return sess["result"]
        rejected = [{"step": st["index"], "goal": st["goal"]}
                    for st in sess["steps"] if st["status"] == S_REJECTED]
        evidence_chain = [
            {"step": st["index"], "goal": st["goal"],
             "assertion": st["final_text"], "origin": st["status"]}
            for st in sess["steps"] if st["status"] in _CONFIRMED
        ]
        result = {
            "conclusion": conclusion[:MAX_CONCLUSION_LEN],
            "evidence_chain": evidence_chain,
            "rejected": rejected,
            "finalized_at": _now(),
        }
        sess["result"] = result
        sess["status"] = SES_FINALIZED
    logger.info("会话 %s 结论已固化: 断言=%d 否决=%d", session_id, len(evidence_chain), len(rejected))
    return result


if __name__ == "__main__":
    # 零网络冒烟自检：monkeypatch 同步客户端，覆盖状态机全部分支。
    import logging as _logging

    _logging.basicConfig(level=logging.INFO)
    _self = sys.modules[__name__]
    _orig = _self.chat_completion

    ctx = {
        "nodes": [{"id": "a", "name": "恶寒", "category": "症状", "description": ""}],
        "edges": [],
    }
    plan = [
        {"goal": "判断表虚表实", "clues": ["恶寒", "汗出与否"]},
        {"goal": "定方", "clues": ["证型"]},
        {"goal": "核对禁忌", "clues": ["孕妇"]},
    ]
    sid = create_session("brain_ut", "恶寒发热无汗如何处方", plan, graph_context=ctx)

    # 1) 未判定完就固化 → 拦截
    try:
        finalize_session(sid)
        raise AssertionError("未完成判定不应允许固化")
    except DeliberationError as exc:
        print("未完成拦截 ✓ ", exc)

    # 2) 无候选直接采纳 → 拦截
    try:
        judge_step(sid, 0, "adopt")
        raise AssertionError("无候选不应可采纳")
    except DeliberationError as exc:
        print("无候选拦截 ✓ ", exc)

    # 3) 逐步执行 → 采纳 / 修改 / 否决（候选带脏围栏也应被清洗）
    _self.chat_completion = lambda msgs, **kw: "恶寒无汗脉浮紧，属表实证。\n```"
    execute_step(sid, 0)
    judge_step(sid, 0, "adopt")
    execute_step(sid, 1)
    judge_step(sid, 1, "modify", revised="表实证宜麻黄汤类方")
    execute_step(sid, 2)
    judge_step(sid, 2, "reject")

    summary = [s for s in list_sessions("brain_ut") if s["session_id"] == sid][0]
    assert summary["resolved"] == 3 and summary["total"] == 3, summary
    print("三态判定 ✓  resolved = 3/3")

    # 4) 终态锁定与非法动作
    try:
        judge_step(sid, 0, "adopt")
        raise AssertionError("终态步骤不应可重复判定")
    except ValueError as exc:
        print("终态锁定 ✓ ", exc)
    try:
        judge_step(sid, 1, "auto")
        raise AssertionError("非法动作应报错")
    except ValueError as exc:
        print("非法动作拦截 ✓ ", exc)

    # 5) 结论 LLM 返回空白 → 报错且会话保持 ACTIVE 可重试
    _self.chat_completion = lambda msgs, **kw: "   "
    try:
        finalize_session(sid)
        raise AssertionError("空白结论应报错")
    except DeliberationError as exc:
        print("空白结论拦截 ✓ ", exc)
    sess = get_session(sid)
    assert sess["status"] == SES_ACTIVE and sess["result"] is None
    print("失败可重试 ✓  会话仍为 ACTIVE")

    # 6) 正常固化：证据链只含已确认步骤、否决步单独登记、重复固化幂等
    _self.chat_completion = lambda msgs, **kw: "综合已确认断言：恶寒无汗属表实证，宜麻黄汤类方发汗解表。"
    result = finalize_session(sid)
    assert [e["step"] for e in result["evidence_chain"]] == [0, 1], result
    assert result["evidence_chain"][1]["assertion"] == "表实证宜麻黄汤类方"
    assert result["rejected"] == [{"step": 2, "goal": "核对禁忌"}]
    assert get_session(sid)["status"] == SES_FINALIZED
    assert finalize_session(sid) is result
    print("结论固化 ✓  conclusion =", result["conclusion"])

    # 7) 全否决 → 无法固化
    sid2 = create_session("brain_ut", "无关请求", [{"goal": "g", "clues": ["c"]}])
    execute_step(sid2, 0)
    judge_step(sid2, 0, "reject")
    try:
        finalize_session(sid2)
        raise AssertionError("全否决不应产出结论")
    except DeliberationError as exc:
        print("全否决拦截 ✓ ", exc)

    close_session(sid)
    close_session(sid2)
    _self.chat_completion = _orig
    print("deliberation 冒烟自检全部通过 ✓")

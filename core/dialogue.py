"""
SynapseMind - 多轮对话推演会话层 (core/dialogue.py)

升级B·多轮对话推演：推演保留对话上下文与追问链（为什么/还有呢），
激活支持增量扩散 —— 本模块是会话与轮次快照的唯一权威存储。

链路定位：
    POST /api/dialogue (B4 路由)
      → create_session()   开启会话（一个会话 = 一条追问链）
      → append_turn()      每轮落快照：意图 / 融合种子 / 推演产物摘要
      → last_turn()        B2 种子融合与 B3 why 回溯上一轮的入口
      → close_session()    用户主动关闭（显式删除，无隐藏终态）

核心铁律（与 deliberation / inference 同源）：
- 会话层只做上下文记忆与快照存取：推演真相永远由当轮 CONFIRMED 子图
  重扩散得出（个人级小图 × 3 跳，重跑代价可忽略），本层绝不缓存图谱、
  绝不参与任何推演计算 —— 与「图谱真相以 Kùzu 为准」同一条红线；
- 快照不可变：turn 一经 append 不得改写（无编辑/回滚 API），错问就
  继续追问或关闭会话重来，杜绝暗中改写历史；
- insufficient 轮同样入快照（保留 clue/seeds 供追问重试），但
  追问链以「最近一次 ok 轮」为基准，失败轮不阻断追问。

存储形态：进程内注册表（同 deliberation 风格）。对话会话是易失的
交互上下文而非知识资产，重启即清空属预期行为，不落盘、不进 /data。
"""

import copy
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger("synapsemind.dialogue")


class DialogueError(RuntimeError):
    """对话会话流程错误（会话不存在 / 轮数超限等）。"""


# 会话规模上限（防遗忘关闭的僵尸会话拖垮内存）
MAX_SESSIONS = 64    # 全局会话数；超限自动淘汰最旧会话
MAX_TURNS = 50       # 单会话轮数；超限要求关闭后另起会话

# B3 意图标签白名单（本层只校验，解析逻辑在 B3）
INTENT_NORMAL, INTENT_WHY, INTENT_MORE = "normal", "why", "more"
_INTENTS = (INTENT_NORMAL, INTENT_WHY, INTENT_MORE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# 会话注册表：session_id → 会话字典（结构见 create_session）
_sessions: dict = {}
_lock = threading.RLock()


# ───────────────────────────────────────────────
# 会话生命周期
# ───────────────────────────────────────────────

def create_session(brain_id: str, title: str = "") -> str:
    """开启对话会话，返回 session_id。

    Args:
        brain_id: 归属脑（当前版本单脑 default，字段保留多脑扩展位）。
        title:    可选会话标题；缺省时取首轮 clue 自动补全。
    """
    if not isinstance(brain_id, str) or not brain_id.strip():
        raise ValueError("brain_id 不得为空")

    with _lock:
        # 容量守卫：超限淘汰 created_at 最旧的会话（FIFO，交互上下文可弃）
        if len(_sessions) >= MAX_SESSIONS:
            oldest = min(_sessions, key=lambda s: _sessions[s]["created_at"])
            del _sessions[oldest]
            logger.info("对话会话超限(%d)，淘汰最旧: %s", MAX_SESSIONS, oldest)

        session_id = uuid.uuid4().hex
        _sessions[session_id] = {
            "session_id": session_id,
            "brain_id": brain_id.strip(),
            "title": (title or "").strip()[:128],
            "created_at": _now(),
            "turns": [],   # 每轮快照，结构见 append_turn
        }
    logger.info("创建对话会话 %s: brain=%s", session_id, brain_id)
    return session_id


def _get_locked(session_id: str) -> dict:
    sess = _sessions.get(session_id)
    if sess is None:
        raise DialogueError(f"对话会话不存在: {session_id}")
    return sess


def get_session(session_id: str) -> dict:
    """取完整会话（含全部轮次快照；深拷贝防调用方篡改内部状态）。"""
    with _lock:
        return copy.deepcopy(_get_locked(session_id))


def list_sessions(brain_id: Optional[str] = None) -> List[dict]:
    """会话摘要列表（新→旧；可按 brain 过滤），供前端会话选择器使用。"""
    with _lock:
        matched = [s for s in _sessions.values()
                   if brain_id is None or s["brain_id"] == brain_id]
        matched.sort(key=lambda s: (s["created_at"], s["session_id"]),
                     reverse=True)
        return [
            {
                "session_id": s["session_id"],
                "brain_id": s["brain_id"],
                "title": s["title"],
                "created_at": s["created_at"],
                "turn_count": len(s["turns"]),
                "last_clue": s["turns"][-1]["clue"] if s["turns"] else None,
                "last_status": s["turns"][-1]["status"] if s["turns"] else None,
            }
            for s in matched
        ]


def close_session(session_id: str) -> bool:
    """关闭并移除会话（用户显式丢弃追问链）；不存在返回 False（幂等）。"""
    with _lock:
        if session_id not in _sessions:
            return False
        del _sessions[session_id]
    logger.info("关闭对话会话: %s", session_id)
    return True


# ───────────────────────────────────────────────
# 轮次快照
# ───────────────────────────────────────────────

def append_turn(session_id: str, *, clue: str, intent: str,
                seeds: dict, result: dict) -> dict:
    """追加一轮快照并返回该轮（含分配的 turn_index）。

    Args:
        clue:   用户本轮输入原文（快捷追问按钮的话术同样算输入）。
        intent: B3 解析出的意图标签（normal/why/more）。
        seeds:  本轮最终种子 {concept_id: weight}（B2 融合后的产物）。
        result: pure_reason 产物（status/answer/activations 全量快照）。
    """
    clue = str(clue or "").strip()
    if not clue:
        raise ValueError("clue 不得为空")
    if intent not in _INTENTS:
        raise ValueError(f"非法意图: {intent!r}（合法值: {_INTENTS}）")

    with _lock:
        sess = _get_locked(session_id)
        if len(sess["turns"]) >= MAX_TURNS:
            raise DialogueError(
                f"会话轮数已达上限({MAX_TURNS})，请关闭后另起会话")
        result = result or {}
        turn = {
            "turn_index": len(sess["turns"]),
            "clue": clue[:256],
            "intent": intent,
            "seeds": {str(k): float(v) for k, v in dict(seeds or {}).items()},
            "status": str(result.get("status", "insufficient")),
            "message": str(result.get("message", "")),
            "answer": copy.deepcopy(result.get("answer")),
            "activations": copy.deepcopy(result.get("activations", [])),
            "created_at": _now(),
        }
        sess["turns"].append(turn)
        # 空标题以首轮 clue 自动补全（列表页一眼可辨）
        if not sess["title"]:
            sess["title"] = turn["clue"][:64]
    return copy.deepcopy(turn)


# ── 升级B2：增量扩散 · 余温种子融合 ──
# 「还有呢」语义：上轮激活的高能概念作为本轮二级种子，衰减后并入新种子
# 重新扩散。图规模小（≤3 跳 CONFIRMED 子图），子图重跑代价可忽略且
# 确定性结果与真增量扩散一致（复用 pure_reason 单一实现，不另起炉灶）。
CARRYOVER_TOP_N = 3    # 上轮激活序列取前 N 个高能节点作余温种子
CARRYOVER_DECAY = 0.5  # 余温衰减系数（余温权重 = 上轮峰值能量 × decay）


def carryover_seeds(turn: Optional[dict], *, top_n: int = CARRYOVER_TOP_N,
                    decay: float = CARRYOVER_DECAY,
                    exclude_ids: Optional[set] = None) -> dict:
    """从上一轮快照提取余温种子：TopN 高能节点峰值能量 × 衰减系数。

    能量取峰值（peak）而非终态（energy）：波传播语义下节点能量当跳
    全部向下游传导，种子/枢纽节点终态常归零（inference 激活序列按
    peak 降序正是为此）；若按终态提取，单结论小图的余温会被整支掏空。
    快照缺 peak（手工构造桩）时回退终态 energy。

    Args:
        turn:   append_turn 落下的轮次快照（通常来自 last_turn()）。
        top_n:  截取激活序列前 N 个（序列本就按峰值降序，见 inference）。
        decay:  衰减系数（0~1）；0 等价于关闭余温。
        exclude_ids: 排除的概念 id 集合（B4：排除上轮结论——结论多为
            端点无出边，从它续扩散必然空转，跳过它取次级高能节点）。
            过滤先于 TopN 截取：排除后仍取足 N 个合格节点。

    Returns:
        {concept_id: 余温权重}；非 ok 轮 / 激活为空 / 全部能量归零时
        返回空 dict（调用方零余温，退化为普通首轮推演）。
    """
    if not turn or turn.get("status") != "ok":
        return {}
    banned = {str(x) for x in (exclude_ids or ())}
    acts = turn.get("activations") or []
    seeds: dict = {}
    picked = 0
    for act in acts:
        if picked >= max(0, int(top_n)):
            break
        cid = str(act.get("id") or "").strip()
        if not cid or cid in banned:
            continue
        base = act.get("peak")
        if base is None:
            base = act.get("energy")
        try:
            w = float(base or 0.0) * float(decay)
        except (TypeError, ValueError):
            continue
        if 0.0 < w <= 1.0:  # 零峰值（未真激活）与越界值不产生余温
            seeds[cid] = w
            picked += 1
    return seeds


# ── 升级B3：追问意图解析 ──
# why（为什么）与 more（还有呢）是纯前端追问语义，由 B4 路由调用本函数
# 归一化为推演入参；本函数只读快照、绝不改写轮次历史。

def resolve_followup(session_id: str, intent: str, clue: str = "") -> dict:
    """解析追问意图 → 推演入参（clue / extra_seeds / 引用轮）。

    - why（为什么）：回溯上一条 ok 轮结论的证据链，附 answer 快照，
      零图谱调用——依据链在落轮时已确定性快照（杜绝图谱后续变动导致
      「解释与当初答案不一致」）。
    - more（还有呢）：取上一条 ok 轮的余温种子（TopN × 衰减）作增量扩散，
      clue 允许为空（纯「还有呢」靠余温续扩散，见 inference extra_seeds）。
    - normal（普通输入）：零回溯零余温，开新轮全新推演。

    Args:
        session_id: 对话会话 id。
        intent:     INTENT_NORMAL / INTENT_WHY / INTENT_MORE。
        clue:       本轮用户原文（why/more 下允许为空，normal 由调用方校验）。

    Returns:
        {"intent", "clue", "extra_seeds", "why_ref"}；why_ref 仅 why 有值
        （{turn_index, conclusion, chain, evidence}），其余为 None。

    Raises:
        DialogueError: 会话不存在。
        ValueError:    intent 非法；normal 轮 clue 为空（沿用 append_turn 校验）。
    """
    if intent not in _INTENTS:
        raise ValueError(f"非法意图: {intent!r}（合法值: {_INTENTS}）")
    sess = get_session(session_id)  # 不存在在此抛 DialogueError

    if intent == INTENT_NORMAL:
        if not (clue or "").strip():
            raise ValueError("clue 不得为空")
        return {"intent": intent, "clue": clue.strip(),
                "extra_seeds": None, "why_ref": None}

    prev = last_turn(session_id, ok_only=True)
    if prev is None:
        # 无任何成功轮可回溯：why/more 退化为普通轮语义（要求真实线索），
        # 绝不静默产出空解释/空余温——由调用方把该 ValueError 转 400。
        if not (clue or "").strip():
            raise ValueError("暂无可追问的成功推演轮次，请先输入有效线索")
        return {"intent": INTENT_NORMAL, "clue": clue.strip(),
                "extra_seeds": None, "why_ref": None}

    if intent == INTENT_WHY:
        ans = prev.get("answer") or {}
        return {"intent": intent, "clue": (clue or "").strip(),
                "extra_seeds": None,
                "why_ref": {"turn_index": prev["turn_index"],
                            "conclusion": ans.get("conclusion"),
                            "chain": ans.get("chain") or [],
                            "evidence": ans.get("assertion", {}).get("evidence")
                                       if isinstance(ans.get("assertion"), dict) else None}}

    # INTENT_MORE：余温续扩散
    return {"intent": intent, "clue": (clue or "").strip(),
            "extra_seeds": carryover_seeds(prev) or None,
            "why_ref": None}


def last_turn(session_id: str, *, ok_only: bool = False) -> Optional[dict]:
    """取最近一轮（ok_only=True 时取最近一次推演成功轮）；无则 None。

    B2 种子融合 / B3 why 回溯都以「最近 ok 轮」为基准：insufficient
    轮保留在历史中，但不阻断追问链。
    """
    with _lock:
        turns = _get_locked(session_id)["turns"]
        for turn in reversed(turns):
            if not ok_only or turn["status"] == "ok":
                return copy.deepcopy(turn)
    return None


def get_turn(session_id: str, turn_index: int) -> Optional[dict]:
    """按下标取轮次快照；越界返回 None（不抛，路由层自行 404）。"""
    with _lock:
        turns = _get_locked(session_id)["turns"]
        if 0 <= int(turn_index) < len(turns):
            return copy.deepcopy(turns[int(turn_index)])
    return None


# ───────────────────────────────────────────────
# 自检
# ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1) 创建 → 空轮次摘要；空标题在首轮自动补全
    sid = create_session("brain_ut")
    assert list_sessions("brain_ut")[0]["session_id"] == sid
    assert list_sessions("brain_ut")[0]["turn_count"] == 0

    ok_result = {
        "status": "ok", "message": "", "clue": "恶寒 发热",
        "seeds": {},
        "answer": {"conclusion": {"id": "c", "name": "表实证"}},
        "activations": [{"id": "c", "name": "表实证", "energy": 0.4, "hop": 1}],
    }
    t0 = append_turn(sid, clue="恶寒 发热", intent=INTENT_NORMAL,
                     seeds={"a": 0.5, "b": 0.5}, result=ok_result)
    assert t0["turn_index"] == 0 and t0["status"] == "ok"
    assert get_session(sid)["title"] == "恶寒 发热", "空标题应由首轮 clue 补全"
    print("建会话 + 首轮快照 + 标题补全 ✓")

    # 2) 非法输入拦截
    for bad in ({"clue": "", "intent": INTENT_NORMAL, "seeds": {}, "result": ok_result},
                {"clue": "x", "intent": "auto", "seeds": {}, "result": ok_result}):
        try:
            append_turn(sid, **bad)
            raise AssertionError(f"应拦截: {bad}")
        except ValueError as exc:
            print("非法输入拦截 ✓ ", exc)

    # 3) insufficient 轮入快照但不阻断追问链（ok_only 跳到上一 ok 轮）
    bad_result = {"status": "insufficient", "message": "尚未掌握该知识",
                  "clue": "玄学", "seeds": {}, "answer": None, "activations": []}
    t1 = append_turn(sid, clue="玄学", intent=INTENT_NORMAL,
                     seeds={}, result=bad_result)
    assert t1["turn_index"] == 1 and t1["status"] == "insufficient"
    assert last_turn(sid)["turn_index"] == 1
    assert last_turn(sid, ok_only=True)["turn_index"] == 0, "应回溯到最近 ok 轮"
    print("insufficient 入快照 + ok_only 回溯 ✓")

    # 4) 深拷贝隔离：外部改写不影响注册表内部
    snapshot = get_session(sid)
    snapshot["turns"][0]["clue"] = "篡改"
    assert get_session(sid)["turns"][0]["clue"] == "恶寒 发热"
    print("快照深拷贝隔离 ✓")

    # 5) 轮数上限拦截（-m 方式下模块以 __main__ 执行，须取当前实例而非二次导入）
    import sys
    _d = sys.modules[__name__]
    origin, _d.MAX_TURNS = _d.MAX_TURNS, 2
    try:
        append_turn(sid, clue="第三轮", intent=INTENT_NORMAL,
                    seeds={}, result=ok_result)
        raise AssertionError("超限轮应报错")
    except DialogueError as exc:
        print("轮数上限拦截 ✓ ", exc)
    finally:
        _d.MAX_TURNS = origin

    # 6) 关闭幂等 + 失效访问
    assert close_session(sid) is True
    assert close_session(sid) is False
    try:
        get_session(sid)
        raise AssertionError("关闭后访问应报错")
    except DialogueError as exc:
        print("关闭后拦截 ✓ ", exc)

    # ── 升级B2：余温种子融合 ──
    warm = {"status": "ok", "activations": [
        {"id": "c", "name": "表实证", "energy": 0.8},
        {"id": "d", "name": "咳嗽", "energy": 0.3},
    ]}
    cs = carryover_seeds(warm)
    assert cs == {"c": 0.4, "d": 0.15}, cs  # 0.8×0.5 / 0.3×0.5，Top3 全取
    assert carryover_seeds({"status": "insufficient", "activations": []}) == {}
    assert carryover_seeds(None) == {}
    assert carryover_seeds(warm, top_n=1) == {"c": 0.4}          # 截取 Top1
    assert carryover_seeds(warm, decay=0) == {}                   # 关闭余温
    dirty = carryover_seeds({"status": "ok", "activations": [
        {"id": "", "energy": 0.9},            # 空 id 丢弃
        {"id": "x", "energy": "abc"},          # 非法能量丢弃
        {"id": "y", "energy": 0.0},            # 零能量不产生余温
        {"id": "z", "energy": 7.7},            # 越界丢弃
    ]})
    assert dirty == {}, dirty
    print("余温种子融合 ✓ ", cs)

    # ── 升级B3：追问意图解析 ──
    # 无任何成功轮时：why/more 且无线索 → 报错；有线索 → 退化普通轮
    sid3 = create_session("brain_ut")
    try:
        resolve_followup(sid3, INTENT_MORE, clue="")
        raise AssertionError("无成功轮的空追问应报错")
    except ValueError as exc:
        print("无成功轮空追问拦截 ✓ ", exc)
    degraded = resolve_followup(sid3, INTENT_MORE, clue="咳嗽怎么回事")
    assert degraded["intent"] == INTENT_NORMAL and degraded["extra_seeds"] is None, degraded
    try:
        resolve_followup(sid3, INTENT_NORMAL, clue="   ")
        raise AssertionError("normal 空 clue 应报错")
    except ValueError as exc:
        print("normal 空 clue 拦截 ✓ ", exc)
    close_session(sid3)

    # 有成功轮后：why 回溯链 + more 余温种子 + normal 零回溯
    sid4 = create_session("brain_ut")
    append_turn(sid4, clue="恶寒 发热", intent=INTENT_NORMAL, seeds={"a": 1.0}, result=ok_result)
    why = resolve_followup(sid4, INTENT_WHY)
    assert why["intent"] == INTENT_WHY and why["why_ref"]["turn_index"] == 0, why
    assert why["why_ref"]["conclusion"] == {"id": "c", "name": "表实证"}, why
    more = resolve_followup(sid4, INTENT_MORE)
    assert more["intent"] == INTENT_MORE and more["extra_seeds"] == {"c": 0.2}, more
    norm = resolve_followup(sid4, INTENT_NORMAL, clue="新线索")
    assert norm["intent"] == INTENT_NORMAL and norm["extra_seeds"] is None \
        and norm["why_ref"] is None, norm
    close_session(sid4)
    print("追问意图解析 ✓  why回溯/more余温/normal新轮")

    # ── 升级B6：构造子图离线回归（多轮上下文传递 / 种子衰减 / insufficient 不污染）──
    # 纯函数核心直跑（零 Kùzu），锁定对话层与推演核心的对接语义。
    from core.inference import _reason_on_subgraph

    demo = {
        "nodes": [
            {"id": "n_a", "name": "恶寒", "category": "症状"},
            {"id": "n_b", "name": "发热", "category": "症状"},
            {"id": "n_c", "name": "太阳伤寒", "category": "证候"},
            {"id": "n_d", "name": "无汗", "category": "症状"},
        ],
        "edges": [
            {"rel_id": "e1", "src_id": "n_a", "dst_id": "n_c",
             "relation": "indicates", "weight": 1.0, "evidence": "恶寒为太阳伤寒首证"},
            {"rel_id": "e2", "src_id": "n_b", "dst_id": "n_c",
             "relation": "indicates", "weight": 1.0, "evidence": "发热恶寒并见属太阳"},
            {"rel_id": "e3", "src_id": "n_c", "dst_id": "n_d",
             "relation": "presents", "weight": 0.9, "evidence": "太阳伤寒常见无汗"},
        ],
    }

    # 轮1：首轮推演 → 双前提汇流（fanin=2）命中太阳伤寒
    r1 = _reason_on_subgraph(demo, "恶寒 发热")
    assert r1["status"] == "ok" and r1["answer"]["conclusion"]["id"] == "n_c", r1
    assert r1["answer"]["assertion"] is not None, "双前提应命中断言"

    # 轮2：多轮上下文传递——why 回溯轮1结论/证据链（零图谱，纯快照引用）
    sid5 = create_session("brain_ut")
    append_turn(sid5, clue="恶寒 发热", intent=INTENT_NORMAL,
                seeds=r1["seeds"], result=r1)
    why5 = resolve_followup(sid5, INTENT_WHY)
    assert why5["why_ref"]["turn_index"] == 0, why5
    assert why5["why_ref"]["conclusion"]["id"] == "n_c", why5
    assert why5["why_ref"]["chain"], "证据链应来自轮1落轮快照"

    # 轮3：种子衰减——余温权重 = 上轮激活峰值 × decay；结论节点可被排除
    # （B4 语义：结论多为端点无出边，从它续扩散必然空转，路由层排除后取次级）
    t0 = get_turn(sid5, 0)
    peak_by = {a["id"]: float(a.get("peak") or a.get("energy") or 0.0)
               for a in t0["activations"]}
    cs_full = carryover_seeds(t0)
    assert cs_full, "子图余温不应为空"
    assert all(abs(w - peak_by[c] * CARRYOVER_DECAY) < 1e-9
               for c, w in cs_full.items()), (cs_full, peak_by)
    cs_x = carryover_seeds(t0, exclude_ids={"n_c"})
    assert cs_x and "n_c" not in cs_x, "排除结论节点后仍应有合格余温"
    assert all(abs(w - peak_by[c] * CARRYOVER_DECAY) < 1e-9
               for c, w in cs_x.items()), (cs_x, peak_by)
    print("种子峰值衰减 ✓  full=%s excl_conclusion=%s" % (cs_full, cs_x))

    # insufficient 分支不污染会话：坏轮入历史但零余温，追问链穿透回溯最近 ok 轮
    bad5 = {"status": "insufficient", "message": "尚未掌握该知识", "clue": "玄学",
            "seeds": {}, "answer": None, "activations": []}
    tb = append_turn(sid5, clue="玄学", intent=INTENT_NORMAL, seeds={}, result=bad5)
    assert tb["status"] == "insufficient"
    assert carryover_seeds(tb) == {}, "insufficient 轮绝不产生余温"
    more5 = resolve_followup(sid5, INTENT_MORE)
    assert more5["extra_seeds"] == cs_full, \
        "追问链应穿透 insufficient 轮回溯最近 ok 轮取余温"
    close_session(sid5)
    print("多轮上下文传递 + insufficient 不污染会话 ✓")

    print("dialogue 冒烟自检全部通过 ✓")

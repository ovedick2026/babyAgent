"""
SynapseMind - 后台知识提炼引擎 (core/distill_engine.py)

Task 11.3: 把「待提炼文档」逐段喂给速读秘书并落入待收纳池的后台引擎。

核心设计（与需求逐条对应）：
- 贪多嚼不烂：core.distill.split_segments 切出完整语义段，一次只提炼一段
  （一段话意思完整，绝不半截理解），段间把进度落库 —— 前端轮询看到的
  百分比是持久化真值，刷新/重启均不回跳；
- 暂停/继续/取消：以数据库中的 status 为唯一真值。API 层直接改状态，
  引擎在每段开始前重读状态，发现不再是 RUNNING 即安静退出 ——
  暂停粒度 = 一段（当前段会正常完成，不会截断 LLM 调用）；
- 全局单任务互斥：整个系统同一时刻只跑一个文档的提炼（跨脑互斥），
  避免 LLM 并发挤兑与进度混乱；
- 启动自愈：进程曾被杀时遗留的 RUNNING 孤儿统一改 PAUSED（可「继续」）。

线程模型：
- 单条守护线程 _worker_loop 常驻，_wake 事件驱动，空闲零 CPU；
- _current 槽位即互斥锁的语义载体（持有人=正在提炼的任务），
  全部读写都在 _engine_lock 内完成。
"""

import logging
import threading
import time
from typing import Optional

from core import intake_store
from core.brain_manager import get_or_create
from core.distill import split_segments
from core.intake import commit_proposals
from core.metabolism import MetabolismError, skim_extract_proposals

logger = logging.getLogger("synapsemind.distill_engine")

# 单段 LLM 解析失败的重试次数（含首次）：瞬时网络抖动/偶发脏返回不致死
_SEG_ATTEMPTS = 2
_RETRY_BACKOFF = 1.5  # 秒


class DistillEngineError(Exception):
    """提炼引擎业务异常：message 面向用户可读（API 层转 4xx）。"""


# ───────────────────────────────────────────────
# 引擎状态（全部受 _engine_lock 保护）
# ───────────────────────────────────────────────

_engine_lock = threading.Lock()
_current: Optional[dict] = None      # {"brain_id", "doc_id"} 正在提炼的任务
_job: Optional[dict] = None          # 待处理的下一个任务槽（容量恒为 1）
_wake = threading.Event()            # 唤醒守护线程
_worker_thread: Optional[threading.Thread] = None


def _ensure_worker() -> None:
    """保证守护线程存活（boot 时启动；防御性懒启动兜底）。"""
    global _worker_thread
    with _engine_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(
                target=_worker_loop, name="distill-engine", daemon=True
            )
            _worker_thread.start()
            logger.info("提炼引擎守护线程已启动")


def _worker_loop() -> None:
    """常驻消费循环：醒来 → 取走唯一任务槽 → 执行 → 回到休眠。"""
    global _job
    while True:
        _wake.wait()
        _wake.clear()  # 先清信号再取任务：此后任何 start_doc 的 set 都不会丢
        with _engine_lock:
            job, _job = _job, None
        if job:
            _run_job(job["brain_id"], job["doc_id"])


# ───────────────────────────────────────────────
# 对外 API（供 web_routes 调用）
# ───────────────────────────────────────────────

def start_doc(brain_id: str, doc_id: int) -> dict:
    """开始/继续提炼指定文档（QUEUED/FAILED=从头，PAUSED=续跑）。

    同步把状态置为 RUNNING 后再入队，前端轮询立即可见；
    全局互斥：已有任务在跑时拒绝（提示先等待或暂停）。

    Raises:
        DistillEngineError: 文档不存在 / 状态不允许 / 引擎忙碌。
    """
    global _current, _job
    with _engine_lock:
        if _current is not None:
            raise DistillEngineError(
                f"已有提炼任务在进行中（doc={_current['doc_id']}），"
                "请等待完成或先暂停"
            )
        conn = get_or_create(brain_id).get_sqlite()
        doc = intake_store.get_doc(conn, doc_id)
        if doc is None:
            raise DistillEngineError(f"文档不存在: doc={doc_id}")
        if doc["status"] == intake_store.STATUS_RUNNING:
            raise DistillEngineError("该文档已在提炼中")
        # PAUSED 续跑保留进度；QUEUED/FAILED 重跑清零进度
        reset = doc["status"] != intake_store.STATUS_PAUSED
        intake_store.update_status(
            conn, doc_id, intake_store.STATUS_RUNNING, reset_progress=reset
        )
        _current = {"brain_id": brain_id, "doc_id": doc_id}
        _job = {"brain_id": brain_id, "doc_id": doc_id}
    _ensure_worker()
    _wake.set()
    logger.info("提炼任务入队: brain=%s doc=%s（%s）",
                brain_id, doc_id, "续跑" if not reset else "从头")
    return {"doc_id": doc_id, "status": intake_store.STATUS_RUNNING}


def pause_doc(brain_id: str, doc_id: int) -> dict:
    """暂停：仅当该文档正是当前运行任务时有效；状态即时 PAUSED，
    引擎在当前段完成后安静退出（进度保留，「继续」可复活）。"""
    with _engine_lock:
        cur = _current
    if cur is None or cur != {"brain_id": brain_id, "doc_id": doc_id}:
        raise DistillEngineError("该文档当前不在提炼中，无法暂停")
    conn = get_or_create(brain_id).get_sqlite()
    intake_store.update_status(conn, doc_id, intake_store.STATUS_PAUSED)
    logger.info("提炼暂停请求: brain=%s doc=%s（当前段完成后生效）", brain_id, doc_id)
    return {"doc_id": doc_id, "status": intake_store.STATUS_PAUSED}


def cancel_doc(brain_id: str, doc_id: int) -> dict:
    """取消：RUNNING（须为当前任务）或 PAUSED → 回 QUEUED 并清零进度；
    按钮语义「取消后变回提炼」由 QUEUED 态保证。"""
    conn = get_or_create(brain_id).get_sqlite()
    doc = intake_store.get_doc(conn, doc_id)
    if doc is None:
        raise DistillEngineError(f"文档不存在: doc={doc_id}")
    if doc["status"] == intake_store.STATUS_RUNNING:
        with _engine_lock:
            cur = _current
        if cur != {"brain_id": brain_id, "doc_id": doc_id}:
            raise DistillEngineError("运行中文档只能由引擎取消（状态不一致）")
    elif doc["status"] != intake_store.STATUS_PAUSED:
        raise DistillEngineError(f"当前状态 {doc['status']} 无需取消")
    intake_store.update_status(
        conn, doc_id, intake_store.STATUS_QUEUED, reset_progress=True
    )
    logger.info("提炼取消: brain=%s doc=%s → QUEUED", brain_id, doc_id)
    return {"doc_id": doc_id, "status": intake_store.STATUS_QUEUED}


def is_current(brain_id: str, doc_id: int) -> bool:
    """该文档是否正是正在运行的任务（API 层禁删 RUNNING 文档的判据）。"""
    with _engine_lock:
        return _current == {"brain_id": brain_id, "doc_id": doc_id}


def busy_doc() -> Optional[dict]:
    """当前正在提炼的任务（无则 None），供设置/列表页展示。"""
    with _engine_lock:
        return dict(_current) if _current else None


# ───────────────────────────────────────────────
# 任务执行体
# ───────────────────────────────────────────────

def _still_running(conn, doc_id: int) -> bool:
    """暂停/取消的唯一真值判定：数据库里的 status 是否仍是 RUNNING。"""
    doc = intake_store.get_doc(conn, doc_id)
    return doc is not None and doc["status"] == intake_store.STATUS_RUNNING


def _finish(conn, doc_id: int, status: str, *, error: Optional[str] = None) -> None:
    """终态落库：仅当仍处于 RUNNING 才写终态 —— 若用户恰在最后一段
    完成瞬间点了暂停/取消，尊重用户操作，绝不覆盖其状态。"""
    doc = intake_store.get_doc(conn, doc_id)
    if doc is None or doc["status"] != intake_store.STATUS_RUNNING:
        logger.info("doc=%s 状态已被外部改为 %s，跳过终态 %s",
                    doc_id, doc["status"] if doc else "已删除", status)
        return
    intake_store.update_status(conn, doc_id, status, error=error)


def _skim_with_retry(text: str) -> dict:
    """单段速读 + 一次重试；两次皆败抛最后一次异常（含中文语义）。"""
    last_exc: Exception = MetabolismError("未执行")
    for attempt in range(1, _SEG_ATTEMPTS + 1):
        try:
            return skim_extract_proposals(text)
        except Exception as exc:  # MetabolismError 或任何意外，都值得重试一次
            last_exc = exc
            if attempt < _SEG_ATTEMPTS:
                logger.warning("段落速读失败（第 %d 次），%.1fs 后重试: %s",
                               attempt, _RETRY_BACKOFF, exc)
                time.sleep(_RETRY_BACKOFF)
    raise last_exc


def _run_job(brain_id: str, doc_id: int) -> None:
    """执行单个文档的逐段提炼：段间检查暂停/取消，进度逐段落库。"""
    session = get_or_create(brain_id)
    conn = session.get_sqlite()
    try:
        doc = intake_store.get_doc(conn, doc_id, with_text=True)
        if doc is None or doc["status"] != intake_store.STATUS_RUNNING:
            return  # 入队后被立刻暂停/取消/删除：安静退出

        segments = split_segments(doc["raw_text"])
        total = len(segments)
        if total == 0:
            intake_store.update_progress(conn, doc_id, total=0, done=0, proposals=0)
            _finish(conn, doc_id, intake_store.STATUS_DONE)
            return

        done = int(doc["done"] or 0)
        proposals_total = int(doc["proposals"] or 0)
        if done > total:
            done = total
        if done != int(doc["total"] or 0) and doc["total"]:
            # 分段配置变化导致边界漂移：续跑会错位，安全起见从头重算
            # （此前已入池提案留待人工审阅，本层绝不自动清理图谱数据）
            logger.warning("doc=%s 分段总数变化 (%s→%s)，进度清零重跑",
                           doc_id, doc["total"], total)
            done, proposals_total = 0, 0
        intake_store.update_progress(
            conn, doc_id, total=total, done=done, proposals=proposals_total
        )

        kuzu = session.get_kuzu()
        for idx in range(done, total):
            if not _still_running(conn, doc_id):   # 暂停/取消请求 → 安静退出
                logger.info("doc=%s 在段 %d/%d 前被暂停/取消", doc_id, idx + 1, total)
                return
            seg = segments[idx]
            result = _skim_with_retry(seg)
            if not _still_running(conn, doc_id):
                # LLM 调用期间（可能长达数十秒）被暂停/取消：本段结果作废、
                # 不写进度 —— 用户操作即时生效，绝不残留半截进度或重复提案。
                logger.info("doc=%s 段 %d/%d 执行期间被暂停/取消，结果作废",
                            doc_id, idx + 1, total)
                return
            summary = commit_proposals(kuzu, result["proposals"])
            proposals_total += len(summary["committed"])
            done = idx + 1
            intake_store.update_progress(
                conn, doc_id, done=done, proposals=proposals_total
            )
            logger.info("doc=%s 段 %d/%d 提炼完成，新增提案 %d（累计 %d）",
                        doc_id, done, total, len(summary["committed"]), proposals_total)

        _finish(conn, doc_id, intake_store.STATUS_DONE)
        logger.info("doc=%s 提炼完成: %d 段，共 %d 条提案入待收纳池",
                    doc_id, total, proposals_total)
    except Exception as exc:
        logger.exception("doc=%s 提炼失败: %s", doc_id, exc)
        try:
            _finish(conn, doc_id, intake_store.STATUS_FAILED,
                    error=str(exc)[:500])
        except Exception:
            logger.critical("doc=%s 失败状态落库亦失败", doc_id)
    finally:
        with _engine_lock:
            global _current
            if _current and _current["doc_id"] == doc_id:
                _current = None


# ───────────────────────────────────────────────
# 启动自愈 (boot)
# ───────────────────────────────────────────────

def boot() -> int:
    """启动钩子：全部脑的 RUNNING 孤儿 → PAUSED，并拉起守护线程。

    Returns:
        自愈改写的孤儿文档数（供启动日志观测）。
    """
    from core.brain_manager import list_brains  # 就近引入，避免环依赖
    healed = 0
    for bid in list_brains():
        try:
            conn = get_or_create(bid).get_sqlite()
            healed += intake_store.reset_running_to_paused(conn)
        except Exception as exc:
            logger.error("brain=%s 提炼自愈失败（跳过）: %s", bid, exc)
    if healed:
        logger.warning("提炼引擎启动自愈: %d 条 RUNNING 孤儿 → PAUSED", healed)
    _ensure_worker()
    return healed


if __name__ == "__main__":
    # 冒烟自检（独立临时脑 + 桩 LLM，零网络）：
    # 全生命周期 提交→提炼→暂停→继续→完成→取消→自愈 全断言。
    import logging as _logging
    import shutil

    _logging.basicConfig(level=logging.INFO)

    BRAIN = "_distill_selftest"
    from storage import persistence as _persistence
    shutil.rmtree(_persistence.RUNTIME_DIR / BRAIN, ignore_errors=True)

    import sys as _sys
    _self = _sys.modules[__name__]

    # ── 桩 LLM：每次返回 2 条提案；可用门控模拟慢段 ──
    _gate = threading.Event()
    _gate.set()
    _calls = {"n": 0}

    def _fake_skim(text, **kw):
        _calls["n"] += 1
        _gate.wait(timeout=10)  # 测试可关门把引擎卡在段内，实现确定性暂停
        if not _gate.is_set():
            raise RuntimeError("gate closed")
        return {"proposals": [
            {"source": f"概念{_calls['n']}A", "target": f"概念{_calls['n']}B",
             "relation": "causes", "weight": 0.5, "evidence": text[:30]},
            {"source": f"概念{_calls['n']}B", "target": f"概念{_calls['n']}C",
             "relation": "suggests", "weight": 0.6, "evidence": ""},
        ], "raw": "[]", "dropped": 0}

    _self.skim_extract_proposals = _fake_skim

    from core.intake import list_pending

    session = get_or_create(BRAIN)
    sq, kz = session.get_sqlite(), session.get_kuzu()
    text3 = "甲段知识。乙段知识。\n\n丙段知识。丁段知识。\n\n戊段知识。己段知识。"
    d1 = intake_store.create_doc(sq, "三段文档", text3)

    def wait_status(doc_id, want, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            cur = intake_store.get_doc(sq, doc_id)
            if cur["status"] == want:
                return cur
            time.sleep(0.05)
        raise AssertionError(f"等待 doc={doc_id} → {want} 超时，实际 "
                             f"{intake_store.get_doc(sq, doc_id)['status']}")

    def wait_idle(timeout=10.0):
        """终态落库后 worker 还需释放 _current 槽位；等它空闲消除收尾竞态。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if busy_doc() is None:
                return
            time.sleep(0.05)
        raise AssertionError("等待引擎空闲超时")

    # ① 全流程完成：3 段 × 2 提案
    start_doc(BRAIN, d1)
    d = wait_status(d1, intake_store.STATUS_DONE)
    assert d["total"] == 3 and d["done"] == 3 and d["proposals"] == 6, d
    assert len(list_pending(kz)) == 6
    wait_idle()
    assert busy_doc() is None, "完成后引擎应空转"

    # ② 段内确定性暂停 → 继续 → 完成
    # 门控桩：在第 _arm['n'] 次 skim 调用入口先落门再放行 —— 引擎必然
    # 卡在该段的 LLM 调用内部；主线程暂停/取消后放行，引擎的段后复查
    # 会把该段结果作废（进度零残留，用户操作即时生效）。
    _seg_started = threading.Event()
    _arm = {"n": 0}

    def _fake_skim_gated(text, **kw):
        if _arm["n"] and _calls["n"] + 1 == _arm["n"]:
            _seg_started.set()
            _gate.clear()
        return _fake_skim(text, **kw)

    d2 = intake_store.create_doc(sq, "暂停续跑", text3)
    _self.skim_extract_proposals = _fake_skim_gated
    _arm["n"] = 4  # doc=2 的段 1（全局第 4 次调用）
    start_doc(BRAIN, d2)
    assert _seg_started.wait(timeout=10), "未进入段 1"
    pause_doc(BRAIN, d2)   # 引擎此刻卡在段 1 的 LLM 调用内
    _gate.set()            # 放行；引擎段后复查见 PAUSED → 结果作废
    wait_status(d2, intake_store.STATUS_PAUSED)
    d = intake_store.get_doc(sq, d2)
    assert d["done"] == 0 and d["proposals"] == 0, d
    _self.skim_extract_proposals = _fake_skim
    wait_idle()
    start_doc(BRAIN, d2)  # 「继续」
    d = wait_status(d2, intake_store.STATUS_DONE)
    assert d["done"] == 3 and d["proposals"] == 6, d

    # ③ 运行中取消 → QUEUED 清零；段内结果作废
    d3 = intake_store.create_doc(sq, "取消测试", text3)
    _self.skim_extract_proposals = _fake_skim_gated
    _seg_started.clear()
    _arm["n"] = _calls["n"] + 1  # 下一次 skim 调用即门控目标
    start_doc(BRAIN, d3)
    assert _seg_started.wait(timeout=10), "未进入门控段"
    cancel_doc(BRAIN, d3)
    _gate.set()
    d = wait_status(d3, intake_store.STATUS_QUEUED)
    assert d["done"] == 0 and d["proposals"] == 0, d
    wait_idle()

    # ④ 忙碌互斥：卡住引擎后 start 第二条应被拒
    _gate.clear()
    _arm["n"] = _calls["n"] + 1
    start_doc(BRAIN, d3)
    deadline = time.time() + 5
    while time.time() < deadline and busy_doc() is None:
        time.sleep(0.05)
    try:
        start_doc(BRAIN, d1)
        raise AssertionError("忙碌时应拒绝第二个任务")
    except DistillEngineError:
        pass
    cancel_doc(BRAIN, d3)
    _gate.set()
    wait_status(d3, intake_store.STATUS_QUEUED)
    wait_idle()
    _self.skim_extract_proposals = _fake_skim

    # ⑤ 启动自愈：伪造 RUNNING 孤儿 → boot 后变 PAUSED
    d4 = intake_store.create_doc(sq, "孤儿", text3)
    intake_store.update_status(sq, d4, intake_store.STATUS_RUNNING,
                               reset_progress=True)
    healed = boot()
    assert healed >= 1
    assert intake_store.get_doc(sq, d4)["status"] == intake_store.STATUS_PAUSED

    # ⑥ 删除保护判据：运行中 is_current=True
    _gate.clear()
    start_doc(BRAIN, d4)
    deadline = time.time() + 5
    while time.time() < deadline and not is_current(BRAIN, d4):
        time.sleep(0.05)
    assert is_current(BRAIN, d4) and not is_current(BRAIN, d1)
    cancel_doc(BRAIN, d4)
    _gate.set()

    print("distill_engine 自检通过 ✓ 完成/暂停续跑/取消/互斥/自愈/运行判据")

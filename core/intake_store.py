"""
SynapseMind - 知识投喂文档存取 (core/intake_store.py)

Task 11.2: 「待提炼列表」的持久层 —— 每一份用户提交的知识（粘贴文本
或 txt 全文）在库中是一行 intake_docs 记录，随 brain.db 走多脑隔离与
/data 镜像刷盘，容器重启不丢。

状态机（CHECK 约束强校验）：
    QUEUED   待提炼（列表初始态，「提炼」按钮可见）
    RUNNING  提炼中（进度百分比 + 暂停/取消可见）
    PAUSED   已暂停（「继续」可见；重启自愈 RUNNING→PAUSED 的落点）
    DONE     已完成（全部段落提炼完毕，提案已入待收纳池）
    FAILED   失败（error 字段记录原因，可重新「提炼」）

合法迁移由上层引擎/API 把关，本层只做状态值合法性（成员资格）校验。

设计取舍：
- 每个公开函数入口都执行 ensure_intake_table()：CREATE TABLE IF NOT
  EXISTS 在 SQLite 上是纳秒级 DDL，换来「调用方零迁移心智」，老库
  / 新库 / 内存库通吃；
- 全部查询参数化，绝不拼字符串；
- raw_text 原文整行保存 —— 「删除此条知识」的语义即删除整行。
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("synapsemind.intake_store")

# ───────────────────────────────────────────────
# 常量（供引擎 / API / 前端共享，避免魔法字符串）
# ───────────────────────────────────────────────

STATUS_QUEUED = "QUEUED"
STATUS_RUNNING = "RUNNING"
STATUS_PAUSED = "PAUSED"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"

ALL_STATUSES = (STATUS_QUEUED, STATUS_RUNNING, STATUS_PAUSED, STATUS_DONE, STATUS_FAILED)


def _now_iso() -> str:
    """UTC ISO 时间戳（字符串排序即时间排序，便于列表按最新在前）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ───────────────────────────────────────────────
# 建表（幂等）
# ───────────────────────────────────────────────

def ensure_intake_table(conn: sqlite3.Connection) -> None:
    """幂等创建 intake_docs 表：任意库（老 brain.db / 全新库 / 内存库）零副作用。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS intake_docs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT    NOT NULL,
            raw_text   TEXT    NOT NULL,
            status     TEXT    NOT NULL DEFAULT 'QUEUED'
                       CHECK (status IN ('QUEUED','RUNNING','PAUSED','DONE','FAILED')),
            total      INTEGER NOT NULL DEFAULT 0,   -- 语义段总数（分段后回填）
            done       INTEGER NOT NULL DEFAULT 0,   -- 已提炼完成段数（进度百分比分子）
            proposals  INTEGER NOT NULL DEFAULT 0,   -- 已产出突触提案数
            error      TEXT,                          -- FAILED 原因
            created_at TEXT    NOT NULL,
            updated_at TEXT    NOT NULL
        )
        """
    )
    conn.commit()


# ───────────────────────────────────────────────
# CRUD
# ───────────────────────────────────────────────

def create_doc(conn: sqlite3.Connection, title: str, raw_text: str) -> int:
    """投喂一条知识：初始状态 QUEUED，返回自增 id。

    Raises:
        ValueError: 标题或正文为空白（空提交无意义，直接拒绝）。
    """
    if not (title or "").strip():
        raise ValueError("标题不能为空")
    if not (raw_text or "").strip():
        raise ValueError("知识正文不能为空")

    ensure_intake_table(conn)
    now = _now_iso()
    cur = conn.execute(
        "INSERT INTO intake_docs (title, raw_text, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (title.strip(), raw_text, STATUS_QUEUED, now, now),
    )
    conn.commit()
    doc_id = int(cur.lastrowid)
    logger.info("intake_docs: 新增 doc=%s「%s」（%d 字，QUEUED）", doc_id, title.strip(), len(raw_text))
    return doc_id


def list_docs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """待提炼列表（新→旧）；不含 raw_text 大字段，列表接口轻量。

    附带 progress 百分比（total=0 时按 0% 处理，避免除零）。
    """
    ensure_intake_table(conn)
    rows = conn.execute(
        "SELECT id, title, status, total, done, proposals, error, created_at, updated_at "
        "FROM intake_docs ORDER BY id DESC"
    ).fetchall()
    docs = []
    for r in rows:
        d = dict(r)
        d["progress"] = round(100.0 * d["done"] / d["total"], 1) if d["total"] else 0.0
        docs.append(d)
    return docs


def get_doc(conn: sqlite3.Connection, doc_id: int, with_text: bool = False) -> Optional[dict[str, Any]]:
    """取单条文档；with_text=True 时附带 raw_text（引擎分段时需要）。不存在返回 None。"""
    ensure_intake_table(conn)
    cols = "id, title, raw_text, status, total, done, proposals, error, created_at, updated_at" \
        if with_text else "id, title, status, total, done, proposals, error, created_at, updated_at"
    row = conn.execute(f"SELECT {cols} FROM intake_docs WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["progress"] = round(100.0 * d["done"] / d["total"], 1) if d["total"] else 0.0
    return d


def update_status(
    conn: sqlite3.Connection,
    doc_id: int,
    status: str,
    *,
    error: Optional[str] = None,
    reset_progress: bool = False,
) -> bool:
    """状态流转（取消=回 QUEUED 且清零进度；重新提炼 FAILED 亦同）。

    Args:
        reset_progress: True 时清零 done/proposals/error（取消、重跑场景）。
    Returns:
        True=成功；False=文档不存在。
    Raises:
        ValueError: status 非法成员。
    """
    if status not in ALL_STATUSES:
        raise ValueError(f"非法状态: {status!r}（合法值: {ALL_STATUSES}）")
    ensure_intake_table(conn)
    if reset_progress:
        cur = conn.execute(
            "UPDATE intake_docs SET status=?, done=0, proposals=0, error=NULL, updated_at=? WHERE id=?",
            (status, _now_iso(), doc_id),
        )
    else:
        cur = conn.execute(
            "UPDATE intake_docs SET status=?, error=?, updated_at=? WHERE id=?",
            (status, error, _now_iso(), doc_id),
        )
    conn.commit()
    return cur.rowcount > 0


def update_progress(conn: sqlite3.Connection, doc_id: int, *, total: int = None,
                    done: int = None, proposals: int = None) -> bool:
    """引擎运行期进度落库（只动给出的字段）；RUNNING 文档每次段完成即调用，
    保证前端轮询看到的百分比是持久化真值（刷新/重启均不回跳）。"""
    ensure_intake_table(conn)
    sets, vals = ["updated_at=?"], [_now_iso()]
    if total is not None:
        if total < 0:
            raise ValueError("total 不能为负")
        sets.append("total=?"); vals.append(total)
    if done is not None:
        if done < 0:
            raise ValueError("done 不能为负")
        sets.append("done=?"); vals.append(done)
    if proposals is not None:
        if proposals < 0:
            raise ValueError("proposals 不能为负")
        sets.append("proposals=?"); vals.append(proposals)
    vals.append(doc_id)
    cur = conn.execute(f"UPDATE intake_docs SET {', '.join(sets)} WHERE id=?", vals)
    conn.commit()
    return cur.rowcount > 0


def delete_doc(conn: sqlite3.Connection, doc_id: int) -> bool:
    """删除此条知识（整行含原文）。RUNNING 禁删由 API 层把关（本层提供纯删除能力）。"""
    ensure_intake_table(conn)
    cur = conn.execute("DELETE FROM intake_docs WHERE id=?", (doc_id,))
    conn.commit()
    if cur.rowcount:
        logger.info("intake_docs: 已删除 doc=%s", doc_id)
    return cur.rowcount > 0


def reset_running_to_paused(conn: sqlite3.Connection) -> int:
    """启动自愈：进程曾被杀时遗留的 RUNNING 孤儿统一改 PAUSED（「继续」可复活）。

    Returns:
        改写的行数（供启动日志观测）。
    """
    ensure_intake_table(conn)
    cur = conn.execute(
        "UPDATE intake_docs SET status=?, updated_at=? WHERE status=?",
        (STATUS_PAUSED, _now_iso(), STATUS_RUNNING),
    )
    conn.commit()
    if cur.rowcount:
        logger.warning("intake_docs: 启动自愈，%d 条 RUNNING 孤儿 → PAUSED", cur.rowcount)
    return cur.rowcount


# ───────────────────────────────────────────────
# 自检
# ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    doc = create_doc(conn, "神经网络基础", "反向传播是一种梯度下降算法。\
\
卷积核负责局部特征提取。")
    assert doc == 1
    d = get_doc(conn, 1)
    assert d["status"] == STATUS_QUEUED and d["progress"] == 0.0

    # 引擎生命周期：分段回填 total → RUNNING → 段完成进度 → 完成
    update_progress(conn, 1, total=2)
    update_status(conn, 1, STATUS_RUNNING)
    update_progress(conn, 1, done=1, proposals=5)
    d = get_doc(conn, 1)
    assert d["progress"] == 50.0 and d["proposals"] == 5
    update_status(conn, 1, STATUS_DONE)
    assert get_doc(conn, 1)["status"] == STATUS_DONE

    # 失败 → 记录原因 → 取消重跑清零
    update_status(conn, 1, STATUS_FAILED, error="LLM 超时")
    assert get_doc(conn, 1)["error"] == "LLM 超时"
    update_status(conn, 1, STATUS_QUEUED, reset_progress=True)
    d = get_doc(conn, 1)
    assert d == {**d, "done": 0, "proposals": 0, "error": None} or (d["done"] == 0 and d["error"] is None)

    # 启动自愈
    update_status(conn, 1, STATUS_RUNNING)
    assert reset_running_to_paused(conn) == 1
    assert get_doc(conn, 1)["status"] == STATUS_PAUSED

    # 非法状态 / 空白提交被拒
    for bad in ("STOPPED", ""):  # 成员外状态 / 空串
        try:
            update_status(conn, 1, bad) if bad else create_doc(conn, " ", "x")
            raise AssertionError("应拒绝非法输入")
        except ValueError:
            pass
    assert get_doc(conn, 999) is None and not delete_doc(conn, 999)
    assert not update_status(conn, 999, STATUS_DONE)

    # 列表轻量（无 raw_text 键）+ 删除
    docs = list_docs(conn)
    assert len(docs) == 1 and "raw_text" not in docs[0] and docs[0]["id"] == 1
    assert delete_doc(conn, 1) and list_docs(conn) == []

    print("intake_store 自检通过 ✓ 状态机/进度/自愈/删除全部符合预期")
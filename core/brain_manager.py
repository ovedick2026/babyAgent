"""
SynapseMind - 多脑隔离与生命周期管理 (core/brain_manager.py)

核心铁律：
- 脑与脑之间严格隔离：每脑独立 SQLite 库文件 + 独立 Kùzu 图目录，
  全部落在本地高速工作盘 /tmp/synapse_runtime/{brain_id}/（坚决避开 NFS 锁）。
- 刷盘回写 /data 挂载盘由 storage.persistence（sync_to_storage）统一负责，
  S3 灾备由 storage.s3_backup 负责；本模块只管运行期连接的创建与持有。

本文件当前实现：
- Task 2.2.1  BrainSession：单脑会话的惰性连接创建（get_sqlite / get_kuzu）

后续任务将在此文件追加：
- Task 2.2.2  进程内注册表 brain_registry：list_brains / get_or_create
- Task 2.2.3  close_brain：切换大脑时的安全释放（关闭前强制 sync_to_storage）
"""

import logging
import re
import sqlite3
import threading
from typing import Optional

from storage import persistence as _persistence

logger = logging.getLogger("synapsemind.brain_manager")

# brain_id 合法性白名单：字母/数字/下划线/连字符，防路径穿越与非法文件名
BRAIN_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def validate_brain_id(brain_id: str) -> str:
    """校验并原样返回合法 brain_id；非法输入直接抛 ValueError（调用方转 400）。"""
    if not isinstance(brain_id, str) or not BRAIN_ID_RE.match(brain_id):
        raise ValueError(f"非法 brain_id: {brain_id!r}（仅允许字母/数字/下划线/连字符，≤64 位）")
    return brain_id


class BrainSession:
    """
    单脑会话：封装 brain_id 对应的 SQLite 连接与 Kùzu 实例。

    惰性初始化原则（Lazy Init）：
      - 实例化本身零 IO，只有第一次 get_sqlite() / get_kuzu() 才真正建目录、
        建库、建连接；
      - 这样扫描、列表等只读场景不会为从未写入过的脑产生垃圾库文件。

    目录布局（与 storage.s3_backup._locate_brain_sources 的候选顺序一致）：
      /tmp/synapse_runtime/{brain_id}/
      ├── brain.db     ← SQLite 业务状态库（单文件，便于快照）
      └── kuzu/        ← Kùzu 图数据库目录（概念与突触拓扑）
    """

    def __init__(self, brain_id: str):
        self.brain_id = validate_brain_id(brain_id)
        self._dir = _persistence.RUNTIME_DIR / self.brain_id
        self._sqlite_path = self._dir / "brain.db"
        self._kuzu_dir = self._dir / "kuzu"

        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._kuzu_db = None
        self._kuzu_conn = None

        # 初始化互斥锁：并发首访时保证同脑连接单例，杜绝重复建连
        self._init_lock = threading.Lock()

    # ───────────────────────────────────────────────
    # SQLite 业务状态库（惰性初始化）
    # ───────────────────────────────────────────────

    def get_sqlite(self) -> sqlite3.Connection:
        """
        获取本脑的 SQLite 连接（首次调用时建目录并建库）。

        刻意保持默认回滚日志（journal_mode=DELETE）而非 WAL：
        WAL 会把最新写入滞留在 -wal 旁路文件中，而 Task 1.3.1 的快照
        只打包主库单文件，WAL 模式下可能拍到缺尾快照。默认模式下每次
        事务提交即完整落回 brain.db，快照永远一致。
        """
        if self._sqlite_conn is None:
            with self._init_lock:
                if self._sqlite_conn is None:  # 双重检查：锁内二次确认
                    self._dir.mkdir(parents=True, exist_ok=True)
                    conn = sqlite3.connect(str(self._sqlite_path))
                    conn.row_factory = sqlite3.Row          # 查询结果可按列名取值
                    conn.execute("PRAGMA foreign_keys=ON;")  # 强制引用完整性
                    self._sqlite_conn = conn
                    logger.info("SQLite 连接就绪: brain=%s → %s", self.brain_id, self._sqlite_path)
        return self._sqlite_conn

    @property
    def sqlite_path(self) -> Optional[str]:
        """本脑 SQLite 库文件路径（供快照打包 / 刷盘诊断使用）。"""
        return str(self._sqlite_path)

    # ───────────────────────────────────────────────
    # Kùzu 图数据库（惰性初始化）
    # ───────────────────────────────────────────────

    def get_kuzu(self):
        """
        获取本脑的 Kùzu 连接（首次调用时建目录并实例化图引擎）。

        kuzu.Database 指向不存在目录时会自动初始化空图；
        懒加载 import kuzu 使纯手工模式的冷启动不被 C++ 扩展拖慢。
        """
        if self._kuzu_conn is None:
            with self._init_lock:
                if self._kuzu_conn is None:  # 双重检查
                    import kuzu  # 延迟导入：仅真正用到图谱时加载
                    self._dir.mkdir(parents=True, exist_ok=True)
                    self._kuzu_db = kuzu.Database(str(self._kuzu_dir))
                    self._kuzu_conn = kuzu.Connection(self._kuzu_db)
                    logger.info("Kùzu 实例就绪: brain=%s → %s", self.brain_id, self._kuzu_dir)
        return self._kuzu_conn

    @property
    def kuzu_dir(self) -> str:
        """本脑 Kùzu 图目录路径（供快照打包 / 目录级同步使用）。"""
        return str(self._kuzu_dir)

    # ───────────────────────────────────────────────
    # 句柄释放（低层；注册表级安全释放见 Task 2.2.3 close_brain）
    # ───────────────────────────────────────────────

    def close(self) -> None:
        """释放本脑全部句柄：Kùzu 实例显式 close（防文件句柄占用导致刷盘失败）+ SQLite 断连。"""
        with self._init_lock:
            if self._kuzu_conn is not None:
                try:
                    self._kuzu_conn.close()
                    if self._kuzu_db is not None:
                        self._kuzu_db.close()
                except Exception as exc:  # kuzu 异常类型不稳，兜底捕获
                    logger.warning("brain=%s Kùzu 关闭异常（忽略）: %s", self.brain_id, exc)
                self._kuzu_conn = None
                self._kuzu_db = None

            if self._sqlite_conn is not None:
                try:
                    self._sqlite_conn.close()
                except sqlite3.Error as exc:
                    logger.warning("brain=%s SQLite 关闭异常（忽略）: %s", self.brain_id, exc)
                self._sqlite_conn = None

            logger.info("brain=%s 会话句柄已全部释放", self.brain_id)


# ───────────────────────────────────────────────
# 多脑注册表与 get_or_create (Task 2.2.2)
# ───────────────────────────────────────────────

# 进程内注册表：brain_id → BrainSession（保证同脑全进程连接单例）
_brain_registry: dict[str, "BrainSession"] = {}

# 注册表互斥锁：并发首访/创建不同脑时保护字典写入
_registry_lock = threading.Lock()


def list_brains() -> list[str]:
    """
    扫描运行时目录，列出所有已物化的脑（目录布局为准）。

    判定标准：RUNTIME_DIR 下存在合法命名的子目录且其中含 brain.db。
    纯注册表记录（无落盘文件）不算已物化，不进入列表 ——
    避免容器重启后幽灵脑复活。

    Returns:
        排序后的 brain_id 列表（保证 UI 列表展示顺序稳定）。
    """
    runtime = _persistence.RUNTIME_DIR
    if not runtime.exists():
        return []
    brains = [
        d.name
        for d in runtime.iterdir()
        if d.is_dir() and BRAIN_ID_RE.match(d.name) and (d / "brain.db").is_file()
    ]
    return sorted(brains)


def get_or_create(brain_id: str) -> BrainSession:
    """
    取得 brain_id 对应的 BrainSession 单例；不存在则建目录并初始化空库。

    行为约定：
      - 命中注册表 → 直接返回（零额外 IO，热路径毫秒级）；
      - 未命中 → 校验 brain_id → 新建 BrainSession 入表 → 幂等地
        物化目录与空 SQLite 库（保证新脑立刻能被 list_brains() 扫描到）；
      - Kùzu 引擎遵循惰性原则，延迟到首次图谱操作才加载（纯手工模式
        建点建边场景不付出 C++ 冷启动代价）。

    锁策略：注册表字典变更持短锁；目录创建与建连等慢 IO 在锁外执行，
    不阻塞其他脑的并发注册。

    Raises:
        ValueError: brain_id 非法（调用方应转为 400 响应）。
    """
    validate_brain_id(brain_id)

    with _registry_lock:
        session = _brain_registry.get(brain_id)
        if session is None:
            session = BrainSession(brain_id)
            _brain_registry[brain_id] = session
            logger.info("注册表: 新登记 brain=%s（当前共 %d 脑）", brain_id, len(_brain_registry))

    # 幂等物化：已存在的脑无副作用，新脑完成目录+空库初始化
    session.get_sqlite()
    return session


# ───────────────────────────────────────────────
# 切换大脑时的安全释放 (Task 2.2.3)
# ───────────────────────────────────────────────

def close_brain(brain_id: str) -> None:
    """
    安全释放指定脑：先强制刷盘回 /data，再显式关闭全部句柄并移出注册表。

    调用时机：切换大脑 / 卸载不活跃脑 / 服务优雅停机。

    固定顺序（顺序即正确性，不可调换）：
      1. 先从注册表摘除（pop）：后续请求拿不到半关闭会话，
         下一次 get_or_create 会重建全新会话，天然幂等；
      2. 强制执行一次 sync_to_storage()：把本脑未落盘的最新状态
         原子写回 /data 镜像 —— 顺序必须先于 close，因为 Kùzu/SQLite
         句柄关闭后缓冲数据无法再刷出；
      3. 最后 close() 释放 Kùzu 实例与 SQLite 连接：防止文件句柄
         长期占用导致 /data（NFS）刷盘失败或镜像撕裂。

    刷盘失败不阻断关闭：/data 旧镜像依然完整可用（原子替换保证），
    仅记录错误日志供诊断，句柄照常释放（进程内存安全优先）。

    Raises:
        ValueError: brain_id 非法。
    """
    validate_brain_id(brain_id)

    # 1) 先摘除注册：保证并发视角下「要么完整可用、要么彻底不存在」
    with _registry_lock:
        session = _brain_registry.pop(brain_id, None)

    if session is None:
        logger.info("close_brain: brain=%s 未在注册表中（本进程未激活），跳过", brain_id)
        return

    # 2) 关闭前强制刷盘：句柄活着的时候才能把缓冲数据写出去
    try:
        result = _persistence.sync_to_storage()
        logger.info("close_brain: brain=%s 关闭前刷盘完成 %s", brain_id, result)
    except Exception as exc:  # 刷盘异常绝不阻断句柄释放
        logger.error("close_brain: brain=%s 关闭前刷盘失败（/data 保留旧镜像）: %s", brain_id, exc)

    # 3) 显式释放 Kùzu 与 SQLite 句柄，杜绝 NFS 文件锁悬挂
    session.close()

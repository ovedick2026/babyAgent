"""
SynapseMind - 防锁死持久化架构 (storage/persistence.py)

核心铁律：SQLite 与 Kùzu 必须运行在本地高速目录（避开 NFS 锁死），
再通过原子同步写回 /data 挂载盘。

已实现：
- Task 1.2.1  路径常量 + 工作副本目录初始化 + 挂载盘可写性探测
- Task 1.2.2  sync_sqlite(): SQLite 原生 backup API 原子刷盘

后续任务将在此文件追加：
- Task 1.2.3  sync_kuzu():     Kùzu 目录原子同步
- Task 1.2.4  restore_on_boot() / sync_to_storage(): 统一入口
"""

import logging
import os
import sqlite3
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger("synapsemind.persistence")

# ───────────────────────────────────────────────
# 路径常量 (Path Constants)
# ───────────────────────────────────────────────

# HF 挂载盘（NFS）：状态镜像持久化目录（支持环境变量覆盖，便于本地开发测试）
DATA_DIR: Path = Path(os.environ.get("SYNAPSE_DATA_DIR", "/data"))

# 容器本地高速暂存区：毫秒级事务读写，无 NFS 锁风险
RUNTIME_DIR: Path = Path(os.environ.get("SYNAPSE_RUNTIME_DIR", "/tmp/synapse_runtime"))

# SQLite 与 Kùzu 各自独立的运行时子目录
SQLITE_RUNTIME_DIR: Path = RUNTIME_DIR / "sqlite"
KUZU_RUNTIME_DIR: Path = RUNTIME_DIR / "kuzu"

# /data 挂载盘上的镜像目录（与运行时结构一一对应）
SQLITE_DATA_DIR: Path = DATA_DIR / "sqlite"
KUZU_DATA_DIR: Path = DATA_DIR / "kuzu"

# ───────────────────────────────────────────────
# 运行状态
# ───────────────────────────────────────────────

# True = 挂载盘可用（正常持久化）；False = 已降级为仅本地模式
data_dir_writable: bool = False

# 刷盘互斥锁：防止手动触发与定时任务并发刷盘造成镜像撕裂
_sync_lock = threading.Lock()


def _probe_writability(target: Path) -> bool:
    """
    探测目标目录（或其父目录）是否真实可写。

    采用「写入并删除临时文件」的真实 IO 探测，
    比 os.access() 更贴近实际刷盘行为（可捕获只读挂载、权限不足等）。"""
    probe_dir = target if target.exists() else target.parent
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".synapse_probe_", dir=str(probe_dir), delete=False
        ) as fp:
            fp.write(b"probe")
            probe_path = Path(fp.name)
        probe_path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.warning("可写性探测失败 (%s): %s", probe_dir, exc)
        return False


def init_persistence() -> bool:
    """
    启动初始化入口（供 app.py 启动钩子调用）。

    流程：
    1. 建立 /tmp 高速工作副本目录结构（sqlite / kuzu 子目录）。
    2. 探测 /data 挂载盘可写性，可写则建立镜像目录。
    3. 不可写时降级为「仅本地模式」：仅打日志、不抛异常，保证服务始终可启动。

    Returns:
        bool: 挂载盘是否可用（True=正常持久化，False=仅本地模式）
    """
    global data_dir_writable

    # 1) 本地高速工作盘（必建，系统运行的最低要求）
    for d in (SQLITE_RUNTIME_DIR, KUZU_RUNTIME_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger.info("本地高速工作盘就绪: %s", RUNTIME_DIR)

    # 2) 挂载盘探测与镜像目录建立
    data_dir_writable = _probe_writability(DATA_DIR)
    if data_dir_writable:
        for d in (SQLITE_DATA_DIR, KUZU_DATA_DIR):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("持久挂载盘就绪，镜像目录已建立: %s", DATA_DIR)
    else:
        logger.warning(
            "⚠️ /data 挂载盘不可写，已降级为【仅本地模式】："
            "数据仅存在于 %s，容器重启后可能丢失！",
            RUNTIME_DIR,
        )

    return data_dir_writable


def sync_sqlite(db_filename: str = "brain_state.db") -> bool:
    """
    Task 1.2.2: SQLite 原子刷盘。

    使用 SQLite 原生 backup API 将 /tmp 工作库的一致性快照复制到
    /data 同分区的临时文件，再以 os.replace 原子替换目标库 ——
    任何时刻 /data 上要么是完整旧镜像、要么是完整新镜像，绝无半写状态。

    为什么用 backup API 而非直接文件拷贝：
    逐页复制一致性快照，可规避拷贝期间工作库被继续写入导致的数据撕裂。

    Args:
        db_filename: 工作库文件名（多脑场景下每脑一库）。

    Returns:
        True=刷盘成功；False=跳过或失败（失败时 /data 旧镜像保持不动）。
    """
    if not data_dir_writable:
        logger.warning("挂载盘不可用（仅本地模式），跳过 SQLite 刷盘")
        return False

    src_path = SQLITE_RUNTIME_DIR / db_filename
    if not src_path.exists():
        logger.info("工作库尚未创建，跳过刷盘: %s", src_path)
        return False

    final_path = SQLITE_DATA_DIR / db_filename
    # 临时文件与目标同目录（同分区），保证 os.replace 的原子性
    tmp_path = SQLITE_DATA_DIR / f".{db_filename}.tmp"

    with _sync_lock:
        try:
            src_conn = sqlite3.connect(str(src_path))
            try:
                dst_conn = sqlite3.connect(str(tmp_path))
                try:
                    src_conn.backup(dst_conn)
                finally:
                    dst_conn.close()
            finally:
                src_conn.close()

            # 同分区原子替换：瞬时生效，杜绝半写
            os.replace(tmp_path, final_path)
            logger.info("SQLite 原子刷盘完成: %s → %s", src_path, final_path)
            return True
        except (sqlite3.Error, OSError) as exc:
            logger.error("SQLite 刷盘失败，/data 旧镜像保持不动: %s", exc)
            # 清理残留临时文件，避免下次误判
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False


def sync_kuzu() -> bool:
    """
    Task 1.2.3: Kùzu 目录原子同步。

    Kùzu 是目录型数据库（含 .kz 元数据与数据文件），无法像 SQLite 那样
    用 backup API 逐页复制，因此采用目录级三步换入策略：
      1. 将运行时 Kùzu 目录完整拷贝至 /data 同分区的临时目录；
      2. 旧镜像改名暂存（rollback 用）；
      3. 临时目录 rename 换入正式位置，成功后删除暂存旧镜像。

    任何一步失败都回滚恢复旧镜像，绝不留下半拷贝的损坏状态。
    rename 在同一文件系统内是原子操作，这是原子性的根基 ——
    因此临时目录必须建在 /data（KUZU_DATA_DIR 的同分区）下。

    Returns:
        True=同步成功；False=跳过或失败（失败时 /data 旧镜像保持可用）。
    """
    import shutil  # 延迟导入：仅本函数使用

    if not data_dir_writable:
        logger.warning("挂载盘不可用（仅本地模式），跳过 Kùzu 刷盘")
        return False

    if not KUZU_RUNTIME_DIR.exists() or not any(KUZU_RUNTIME_DIR.iterdir()):
        logger.info("Kùzu 运行时目录为空，跳过刷盘")
        return False

    final_path = KUZU_DATA_DIR
    staging_path = DATA_DIR / ".kuzu_staging_tmp"   # 同分区临时目录
    rollback_path = DATA_DIR / ".kuzu_rollback_old" # 旧镜像暂存位

    with _sync_lock:
        try:
            # 1) 完整拷贝到同分区临时目录
            shutil.rmtree(staging_path, ignore_errors=True)
            shutil.copytree(KUZU_RUNTIME_DIR, staging_path)

            # 2) 旧镜像改名暂存（可能不存在，忽略）
            shutil.rmtree(rollback_path, ignore_errors=True)
            if final_path.exists():
                os.rename(final_path, rollback_path)

            # 3) 新镜像原子换入（同分区 rename）
            os.rename(staging_path, final_path)

            # 4) 清理暂存旧镜像
            shutil.rmtree(rollback_path, ignore_errors=True)
            logger.info("Kùzu 目录原子同步完成: %s → %s", KUZU_RUNTIME_DIR, final_path)
            return True
        except (OSError, shutil.Error) as exc:
            logger.error("Kùzu 同步失败，尝试回滚旧镜像: %s", exc)
            # 回滚：若正式位缺失且暂存旧镜像存在，则恢复它
            try:
                if not final_path.exists() and rollback_path.exists():
                    os.rename(rollback_path, final_path)
                    logger.warning("已回滚恢复 /data 上的旧 Kùzu 镜像")
                # 清理残留的半拷贝 staging
                shutil.rmtree(staging_path, ignore_errors=True)
            except OSError as rollback_exc:
                logger.critical("回滚失败，请人工检查 %s: %s", DATA_DIR, rollback_exc)
            return False


def restore_on_boot() -> None:
    """
    Task 1.2.4a: 容器重启自愈拉取。

    HF Space 容器重启后 /tmp 会被清空，本地工作副本丢失。
    本函数在启动时从 /data 镜像拉回上次的持久化状态：
      - SQLite 镜像 → 拷回 /tmp 工作位；
      - Kùzu 镜像目录 → 整目录拷回 /tmp 工作位。
    镜像不存在（全新部署）时静默跳过，由各引擎自行空库初始化。
    """
    import shutil

    # SQLite：整文件拉回
    for db_file in SQLITE_DATA_DIR.glob("*.db") if SQLITE_DATA_DIR.exists() else []:
        dst = SQLITE_RUNTIME_DIR / db_file.name
        try:
            shutil.copy2(db_file, dst)
            logger.info("SQLite 镜像自愈拉取: %s → %s", db_file, dst)
        except OSError as exc:
            logger.error("SQLite 自愈拉取失败 (%s)，将以空库启动: %s", db_file, exc)

    # Kùzu：整目录拉回
    if KUZU_DATA_DIR.exists() and any(KUZU_DATA_DIR.iterdir()):
        try:
            shutil.rmtree(KUZU_RUNTIME_DIR, ignore_errors=True)
            shutil.copytree(KUZU_DATA_DIR, KUZU_RUNTIME_DIR)
            logger.info("Kùzu 镜像自愈拉取: %s → %s", KUZU_DATA_DIR, KUZU_RUNTIME_DIR)
        except (OSError, shutil.Error) as exc:
            logger.error("Kùzu 自愈拉取失败，将以空图启动: %s", exc)
    else:
        logger.info("/data 无 Kùzu 镜像（全新部署或未持久化过），跳过拉取")


def sync_to_storage() -> dict:
    """
    Task 1.2.4b: 统一刷盘入口。

    供变更钩子 / APScheduler 定时器 / 手动备份接口统一调用。
    一次调用内串行完成 SQLite 与 Kùzu 的原子刷盘（内部已有互斥锁）。"""
    result = {
        "sqlite": sync_sqlite(),
        "kuzu": sync_kuzu(),
        "data_dir_writable": data_dir_writable,
    }
    logger.info("统一刷盘结果: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = init_persistence()
    print(f"data_dir_writable = {result}")

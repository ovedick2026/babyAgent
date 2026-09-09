"""
SynapseMind - 防锁死持久化架构 (storage/persistence.py)

核心铁律：SQLite 与 Kùzu 必须运行在本地高速目录（避开 NFS 锁死），
再通过原子同步写回 /data 挂载盘。

【多脑布局对齐修复】（Bug1：重启后待提炼/图谱数据丢失的根因）：
本模块此前仍指向多脑改造前的旧扁平布局（sqlite/brain_state.db 与顶层 kuzu/），
而真实工作数据自 Task 2.2.1 起位于 {RUNTIME_DIR}/{brain_id}/ 之下 ——
导致每次刷盘都命中「工作库不存在，跳过」，/data 镜像从第一天起就是空的。

现布局（运行时与镜像 1:1 对应）：
  运行时:  {RUNTIME_DIR}/{brain_id}/brain.db + {RUNTIME_DIR}/{brain_id}/kuzu/
  镜像:    {DATA_DIR}/brains/{brain_id}/brain.db + {DATA_DIR}/brains/{brain_id}/kuzu/

已实现：
- Task 1.2.1  路径常量 + 工作副本目录初始化 + 挂载盘可写性探测
- Task 1.2.2  sync_sqlite(): SQLite 原生 backup API 原子刷盘
- Task 1.2.3  sync_kuzu():   Kùzu 目录原子同步
- Task 1.2.4  restore_on_boot() / sync_to_storage(): 统一入口
- BugFix      全部刷盘/自愈路径对齐多脑布局；sync_to_storage 支持按脑定向刷盘
"""

import logging
import os
import re
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("synapsemind.persistence")

# ───────────────────────────────────────────────
# 路径常量 (Path Constants)
# ───────────────────────────────────────────────

# HF 挂载盘（NFS）：状态镜像持久化目录（支持环境变量覆盖，便于本地开发测试）
DATA_DIR: Path = Path(os.environ.get("SYNAPSE_DATA_DIR", "/data"))

# 容器本地高速暂存区：毫秒级事务读写，无 NFS 锁风险
RUNTIME_DIR: Path = Path(os.environ.get("SYNAPSE_RUNTIME_DIR", "/tmp/synapse_runtime"))

# 旧扁平布局常量（多脑改造前）：仅作向后兼容保留 —— s3_backup 等模块仍将其
# 作为历史数据候选路径引用；本模块的刷盘/自愈已不再使用它们。
SQLITE_RUNTIME_DIR: Path = RUNTIME_DIR / "sqlite"
KUZU_RUNTIME_DIR: Path = RUNTIME_DIR / "kuzu"
SQLITE_DATA_DIR: Path = DATA_DIR / "sqlite"
KUZU_DATA_DIR: Path = DATA_DIR / "kuzu"

# 多脑布局镜像根目录：{DATA_DIR}/brains/{brain_id}/...
BRAINS_DATA_DIR: Path = DATA_DIR / "brains"

# brain_id 合法命名（与 core/brain_manager.BRAIN_ID_RE 判定标准一致，
# 此处独立定义以避免 storage ←→ core 循环导入）
BRAIN_ID_RE = re.compile(r"[A-Za-z0-9_\-]{1,64}")

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


def iter_runtime_brain_ids() -> List[str]:
    """
    扫描运行时目录，返回合法 brain_id 列表。

    判定标准与 core/brain_manager.discover_brains 一致：
    RUNTIME_DIR 下存在合法命名的子目录且其中含 brain.db。
    """
    if not RUNTIME_DIR.exists():
        return []
    return sorted(
        d.name
        for d in RUNTIME_DIR.iterdir()
        if d.is_dir() and BRAIN_ID_RE.fullmatch(d.name) and (d / "brain.db").is_file()
    )


def init_persistence() -> bool:
    """
    启动初始化入口（供 app.py 启动钩子调用）。

    流程：
    1. 建立 /tmp 高速工作副本根目录。
    2. 探测 /data 挂载盘可写性，可写则建立多脑镜像根目录 brains/。
    3. 不可写时降级为「仅本地模式」：仅打日志、不抛异常，保证服务始终可启动。

    Returns:
        bool: 挂载盘是否可用（True=正常持久化，False=仅本地模式）
    """
    global data_dir_writable

    # 1) 本地高速工作盘（必建，系统运行的最低要求）
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("本地高速工作盘就绪: %s", RUNTIME_DIR)

    # 2) 挂载盘探测与镜像根目录建立
    data_dir_writable = _probe_writability(BRAINS_DATA_DIR)
    if data_dir_writable:
        BRAINS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("持久挂载盘就绪，多脑镜像根目录已建立: %s", BRAINS_DATA_DIR)
    else:
        logger.warning(
            "⚠️ /data 挂载盘不可写，已降级为【仅本地模式】："
            "数据仅存在于 %s，容器重启后可能丢失！",
            RUNTIME_DIR,
        )

    return data_dir_writable


# ───────────────────────────────────────────────
# 单脑刷盘原子操作
# ───────────────────────────────────────────────


def _sync_sqlite_for(brain_id: str) -> bool:
    """
    单脑 SQLite 原子刷盘。

    源:   {RUNTIME_DIR}/{brain_id}/brain.db
    目标: {DATA_DIR}/brains/{brain_id}/brain.db

    使用 SQLite 原生 backup API 逐页复制一致性快照到 /data 同目录临时文件，
    再以 os.replace 原子替换 —— 任何时刻镜像要么是完整旧态、要么是完整新态。
    """
    src_path = RUNTIME_DIR / brain_id / "brain.db"
    if not src_path.exists():
        logger.info("brain=%s 工作库不存在，跳过 SQLite 刷盘", brain_id)
        return True  # 无可刷内容不算失败

    brain_mirror = BRAINS_DATA_DIR / brain_id
    brain_mirror.mkdir(parents=True, exist_ok=True)
    final_path = brain_mirror / "brain.db"
    # 临时文件与目标同目录（同分区），保证 os.replace 的原子性
    tmp_path = brain_mirror / ".brain.db.tmp"

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
        logger.error("brain=%s SQLite 刷盘失败，/data 旧镜像保持不动: %s", brain_id, exc)
        # 清理残留临时文件，避免下次误判
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _sync_kuzu_for(brain_id: str) -> bool:
    """
    单脑 Kùzu 目录原子同步。

    源:   {RUNTIME_DIR}/{brain_id}/kuzu（形态自适应：文件或目录均可）
    目标: {DATA_DIR}/brains/{brain_id}/kuzu

    Kùzu 存储形态随版本演进而异（≥0.11 落为单文件，旧版为目录），
    因此刷盘策略按源形态自适应：
      - 源为文件：同分区临时文件拷贝 + os.replace 原子换入；
      - 源为目录：目录级三步换入（staging → 旧镜像改名回滚位 → rename 换入）。
    rename/replace 均要求临时产物与目标同分区，这是原子性的根基。
    """
    # 0) 尽力 CHECKPOINT：Kùzu ≥0.11 默认 WAL 模式，commit 后数据滞留 kuzu.wal，
    #    主文件可能是空壳 —— 不 checkpoint 就拷贝，镜像必残缺。
    #    延迟导入规避模块级环依赖（brain_manager → persistence）；
    #    写请求线程的 TLS 连接恰是最新写连接，主路径必命中。
    try:
        from core.brain_manager import checkpoint_brain
        checkpoint_brain(brain_id)
    except Exception:
        pass  # 任何失败交由下方 WAL 镜像兜底

    src_path = RUNTIME_DIR / brain_id / "kuzu"
    src_wal = RUNTIME_DIR / brain_id / "kuzu.wal"
    if not src_path.exists():
        logger.info("brain=%s Kùzu 运行时库不存在，跳过刷盘", brain_id)
        return True  # 无可刷内容不算失败
    if src_path.is_dir() and not any(src_path.iterdir()):
        logger.info("brain=%s Kùzu 运行时目录为空，跳过刷盘", brain_id)
        return True

    brain_mirror = BRAINS_DATA_DIR / brain_id
    brain_mirror.mkdir(parents=True, exist_ok=True)
    final_path = brain_mirror / "kuzu"
    final_wal = brain_mirror / "kuzu.wal"

    try:
        if src_path.is_file():
            # 单文件形态（Kùzu ≥0.11）：同分区临时文件 + os.replace 原子替换
            fd, tmp_name = tempfile.mkstemp(prefix=".kuzu_", dir=str(brain_mirror))
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                shutil.copyfile(src_path, tmp_path)
                os.replace(tmp_path, final_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
            # WAL 兜底镜像：checkpoint 失败时数据仍在 wal，一并刷出
            # （恢复时 Kùzu 自动 replay）；源无 wal 则清掉镜像旧 wal，
            # 防止恢复时把陈旧 WAL replay 到新主文件上。
            if src_wal.is_file() and src_wal.stat().st_size > 0:
                fd2, tmp2_name = tempfile.mkstemp(prefix=".kuzuwal_", dir=str(brain_mirror))
                os.close(fd2)
                tmp_wal = Path(tmp2_name)
                try:
                    shutil.copyfile(src_wal, tmp_wal)
                    os.replace(tmp_wal, final_wal)
                except BaseException:
                    tmp_wal.unlink(missing_ok=True)
                    raise
            elif final_wal.exists():
                final_wal.unlink()
            logger.info("Kùzu 单文件原子同步完成: %s → %s", src_path, final_path)
        else:
            # 目录形态（旧版 Kùzu）：三步换入
            staging_path = brain_mirror / ".kuzu_staging_tmp"
            rollback_path = brain_mirror / ".kuzu_rollback_old"
            shutil.rmtree(staging_path, ignore_errors=True)
            shutil.copytree(src_path, staging_path)
            shutil.rmtree(rollback_path, ignore_errors=True)
            if final_path.exists():
                os.rename(final_path, rollback_path)
            os.rename(staging_path, final_path)
            shutil.rmtree(rollback_path, ignore_errors=True)
            logger.info("Kùzu 目录原子同步完成: %s → %s", src_path, final_path)
        return True
    except (OSError, shutil.Error) as exc:
        logger.error("brain=%s Kùzu 同步失败，目标保持旧镜像: %s", brain_id, exc)
        # 目录形态下尽力回滚（文件形态 os.replace 天然无半写状态，无需回滚）
        staging_path = brain_mirror / ".kuzu_staging_tmp"
        rollback_path = brain_mirror / ".kuzu_rollback_old"
        try:
            if not final_path.exists() and rollback_path.exists():
                os.rename(rollback_path, final_path)
                logger.warning("已回滚恢复 brain=%s 的旧 Kùzu 镜像", brain_id)
            shutil.rmtree(staging_path, ignore_errors=True)
        except OSError as rollback_exc:
            logger.critical("回滚失败，请人工检查 %s: %s", brain_mirror, rollback_exc)
        return False


# ───────────────────────────────────────────────
# 公开刷盘入口（保持向后兼容的聚合签名）
# ───────────────────────────────────────────────


def sync_sqlite(brain_id: Optional[str] = None) -> bool:
    """
    SQLite 原子刷盘（多脑聚合入口）。

    Args:
        brain_id: 指定则只刷该脑；None（默认）刷运行时全部合法脑。

    Returns:
        True=全部目标刷盘成功（含无可刷内容的空转）；False=任一失败。
    """
    if not data_dir_writable:
        logger.warning("挂载盘不可用（仅本地模式），跳过 SQLite 刷盘")
        return False

    targets = [brain_id] if brain_id else iter_runtime_brain_ids()
    if not targets:
        logger.info("运行时无任何脑数据，跳过 SQLite 刷盘")
        return True

    with _sync_lock:
        results = [_sync_sqlite_for(b) for b in targets]
    return all(results)


def sync_kuzu(brain_id: Optional[str] = None) -> bool:
    """
    Kùzu 目录原子同步（多脑聚合入口）。

    Args:
        brain_id: 指定则只刷该脑；None（默认）刷运行时全部合法脑。

    Returns:
        True=全部目标同步成功（含无可刷内容的空转）；False=任一失败。
    """
    if not data_dir_writable:
        logger.warning("挂载盘不可用（仅本地模式），跳过 Kùzu 刷盘")
        return False

    targets = [brain_id] if brain_id else iter_runtime_brain_ids()
    if not targets:
        logger.info("运行时无任何脑数据，跳过 Kùzu 刷盘")
        return True

    with _sync_lock:
        results = [_sync_kuzu_for(b) for b in targets]
    return all(results)


def restore_on_boot(brain_id: Optional[str] = None) -> None:
    """
    容器重启自愈拉取（多脑）。

    HF Space 容器重启后 /tmp 会被清空，本地工作副本丢失。
    本函数在启动时从 /data 多脑镜像拉回上次的持久化状态：
      - {DATA_DIR}/brains/{id}/brain.db → 拷回 {RUNTIME_DIR}/{id}/brain.db；
      - {DATA_DIR}/brains/{id}/kuzu → 形态自适应拉回（文件/目录均可，Kùzu ≥0.11 为单文件）。
    镜像不存在（全新部署）时静默跳过，由各引擎自行空库初始化。

    Args:
        brain_id: 指定则只拉取该脑；None（默认）拉取镜像内全部脑。
    """
    if not BRAINS_DATA_DIR.exists():
        logger.info("/data 无多脑镜像目录（全新部署或未持久化过），跳过自愈拉取")
        return

    if brain_id:
        candidates = [brain_id] if (BRAINS_DATA_DIR / brain_id).exists() else []
    else:
        candidates = sorted(
            d.name
            for d in BRAINS_DATA_DIR.iterdir()
            if d.is_dir() and BRAIN_ID_RE.fullmatch(d.name)
        )

    for bid in candidates:
        mirror = BRAINS_DATA_DIR / bid
        work_dir = RUNTIME_DIR / bid
        try:
            work_dir.mkdir(parents=True, exist_ok=True)

            # SQLite：整文件拉回
            mirror_db = mirror / "brain.db"
            if mirror_db.is_file():
                shutil.copy2(mirror_db, work_dir / "brain.db")
                logger.info("brain=%s SQLite 镜像自愈拉取完成", bid)

            # Kùzu：形态自适应拉回（先清运行时旧位，防文件/目录形态混杂）
            mirror_kuzu = mirror / "kuzu"
            mirror_wal = mirror / "kuzu.wal"
            runtime_wal = work_dir / "kuzu.wal"
            if mirror_kuzu.is_file():
                dst = work_dir / "kuzu"
                dst.unlink(missing_ok=True)
                shutil.copy2(mirror_kuzu, dst)
                # WAL 兜底拉回（与 _sync_kuzu_for 的 WAL 镜像严格对称）：
                # checkpoint 未刷净的事务滞留在 wal，恢复时 Kùzu 自动 replay；
                # 镜像无 wal 则清掉运行时旧 wal，防止把陈旧 WAL replay 到新主文件上。
                if mirror_wal.is_file() and mirror_wal.stat().st_size > 0:
                    shutil.copy2(mirror_wal, runtime_wal)
                    logger.info("brain=%s Kùzu WAL 镜像自愈拉取完成", bid)
                else:
                    runtime_wal.unlink(missing_ok=True)
                logger.info("brain=%s Kùzu 单文件镜像自愈拉取完成", bid)
            elif mirror_kuzu.is_dir() and any(mirror_kuzu.iterdir()):
                dst = work_dir / "kuzu"
                if dst.is_dir():
                    shutil.rmtree(dst, ignore_errors=True)
                else:
                    dst.unlink(missing_ok=True)
                shutil.copytree(mirror_kuzu, dst)
                logger.info("brain=%s Kùzu 目录镜像自愈拉取完成", bid)
        except (OSError, shutil.Error) as exc:
            logger.error("brain=%s 自愈拉取失败，将以空库/空图启动: %s", bid, exc)

    if not candidates:
        logger.info("/data 多脑镜像内无可用脑数据，跳过自愈拉取")


def sync_to_storage(brain_id: Optional[str] = None) -> dict:
    """
    统一刷盘入口。

    供变更钩子 / APScheduler 定时器 / 手动备份接口统一调用。
    一次调用内串行完成 SQLite 与 Kùzu 的原子刷盘（内部已有互斥锁）。

    Args:
        brain_id: 指定则只刷该脑（投喂/收纳等单脑变更场景的定向刷盘）；
                  None（默认）刷运行时全部脑（定时兜底/切换大脑场景）。
    """
    result = {
        "sqlite": sync_sqlite(brain_id),
        "kuzu": sync_kuzu(brain_id),
        "brain_id": brain_id,
        "data_dir_writable": data_dir_writable,
    }
    logger.info("统一刷盘结果: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = init_persistence()
    print(f"data_dir_writable = {ok}")
    print(f"runtime brains = {iter_runtime_brain_ids()}")

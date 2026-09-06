"""
SynapseMind - S3 快照备份模块 (storage/s3_backup.py)

备份体系铁律：
- manual / hourly / daily 三种前缀各自严格保留 ≤3 份（轮转逻辑见 Task 1.3.3）。
- 快照为自包含 tar.gz：内含该 brain 的 SQLite 库文件 + Kùzu 图数据库目录，
  恢复时不依赖任何外部状态。

本文件当前实现：
- Task 1.3.1  pack_snapshot(brain_id): 快照打包
  （tar.gz 临时文件，文件名含 brain_id 与 UTC 时间戳，便于排序与轮转）

后续任务将在此文件追加：
- Task 1.3.2  S3 客户端初始化与 upload_snapshot / download_snapshot
- Task 1.3.3  rotate_prefix: 按前缀轮转清理（保留最新 3 份）
- Task 1.3.4  APScheduler 定时备份注册
- Task 1.3.5  manual_backup / restore_from_cloud
"""

import logging
import re
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from storage.persistence import (
    RUNTIME_DIR,
    SQLITE_RUNTIME_DIR,
    KUZU_RUNTIME_DIR,
)

logger = logging.getLogger("synapsemind.s3_backup")

# 快照临时文件目录（上传 S3 后即可删除，不占用持久盘）
SNAPSHOT_TMP_DIR: Path = Path(tempfile.gettempdir()) / "synapse_snapshots"

# S3 对象前缀规范（供 Task 1.3.2 / 1.3.3 使用）：
#   backups/{brain_id}/manual|hourly|daily/snapshot_{brain_id}_{ts}.tar.gz
S3_BACKUP_ROOT = "backups"


def _brain_sqlite_candidates(brain_id: str):
    """按优先级给出该 brain 的 SQLite 工作库可能路径。"""
    yield RUNTIME_DIR / brain_id / "brain.db"      # 多脑目录布局（Task 2.2.1 落地后）
    yield SQLITE_RUNTIME_DIR / f"{brain_id}.db"    # 扁平布局：每脑一库


def _brain_kuzu_candidates(brain_id: str):
    """按优先级给出该 brain 的 Kùzu 图目录可能路径。"""
    yield RUNTIME_DIR / brain_id / "kuzu"          # 多脑目录布局
    yield KUZU_RUNTIME_DIR / brain_id              # 扁平布局：每脑一图


def _locate_brain_sources(brain_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """定位 brain 的 SQLite 库与 Kùzu 目录，返回 (sqlite_path|None, kuzu_dir|None)。"""
    sqlite_path: Optional[Path] = None
    for cand in _brain_sqlite_candidates(brain_id):
        if cand.is_file():
            sqlite_path = cand
            break

    kuzu_dir: Optional[Path] = None
    for cand in _brain_kuzu_candidates(brain_id):
        if cand.is_dir() and any(cand.iterdir()):
            kuzu_dir = cand
            break

    return sqlite_path, kuzu_dir


def pack_snapshot(brain_id: str) -> Optional[Path]:
    """
    Task 1.3.1: 快照打包。

    将指定 brain 的 SQLite 库文件与 Kùzu 图目录压缩为单个自包含 tar.gz：
      - 归档内路径: sqlite/<库名>  与  kuzu/<图目录名>/...（恢复时按此映射回写）
      - 文件名: snapshot_{brain_id}_{UTC时间戳}.tar.gz
        （UTC 时间戳字典序即时间序，天然支持下游轮转按名排序取最新 3 份）

    注意：为保证快照一致性，调用方（如 manual_backup / 定时任务）应先执行
    sync_to_storage() 或确保工作库已静默，再调用本函数。

    Args:
        brain_id: 目标脑 id（仅允许字母/数字/下划线/连字符，防路径穿越）。

    Returns:
        快照文件 Path；该 brain 无任何可打包内容时返回 None；
        打包失败时清理半成品并返回 None（不抛出，交由调用方决定重试策略）。
    """
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", brain_id):
        raise ValueError(f"非法 brain_id: {brain_id!r}")

    sqlite_path, kuzu_dir = _locate_brain_sources(brain_id)
    if sqlite_path is None and kuzu_dir is None:
        logger.info("brain=%s 无可打包内容（无 SQLite 库且无 Kùzu 图），跳过快照", brain_id)
        return None

    SNAPSHOT_TMP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = SNAPSHOT_TMP_DIR / f"snapshot_{brain_id}_{ts}.tar.gz"

    try:
        with tarfile.open(snapshot_path, "w:gz") as tar:
            if sqlite_path is not None:
                tar.add(sqlite_path, arcname=f"sqlite/{sqlite_path.name}")
                logger.info("快照纳入 SQLite 库: %s", sqlite_path)
            if kuzu_dir is not None:
                tar.add(kuzu_dir, arcname=f"kuzu/{kuzu_dir.name}")
                logger.info("快照纳入 Kùzu 目录: %s", kuzu_dir)

        size_kb = snapshot_path.stat().st_size / 1024
        logger.info("快照打包完成: %s (%.1f KB)", snapshot_path.name, size_kb)
        return snapshot_path
    except (OSError, tarfile.TarError) as exc:
        logger.error("快照打包失败，已清理半成品: %s", exc)
        try:
            snapshot_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


# ───────────────────────────────────────────────
# S3 客户端与上传/下载 (Task 1.3.2)
# ───────────────────────────────────────────────

# 合法的备份类别（同时用于前缀规范与轮转分组）
BACKUP_KINDS = ("manual", "hourly", "daily")

# s3_cfg 字段约定（待 Task 2.1.1 的 config.py 落地后由调用方注入）：
#   endpoint_url      S3 兼容端点（R2/OSS/MinIO 均适用）
#   access_key_id     访问密钥
#   secret_access_key 私密密钥
#   bucket            目标桶名
S3_CFG_REQUIRED_KEYS = ("endpoint_url", "access_key_id", "secret_access_key", "bucket")


def _build_client(s3_cfg: dict):
    """基于 boto3 构建 S3 兼容客户端（懒加载 boto3，未配置时快速失败）。"""
    missing = [k for k in S3_CFG_REQUIRED_KEYS if not s3_cfg.get(k)]
    if missing:
        raise ValueError(f"S3 配置缺失字段: {missing}")

    import boto3  # 延迟导入：仅在真正需要联网备份时加载

    return boto3.client(
        "s3",
        endpoint_url=s3_cfg["endpoint_url"],
        aws_access_key_id=s3_cfg["access_key_id"],
        aws_secret_access_key=s3_cfg["secret_access_key"],
    )


def _validate_kind(kind: str) -> None:
    if kind not in BACKUP_KINDS:
        raise ValueError(f"非法备份类别 kind={kind!r}，仅允许 {BACKUP_KINDS}")


def _prefix_for(brain_id: str, kind: str) -> str:
    """前缀规范：backups/{brain_id}/manual|hourly|daily/"""
    _validate_kind(kind)
    return f"{S3_BACKUP_ROOT}/{brain_id}/{kind}/"


def upload_snapshot(brain_id: str, kind: str, snapshot_path: Path, s3_cfg: dict) -> str:
    """
    Task 1.3.2a: 上传快照至 S3 兼容对象存储。

    对象键严格遵循前缀规范，快照文件名自带 UTC 时间戳（Task 1.3.1）。

    Returns:
        上传后的对象键（供调用方记录/审计）。
    """
    _validate_kind(kind)
    client = _build_client(s3_cfg)
    key = f"{_prefix_for(brain_id, kind)}{Path(snapshot_path).name}"
    client.upload_file(str(snapshot_path), s3_cfg["bucket"], key)
    logger.info("快照已上传: %s → %s", snapshot_path, key)
    return key


def download_snapshot(
    brain_id: str, kind: str, s3_cfg: dict, key: Optional[str] = None
) -> Optional[Path]:
    """
    Task 1.3.2b: 从云端下载快照到本地临时目录。

    Args:
        key: 指定对象键；缺省时自动选取该前缀下最新一份
             （对象名含 UTC 时间戳，字典序倒序即最新，与轮转逻辑天然一致）。

    Returns:
        本地快照文件 Path；云端该类别无任何归档时返回 None。
    """
    _validate_kind(kind)
    client = _build_client(s3_cfg)
    bucket = s3_cfg["bucket"]
    prefix = _prefix_for(brain_id, kind)

    if key is None:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        objects = [o for o in resp.get("Contents", []) if o["Key"].endswith(".tar.gz")]
        if not objects:
            logger.info("云端无归档: prefix=%s", prefix)
            return None
        key = max(o["Key"] for o in objects)

    SNAPSHOT_TMP_DIR.mkdir(parents=True, exist_ok=True)
    local_path = SNAPSHOT_TMP_DIR / Path(key).name
    client.download_file(bucket, key, str(local_path))
    logger.info("快照已下载: %s → %s", key, local_path)
    return local_path


def rotate_prefix(brain_id: str, kind: str, s3_cfg: dict, keep: int = 3) -> int:
    """
    Task 1.3.3: 按前缀轮转清理，保证该类别归档严格 ≤ keep 份（默认 3）。

    规则：
      - 列出 `backups/{brain_id}/{kind}/` 前缀下全部 .tar.gz 归档；
      - 对象名含 UTC 时间戳（Task 1.3.1），字典序倒序即时间倒序，
        按此保留最新 keep 份，其余更旧归档全部删除；
      - 每次备份（manual / hourly / daily）成功后由调用方自动触发本函数。

    Args:
        brain_id: 目标脑 id。
        kind:     备份类别（manual/hourly/daily）。
        s3_cfg:   S3 连接配置（见 S3_CFG_REQUIRED_KEYS）。
        keep:     保留的最新份数，默认 3。

    Returns:
        本次实际删除的归档数量。
    """
    _validate_kind(kind)
    if keep < 1:
        raise ValueError(f"keep 必须 ≥ 1，当前为 {keep!r}")

    client = _build_client(s3_cfg)
    bucket = s3_cfg["bucket"]
    prefix = _prefix_for(brain_id, kind)

    # 分页列出全部归档（避免单次 list 1000 条上限截断）
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".tar.gz"):
                keys.append(obj["Key"])

    if len(keys) <= keep:
        logger.info("轮转检查: prefix=%s 共 %d 份 ≤ %d 份，无需清理", prefix, len(keys), keep)
        return 0

    keys.sort(reverse=True)          # 字典序倒序 = 时间倒序
    stale_keys = keys[keep:]         # 保留最新 keep 份，其余为待删除

    client.delete_objects(
        Bucket=bucket,
        Delete={"Objects": [{"Key": k} for k in stale_keys], "Quiet": True},
    )
    for k in stale_keys:
        logger.info("轮转删除过期归档: %s", k)
    logger.info("轮转完成: prefix=%s 删除 %d 份，保留最新 %d 份", prefix, len(stale_keys), keep)
    return len(stale_keys)


# ───────────────────────────────────────────────
# APScheduler 定时备份注册 (Task 1.3.4)
# ───────────────────────────────────────────────

import threading
from collections import defaultdict

# 每脑一把备份互斥锁：定时任务与手动备份（Task 1.3.5）共用，
# 同一 brain 同一时刻只允许一个备份流程在跑，冲突方直接跳过当次，绝不排队阻塞。
_backup_locks: dict = defaultdict(threading.Lock)


def _resolve_cfg(source):
    """配置既可以是 dict，也可以是返回 dict 的可调用（触发时刻实时求值，支持运行时改配置）。"""
    return source() if callable(source) else source


def run_backup_once(brain_id: str, kind: str, s3_cfg) -> bool:
    """
    单脑单类别备份闭环：打包 → 上传 → 轮转。

    并发防护：与手动备份共用每脑互斥锁，若该 brain 已有备份在执行
    （无论手动还是定时），本次直接跳过并记日志，避免快照撕裂或重复上传。

    Args:
        brain_id: 目标脑 id。
        kind:     备份类别（manual/hourly/daily）。
        s3_cfg:   S3 配置 dict 或返回 dict 的可调用。

    Returns:
        True=备份成功；False=跳过（并发冲突 / 无可打包内容）。
    """
    s3_cfg = _resolve_cfg(s3_cfg)
    lock = _backup_locks[brain_id]
    if not lock.acquire(blocking=False):
        logger.info("brain=%s 已有备份进行中，跳过本次 kind=%s 备份", brain_id, kind)
        return False

    snapshot_path: Optional[Path] = None
    try:
        snapshot_path = pack_snapshot(brain_id)
        if snapshot_path is None:
            return False
        upload_snapshot(brain_id, kind, snapshot_path, s3_cfg)
        rotate_prefix(brain_id, kind, s3_cfg)
        return True
    finally:
        # 临时快照上传后即清理，不残留占用本地盘
        if snapshot_path is not None:
            try:
                snapshot_path.unlink(missing_ok=True)
            except OSError:
                pass
        lock.release()


def run_scheduled_backups(brain_ids, kind: str, s3_cfg) -> dict:
    """
    对多个 brain 依次执行定时备份。

    单脑异常只记日志不影响其余脑（多脑隔离，调度线程永不被单点故障拖垮）。

    Returns:
        {brain_id: 是否成功} 结果表。
    """
    results: dict = {}
    for brain_id in brain_ids:
        try:
            results[brain_id] = run_backup_once(brain_id, kind, s3_cfg)
        except Exception as exc:  # noqa: BLE001 单脑失败必须与其它脑隔离
            logger.error("brain=%s 的 %s 定时备份异常: %s", brain_id, kind, exc)
            results[brain_id] = False
    return results


def register_backup_jobs(scheduler, list_brain_ids, s3_cfg, backup_policy) -> list:
    """
    Task 1.3.4: 向 APScheduler 注册定时备份任务。

    - 每小时任务：整点触发，受 backup_policy.hourly_enabled 开关控制；
    - 每日任务：默认每日 03:00 触发（可用 backup_policy.daily_hour 覆盖），
      受 backup_policy.daily_enabled 开关控制。

    Args:
        scheduler:      apscheduler.schedulers.base.BaseScheduler 实例
                        （由 app.py 启动钩子创建并传入，见 Task 7.1）。
        list_brain_ids: 返回当前全部 brain_id 列表的可调用；每次触发时实时求值，
                        天然兼容 Task 2.2.2 之后动态增删的脑。
        s3_cfg:         S3 配置 dict 或返回 dict 的可调用（config.py 落地后注入）。
        backup_policy:  备份策略 dict 或返回 dict 的可调用。

    Returns:
        实际注册成功的任务类别列表（如 ["hourly", "daily"]）。
    """
    policy = _resolve_cfg(backup_policy)
    registered: list = []

    if policy.get("hourly_enabled"):
        scheduler.add_job(
            run_scheduled_backups,
            trigger="cron",
            minute=0,  # 每小时整点
            args=[lambda: list_brain_ids(), "hourly", s3_cfg],
            id="backup_hourly",
            replace_existing=True,
            misfire_grace_time=3600,
            coalesce=True,  # 错过多次触发只补一次，防止积压风暴
        )
        registered.append("hourly")
        logger.info("已注册每小时备份任务（整点触发）")

    if policy.get("daily_enabled"):
        daily_hour = int(policy.get("daily_hour", 3))
        scheduler.add_job(
            run_scheduled_backups,
            trigger="cron",
            hour=daily_hour,
            minute=0,
            args=[lambda: list_brain_ids(), "daily", s3_cfg],
            id="backup_daily",
            replace_existing=True,
            misfire_grace_time=3600,
            coalesce=True,
        )
        registered.append("daily")
        logger.info("已注册每日备份任务（每日 %02d:00 触发）", daily_hour)

    if not registered:
        logger.info("backup_policy 未启用任何定时备份（hourly/daily 均关闭），跳过注册")
    return registered


# ───────────────────────────────────────────────
# 手动备份与一键云端恢复 (Task 1.3.5)
# ───────────────────────────────────────────────

import os
import shutil

# 以模块属性方式活引用 persistence：data_dir_writable 是运行期可变开关，
# 必须在调用时读取最新值（from-import 会把布尔值冻结在导入时刻）。
from storage import persistence as _persistence


def manual_backup(brain_id: str, s3_cfg) -> bool:
    """
    Task 1.3.5a: 手动备份。

    立即 打包 → 上传（manual 前缀）→ 轮转（保留最新 3 份）。
    实现复用 run_backup_once：与定时备份共用每脑互斥锁，并发时
    后到方直接跳过当次，绝不排队阻塞。

    Returns:
        True=备份成功；False=跳过（并发冲突 / 无可打包内容）。
    """
    return run_backup_once(brain_id, "manual", s3_cfg)


def _verify_snapshot_members(tar: tarfile.TarFile) -> list:
    """
    校验归档合法性（防御损坏包与恶意构造包）：
      - 成员仅允许 sqlite/ 或 kuzu/ 前缀（Task 1.3.1 的自包含格式）；
      - 禁止绝对路径与 .. 穿越，拒绝向目标目录之外写任何文件。
    校验失败抛 TarError，由调用方统一处理、放弃全部写回（要么不写、要写全写）。
    """
    members = tar.getmembers()
    if not members:
        raise tarfile.TarError("归档为空")
    has_valid = False
    for m in members:
        name = m.name
        if name.startswith("/") or ".." in Path(name).parts:
            raise tarfile.TarError(f"非法归档路径: {name}")
        if name.startswith(("sqlite/", "kuzu/")):
            has_valid = True
        else:
            raise tarfile.TarError(f"意外归档成员: {name}")
    if not has_valid:
        raise tarfile.TarError("归档不含 SQLite/Kùzu 内容")
    return members


def _writeback_sqlite(tar: tarfile.TarFile, members: list, dst_root: Path) -> int:
    """将归档中 sqlite/<库名> 成员原子写回 dst_root（同分区临时文件 + os.replace）。"""
    restored = 0
    for m in members:
        if not (m.isfile() and m.name.startswith("sqlite/")):
            continue
        db_name = Path(m.name).name
        if not db_name:
            continue
        dst_root.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst_root / f".{db_name}.restore_tmp"
        with tar.extractfile(m) as src, open(tmp_dst, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp_dst, dst_root / db_name)
        restored += 1
        logger.info("SQLite 镜像已恢复: %s", dst_root / db_name)
    return restored


def _writeback_kuzu(tar: tarfile.TarFile, members: list, dst_root: Path) -> None:
    """
    将归档中 kuzu/<图目录> 成员原子写回 dst_root 下的对应子图目录。
    与 persistence.sync_kuzu 同款三步换入：失败回滚旧镜像，绝不留半恢复状态。
    """
    kuzu_members = [m for m in members if m.name.startswith("kuzu/")]
    if not kuzu_members:
        return

    # 归档内顶层图目录名（kuzu/{brain_id}/... 中的 brain_id 段）
    graph_dir = kuzu_members[0].name.split("/")[1]
    if not graph_dir:
        raise tarfile.TarError("无法确定归档内的图目录名")

    dst_root.mkdir(parents=True, exist_ok=True)
    final = dst_root / graph_dir
    # 暂存与回滚目录均置于 dst_root 同分区，保证 rename 原子性
    staging = dst_root.parent / f".{dst_root.name}.{graph_dir}.restore_staging"
    rollback = dst_root.parent / f".{dst_root.name}.{graph_dir}.restore_old"

    try:
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for m in kuzu_members:
            if not m.isfile():
                continue
            rel = Path(m.name).relative_to(f"kuzu/{graph_dir}")
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

        # 三步换入（同分区 rename 保证原子性）
        shutil.rmtree(rollback, ignore_errors=True)
        if final.exists():
            os.rename(final, rollback)
        os.rename(staging, final)
        shutil.rmtree(rollback, ignore_errors=True)
        logger.info("Kùzu 镜像已恢复: %s", final)
    except (OSError, shutil.Error):
        shutil.rmtree(staging, ignore_errors=True)
        if not final.exists() and rollback.exists():
            os.rename(rollback, final)
            logger.warning("Kùzu 恢复失败，已回滚旧镜像: %s", final)
        raise


def restore_from_cloud(brain_id: str, kind: str, s3_cfg) -> bool:
    """
    Task 1.3.5b: 一键云端恢复。

    流程：
      1. 下载 kind（manual/hourly/daily）前缀下指定的最新一份归档；
      2. 校验归档完整性与路径安全（先校验后落盘，绝不部分写盘）；
      3. 原子覆盖 /data 镜像（挂载盘不可用时直接写运行时目录）；
      4. 挂载盘可用时触发 restore_on_boot() 把镜像自愈拉回 /tmp 工作副本；
         仅本地模式下 /tmp 已直接写好，不再触发拉取（避免被 /data 旧镜像覆盖）。

    注意：应在该 brain 会话尚未打开库连接时调用（Task 2.2.x 落地后需
    先 close_brain 再恢复，防止活句柄读写被替换的文件）。

    Returns:
        True=恢复成功；False=云端无归档 / 校验或写回失败。
    """
    _validate_kind(kind)
    local_path = download_snapshot(brain_id, kind, s3_cfg)
    if local_path is None:
        logger.info("云端无可用归档，恢复中止: brain=%s kind=%s", brain_id, kind)
        return False

    writable = _persistence.data_dir_writable
    sqlite_dst = _persistence.SQLITE_DATA_DIR if writable else _persistence.SQLITE_RUNTIME_DIR
    kuzu_dst = _persistence.KUZU_DATA_DIR if writable else _persistence.KUZU_RUNTIME_DIR

    try:
        with tarfile.open(local_path, "r:gz") as tar:
            members = _verify_snapshot_members(tar)
            _writeback_sqlite(tar, members, sqlite_dst)
            _writeback_kuzu(tar, members, kuzu_dst)

        if writable:
            _persistence.restore_on_boot()
            logger.info("云端恢复完成并已触发自愈拉取: brain=%s kind=%s", brain_id, kind)
        else:
            logger.warning(
                "云端恢复完成（仅本地模式，直接写入运行时目录）: brain=%s kind=%s",
                brain_id, kind,
            )
        return True
    except (tarfile.TarError, OSError, shutil.Error) as exc:
        logger.error("云端恢复失败，目标目录尽量保持恢复前状态: %s", exc)
        return False
    finally:
        try:
            local_path.unlink(missing_ok=True)
        except OSError:
            pass

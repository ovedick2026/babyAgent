"""
SynapseMind - 系统配置中心 (config.py)

Task 2.1.1: config.json 原子读写与默认配置生成

设计要点：
- 配置文件固定存放于 /data/config.json（挂载盘持久化，容器重启不丢配置）；
  挂载盘不可用（仅本地模式）时回退到 /tmp 运行时目录。
- 写入采用「同分区临时文件 + os.replace 原子替换」，任何时刻配置文件
  要么是完整旧内容、要么是完整新内容，绝无半写状态。
- 读取失败（文件损坏 / JSON 解析错误）时回退全默认配置，绝不因配置问题
  阻塞服务启动。
- 已存配置缺失新版本新增键时，自动用默认值深合并补齐（前向兼容升级）。

本文件当前实现：
- Task 2.1.1  load_config / save_config / 默认配置生成

后续任务将在此文件追加：
- Task 2.1.2  set_password / verify_password / api_token 生成与校验
- Task 2.1.3  get_mode/set_mode、get_llm/set_llm、get_s3/set_s3 读写接口
"""

import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

# 以模块属性方式引用 persistence：data_dir_writable 是运行期可变开关，
# 必须在调用时读取最新值（from-import 会把布尔值冻结在导入时刻）。
from storage import persistence as _persistence

logger = logging.getLogger("synapsemind.config")

# 配置读写互斥锁：防止并发写导致交错覆盖
_CONFIG_LOCK = threading.Lock()

CONFIG_FILENAME = "config.json"


def _default_config() -> dict:
    """
    全量默认配置（每次返回全新深拷贝，避免调用方修改污染模板）。

    结构约定（与 Readme/TODO 原始蓝图一致）：
      security:      password_hash（bcrypt，Task 2.1.2 落地）与 api_token
      mode:          manual / assisted / augmented 三种运行模式
      llm:           OpenAI 兼容端点配置（手工模式允许留空）
      s3:            S3 兼容对象存储连接参数（R2/OSS/MinIO 均适用）
      backup_policy: 定时备份开关（hourly/daily 各保留 ≤3 份，Task 1.3.4）
    """
    return {
        "security": {
            "password_hash": None,   # 首次未设密码 → 由引导/设置页设置（Task 2.1.2）
            "api_token": None,       # OpenAI 兼容接口 Bearer Token
        },
        "mode": "manual",            # 默认纯手工模式：零外部依赖、断网可用
        "llm": {
            "base_url": "",          # OpenAI 兼容 /v1 根地址
            "api_key": "",
            "model": "",
        },
        "s3": {
            "endpoint_url": "",
            "access_key_id": "",
            "secret_access_key": "",
            "bucket": "",
        },
        "backup_policy": {
            "hourly_enabled": False,
            "daily_enabled": False,
            "daily_hour": 3,         # 每日快照默认 03:00 触发（Task 1.3.4）
        },
    }


def config_path() -> Path:
    """配置文件路径：挂载盘可用时存 /data（重启不丢），否则回退运行时目录。"""
    if _persistence.data_dir_writable:
        return _persistence.DATA_DIR / CONFIG_FILENAME
    return _persistence.RUNTIME_DIR / CONFIG_FILENAME


def _deep_merge(base: dict, override: dict) -> dict:
    """以 base 默认值打底、override 已存值覆盖的递归深合并（补齐升级新增键）。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _atomic_write_json(path: Path, cfg: dict) -> None:
    """同分区临时文件 + fsync + os.replace 原子写入 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, ensure_ascii=False, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp_path, path)


def load_config() -> dict:
    """
    Task 2.1.1a: 读取配置。

    - 文件不存在（首次运行）→ 生成并落盘全默认配置；
    - 读取/解析失败 → 打日志并回退全默认配置（绝不抛出阻塞启动）；
    - 已存配置缺新键 → 深合并默认值补齐（前向兼容）。

    Returns:
        配置 dict（调用方可直接修改后交 save_config 持久化）。
    """
    with _CONFIG_LOCK:
        path = config_path()

        if not path.exists():
            cfg = _default_config()
            try:
                _atomic_write_json(path, cfg)
                logger.info("首次运行，已生成默认配置: %s", path)
            except OSError as exc:
                logger.warning("默认配置落盘失败（内存默认值继续运行）: %s", exc)
            return cfg

        try:
            with open(path, "r", encoding="utf-8") as fp:
                stored = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("配置读取失败，回退全默认配置（原文件保留待人工检查）: %s", exc)
            return _default_config()

        if not isinstance(stored, dict):
            logger.error("配置根节点非 dict（%s），回退全默认配置", type(stored).__name__)
            return _default_config()

        return _deep_merge(_default_config(), stored)


def save_config(cfg: dict) -> bool:
    """
    Task 2.1.1b: 原子持久化配置。

    优先写挂载盘；挂载盘写失败（中途变只读等）时自动降级写运行时目录，
    保证设置操作不因存储抖动而直接丢失。

    Returns:
        True=落盘成功；False=两个位置均写入失败。
    """
    with _CONFIG_LOCK:
        path = config_path()
        try:
            _atomic_write_json(path, cfg)
            logger.info("配置已原子保存: %s", path)
            return True
        except OSError as exc:
            logger.error("配置写入 %s 失败: %s", path, exc)

        # 挂载盘抖动降级：写本地运行时目录兜底
        fallback = _persistence.RUNTIME_DIR / CONFIG_FILENAME
        try:
            _atomic_write_json(fallback, cfg)
            logger.warning("配置已降级保存至本地运行时目录: %s", fallback)
            return True
        except OSError as exc:
            logger.error("配置降级保存亦失败: %s", exc)
            return False


def get_config_path() -> Optional[Path]:
    """当前配置文件实际路径（供设置页展示诊断信息）；文件尚不存在时返回 None。"""
    path = config_path()
    return path if path.exists() else None


# ───────────────────────────────────────────────
# 密码哈希与 API Token (Task 2.1.2)
# ───────────────────────────────────────────────

import hmac
import secrets

# passlib bcrypt 上下文（懒加载：仅在首次用到密码功能时导入，加快纯手工模式冷启动）
_pwd_context = None


def _get_pwd_context():
    global _pwd_context
    if _pwd_context is None:
        from passlib.context import CryptContext  # 延迟导入
        _pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return _pwd_context


def hash_password(plain_password: str) -> str:
    """将明文密码哈希为 bcrypt 串（自动加盐，每次哈希结果不同属正常现象）。"""
    if not plain_password or not plain_password.strip():
        raise ValueError("密码不可为空")
    return _get_pwd_context().hash(plain_password)


def verify_password(plain_password: str, password_hash: Optional[str]) -> bool:
    """明文与已存 bcrypt 哈希比对；未设过密码（hash 为空）时恒返回 False。"""
    if not password_hash or not plain_password:
        return False
    try:
        return _get_pwd_context().verify(plain_password, password_hash)
    except (ValueError, TypeError):
        # 哈希串损坏/格式非法 → 一律拒绝，绝不抛出泄露内部状态
        return False


def set_password(plain_password: str) -> bool:
    """设置登录密码：bcrypt 哈希后写入 config.security.password_hash 并原子落盘。

    注意：当环境变量密码接管生效时（is_env_password_active() 为 True），
    本函数写入的哈希不会用于登录校验（env 优先），调用方（设置页守卫）应先拦截。"""
    cfg = load_config()
    cfg["security"]["password_hash"] = hash_password(plain_password)
    return save_config(cfg)


# ───────────────────────────────────────────────
# Web 登录密码环境变量接管 (Milestone 8)
# ───────────────────────────────────────────────

# 环境变量名：已设置且非空白串时，其明文优先于 config.json 的 password_hash，
# 成为唯一有效的 Web 登录密码（容器/HF Spaces 部署免首次锁死的引导通道）。
WEB_PASSWORD_ENV = "SYNAPSEMIND_WEB_PASSWORD"


def get_env_password() -> Optional[str]:
    """读取环境变量中的 Web 登录密码。

    空白串（""、纯空格等）一律视为未设置，防止空密码把控制台彻底锁死；
    未设置返回 None。"""
    raw = os.environ.get(WEB_PASSWORD_ENV)
    if raw is None or not raw.strip():
        return None
    return raw


def is_env_password_active() -> bool:
    """环境变量密码当前是否接管生效（供设置页改密守卫与诊断展示）。"""
    return get_env_password() is not None


def check_password(plain_password: str) -> bool:
    """校验登录密码（优先级：环境变量明文 > config.json bcrypt 哈希）。

    - 环境变量已设置：恒时比较明文（hmac.compare_digest 防时序侧信道），
      不再读取配置哈希——config 中旧密码在接管期间完全失效；
    - 环境变量未设置：回落原 bcrypt 哈希比对逻辑（行为零变化）。"""
    env_pwd = get_env_password()
    if env_pwd is not None:
        if not plain_password:
            return False
        return hmac.compare_digest(plain_password.encode("utf-8"),
                                   env_pwd.encode("utf-8"))
    return verify_password(plain_password, load_config()["security"].get("password_hash"))


def generate_api_token() -> str:
    """生成高熵随机 API Token（secrets 256 位随机源，URL 安全，形如 sm_xxxxxxxx）。"""
    return f"sm_{secrets.token_urlsafe(32)}"


def rotate_api_token() -> str:
    """轮换 API Token：生成新值写入 config.security.api_token 并原子落盘，返回明文新值。
    明文仅在轮换瞬间返回一次供用户保存，之后配置中仅存该明文用于接口恒时比对。"""
    token = generate_api_token()
    cfg = load_config()
    cfg["security"]["api_token"] = token
    save_config(cfg)
    return token


def verify_api_token(candidate: Optional[str]) -> bool:
    """恒时比较校验 Bearer Token（防时序侧信道）；未配置 Token 或候选为空一律拒绝。"""
    stored = load_config()["security"].get("api_token")
    if not candidate or not stored:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), stored.encode("utf-8"))


# ───────────────────────────────────────────────
# 运行模式读写 (Task 2.1.3)
# ───────────────────────────────────────────────

# 合法的三种人机协同运行模式（与 Readme 铁律 1 一一对应）
VALID_MODES = ("manual", "assisted", "augmented")


def get_mode() -> str:
    """读取当前运行模式；配置异常时兜底返回最安全的纯手工模式。"""
    mode = load_config().get("mode")
    return mode if mode in VALID_MODES else "manual"


def set_mode(mode: str) -> bool:
    """设置运行模式（manual/assisted/augmented 三选一，非法值拒绝并抛 ValueError）。"""
    if mode not in VALID_MODES:
        raise ValueError(f"非法运行模式: {mode!r}，仅允许 {VALID_MODES}")
    cfg = load_config()
    cfg["mode"] = mode
    return save_config(cfg)


# ───────────────────────────────────────────────
# LLM 配置读写 (Task 2.1.3)
# ───────────────────────────────────────────────

# llm 配置允许出现的字段白名单（防止越权字段混入配置文件）
_LLM_ALLOWED_KEYS = ("base_url", "api_key", "model")


def get_llm() -> dict:
    """
    读取 LLM 配置（OpenAI 兼容端点三元组）。

    红线：手工模式下允许 llm 字段为空并正常返回 —— 本函数不做任何
    必填校验或报错，空配置是否可用由调用方（Task 4.2.x 代谢器）按
    当前模式自行判断。
    """
    llm = load_config().get("llm", {})
    return {k: llm.get(k, "") for k in _LLM_ALLOWED_KEYS}


def set_llm(base_url: str, api_key: str, model: str) -> bool:
    """整体覆写 LLM 配置三元组；仅保留白名单字段，越权字段一律丢弃。"""
    cfg = load_config()
    cfg["llm"] = {
        "base_url": (base_url or "").strip().rstrip("/"),
        "api_key": (api_key or "").strip(),
        "model": (model or "").strip(),
    }
    return save_config(cfg)


# ───────────────────────────────────────────────
# S3 配置读写 (Task 2.1.3)
# ───────────────────────────────────────────────

# s3 配置字段白名单（与 s3_backup.S3_CFG_REQUIRED_KEYS 保持一致）
_S3_ALLOWED_KEYS = ("endpoint_url", "access_key_id", "secret_access_key", "bucket")


def get_s3() -> dict:
    """读取 S3 兼容对象存储连接配置；未配置时返回全空 dict（不报错）。"""
    s3 = load_config().get("s3", {})
    return {k: s3.get(k, "") for k in _S3_ALLOWED_KEYS}


def set_s3(endpoint_url: str, access_key_id: str, secret_access_key: str, bucket: str) -> bool:
    """整体覆写 S3 连接配置（R2/OSS/MinIO 均适用）；仅保留白名单字段。"""
    cfg = load_config()
    cfg["s3"] = {
        "endpoint_url": (endpoint_url or "").strip().rstrip("/"),
        "access_key_id": (access_key_id or "").strip(),
        "secret_access_key": (secret_access_key or "").strip(),
        "bucket": (bucket or "").strip(),
    }
    return save_config(cfg)


def is_s3_configured() -> bool:
    """S3 四要素是否已全部填写（供备份前快速预检，真正连通性由 boto3 实测）。"""
    return all(bool(v) for v in get_s3().values())


def get_backup_policy() -> dict:
    """读取定时备份策略（hourly/daily 开关与触发时刻），供 Task 1.3.4 注册器调用。"""
    return load_config().get("backup_policy", {})


def set_backup_policy(hourly_enabled: bool, daily_enabled: bool) -> bool:
    """写入定时备份开关（Task 6.1.2 设置页专用，原子落盘，其余键保持原值）。"""
    cfg = load_config()
    policy = cfg.setdefault("backup_policy", {})
    policy["hourly_enabled"] = bool(hourly_enabled)
    policy["daily_enabled"] = bool(daily_enabled)
    return save_config(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _persistence.init_persistence()
    demo = load_config()
    print(json.dumps(demo, ensure_ascii=False, indent=2))
    demo["mode"] = "assisted"
    print("save_config →", save_config(demo))
    print("reload mode →", load_config()["mode"])

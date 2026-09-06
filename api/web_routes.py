"""
SynapseMind - Web 控制台路由与会话拦截 (api/web_routes.py)

Task 6.1.1: 登录页与会话拦截。

路由：
- GET  /login   渲染登录页（web/templates/login.html，Jinja2）。
- POST /login   校验密码（config.check_password，bcrypt 哈希比对）
                → 签发 HMAC 签名会话 Cookie → 重定向 /。
- POST /logout  清除会话 Cookie → 重定向 /login。

中间件（install_login_guard）：
- 拦截未登录对管理页的访问 → 302 /login；
- /v1/*（OpenAI 兼容端点）走 Bearer 鉴权，不在 Cookie 守卫范围；
- /login 与静态资源放行。

签名 Cookie 方案（零额外依赖，纯标准库）：
- payload = base64url(json({"exp": 过期 unix 秒}))
- sig     = base64url(hmac_sha256(secret, payload)[:32])
- cookie  = f"{payload}.{sig}"，HttpOnly + SameSite=Lax；
- secret  = HMAC(api_token, "synapsemind.web.session") 域分离派生——
  api_token 轮换即全量会话失效（安全收益而非缺陷）；
- 校验恒时比较（hmac.compare_digest），过期/篡改/无密 一律视为未登录；
- 登录失败简易节流：连续 5 次失败锁 60 秒（进程内计数，防暴力猜解）。
"""

import base64
import hashlib
import hmac
import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

import config

logger = logging.getLogger("synapsemind.web_routes")

router = APIRouter(tags=["web"])

# ───────────────────────────────────────────────
# 会话签名常量与工具
# ───────────────────────────────────────────────

SESSION_COOKIE = "sm_session"
SESSION_TTL_SECONDS = 12 * 3600          # 会话有效期 12 小时
_SESSION_DOMAIN = b"synapsemind.web.session"

_LOGIN_MAX_FAILS = 5                     # 连续失败阈值
_LOGIN_LOCK_SECONDS = 60                 # 锁定时长
_fail_state = {"count": 0, "locked_until": 0.0}   # 进程内节流状态

_templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent.parent / "web" / "templates")
)


def _b64url(raw: bytes) -> str:
    """标准 base64url（无填充），URL/Cookie 安全。"""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _session_secret() -> bytes:
    """由 api_token 域分离派生会话签名密钥（api_token 轮换 → 全部会话失效）。"""
    key = (config.load_config()["security"].get("api_token") or "").encode("utf-8")
    return hmac.new(key, _SESSION_DOMAIN, hashlib.sha256).digest()


def sign_session(exp_epoch: Optional[int] = None) -> str:
    """签发 "payload.sig" 形态的会话令牌（exp 缺省为 now+TTL）。"""
    exp = int(exp_epoch if exp_epoch is not None else time.time() + SESSION_TTL_SECONDS)
    payload = _b64url(json.dumps({"exp": exp}, separators=(",", ":")).encode("utf-8"))
    sig = _b64url(hmac.new(_session_secret(), payload.encode("ascii"), hashlib.sha256).digest()[:32])
    return f"{payload}.{sig}"


def verify_session(token: Optional[str]) -> bool:
    """恒时校验会话令牌：签名匹配且未过期才返回 True；任何异常均视为未登录。"""
    if not token or "." not in token:
        return False
    payload, sig = token.rsplit(".", 1)
    expect = _b64url(hmac.new(_session_secret(), payload.encode("ascii"), hashlib.sha256).digest()[:32])
    if not hmac.compare_digest(sig.encode("ascii"), expect.encode("ascii")):
        return False
    try:
        pad = "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload + pad))
        return float(data.get("exp", 0)) > time.time()
    except (ValueError, TypeError):
        return False


def is_logged_in(request: Request) -> bool:
    """请求级登录态判定（供中间件与后续页面路由共用）。"""
    return verify_session(request.cookies.get(SESSION_COOKIE))


# ───────────────────────────────────────────────
# 登录 / 登出路由
# ───────────────────────────────────────────────

@router.get("/login")
def login_page(request: Request):
    """渲染登录页；已登录则直接进控制台。"""
    if is_logged_in(request):
        return RedirectResponse("/", status_code=302)
    return _templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
def login_submit(response: Response, request: Request, password: str = Form("")):
    """校验密码 → 签发签名 Cookie；失败回显错误（含节流锁）。"""
    now = time.time()
    if now < _fail_state["locked_until"]:
        remain = int(_fail_state["locked_until"] - now) + 1
        logger.warning("登录节流锁定中，剩余 %ds", remain)
        return _templates.TemplateResponse(
            request, "login.html",
            {"error": f"失败次数过多，请 {remain} 秒后再试"}, status_code=429,
        )

    if config.check_password(password):
        _fail_state["count"] = 0
        resp = RedirectResponse("/", status_code=302)
        resp.set_cookie(
            SESSION_COOKIE, sign_session(),
            max_age=SESSION_TTL_SECONDS, httponly=True, samesite="lax",
        )
        logger.info("Web 登录成功")
        return resp

    _fail_state["count"] += 1
    if _fail_state["count"] >= _LOGIN_MAX_FAILS:
        _fail_state["locked_until"] = now + _LOGIN_LOCK_SECONDS
        _fail_state["count"] = 0
        logger.warning("连续登录失败达阈值，锁定 %ds", _LOGIN_LOCK_SECONDS)
    else:
        logger.warning("Web 登录失败（第 %d 次）", _fail_state["count"])
    return _templates.TemplateResponse(
        request, "login.html", {"error": "密码错误，请重试"}, status_code=401,
    )


@router.post("/logout")
def logout():
    """清除会话 Cookie 并回登录页。"""
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# ───────────────────────────────────────────────
# 会话拦截中间件
# ───────────────────────────────────────────────

# 无需登录即可访问的前缀：登录页自身 + OpenAI 兼容 API（Bearer 鉴权）+ 健康检查
_PUBLIC_PREFIXES = ("/login", "/v1/", "/health")


def install_login_guard(app) -> None:
    """
    注册 HTTP 中间件：未登录访问管理页一律 302 /login。

    放行范围：_PUBLIC_PREFIXES 前缀命中，或路径恰为 "/v1"。
    注意：静态资源在后续任务挂载于 /static 下，届时需登录才能取——
    管理页 CSS/JS 属受保护资产，符合「中间件拦截未登录访问管理页」语义。
    """

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        path = request.url.path
        if path in _PUBLIC_PREFIXES or path.startswith(_PUBLIC_PREFIXES):
            return await call_next(request)
        if not is_logged_in(request):
            return RedirectResponse("/login", status_code=302)
        return await call_next(request)


# ───────────────────────────────────────────────
# 设置页与设置 API (Task 6.1.2)
# ───────────────────────────────────────────────

# 快照键 → 显示名的截取规则：去掉前缀保留 文件名（含 brain/UTC 时间戳）
def _list_snapshots(kind: str) -> list:
    """列取某类别（manual/hourly/daily）云端快照键，倒序（最新在前）。"""
    from storage import s3_backup

    s3_cfg = config.get_s3()
    if not config.is_s3_configured():
        return []
    client = s3_backup._build_client(s3_cfg)
    prefix = s3_backup._prefix_for("default", kind)
    resp = client.list_objects_v2(Bucket=s3_cfg["bucket"], Prefix=prefix)
    keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".tar.gz")]
    return sorted(keys, reverse=True)


@router.get("/settings")
def settings_page(request: Request):
    """渲染全能设置页（中间件已保证登录态）。"""
    return _templates.TemplateResponse(request, "settings.html", {})


@router.get("/settings/api")
def settings_api():
    """聚合返回当前全部配置（敏感键只回是否已配置，不回明文）。"""
    llm = config.get_llm()
    s3 = config.get_s3()
    return {
        "mode": config.get_mode(),
        "llm": {
            "base_url": llm.get("base_url", ""),
            "model": llm.get("model", ""),
            "has_key": bool(llm.get("api_key")),
        },
        "s3": {
            "endpoint_url": s3.get("endpoint_url", ""),
            "bucket": s3.get("bucket", ""),
            "access_key_id": s3.get("access_key_id", ""),
            "has_secret": bool(s3.get("secret_access_key")),
        },
        "backup_policy": config.get_backup_policy(),
    }


@router.post("/settings/api/mode")
def api_set_mode(payload: dict):
    try:
        config.set_mode((payload.get("mode") or "").strip())
        return {"ok": True}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/settings/api/password")
def api_set_password(payload: dict):
    pwd = payload.get("password") or ""
    if len(pwd) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    config.set_password(pwd)
    logger.info("登录密码已通过设置页更新")
    return {"ok": True}


@router.post("/settings/api/token")
def api_rotate_token():
    """轮换 API Token；明文仅在本次响应返回一次。"""
    return {"token": config.rotate_api_token()}


@router.post("/settings/api/llm")
def api_set_llm(payload: dict):
    base_url = (payload.get("base_url") or "").strip()
    api_key = (payload.get("api_key") or "").strip()
    model = (payload.get("model") or "").strip()
    if base_url or api_key or model:
        config.set_llm(base_url, api_key, model)
    return {"ok": True}


@router.post("/settings/api/s3")
def api_set_s3(payload: dict):
    config.set_s3(
        (payload.get("endpoint_url") or "").strip(),
        (payload.get("access_key_id") or "").strip(),
        (payload.get("secret_access_key") or "").strip(),
        (payload.get("bucket") or "").strip(),
    )
    return {"ok": True}


@router.post("/settings/api/backup_policy")
def api_set_backup_policy(payload: dict):
    config.set_backup_policy(
        hourly_enabled=bool(payload.get("hourly_enabled")),
        daily_enabled=bool(payload.get("daily_enabled")),
    )
    return {"ok": True}


@router.get("/settings/api/snapshots")
def api_snapshots():
    """三类快照状态（各 ≤3 份）；S3 未配置时全部为空列表。"""
    out = {"manual": [], "hourly": [], "daily": []}
    if config.is_s3_configured():
        for kind in out:
            try:
                out[kind] = _list_snapshots(kind)
            except Exception as exc:   # 云端不可达不让页面崩，置空并记日志
                logger.warning("列举 %s 快照失败: %s", kind, exc)
    return out


@router.post("/settings/api/backup")
def api_manual_backup():
    """立即手动备份（打包→上传→轮转，函数内部保证各 ≤3 份）。"""
    from storage import s3_backup

    s3_cfg = config.get_s3()
    if not config.is_s3_configured():
        raise HTTPException(status_code=400, detail="S3 尚未配置")
    ok = s3_backup.manual_backup("default", s3_cfg)
    if not ok:
        raise HTTPException(status_code=500, detail="备份失败，详见服务端日志")
    return {"ok": True}


@router.post("/settings/api/restore")
def api_restore(payload: dict):
    """从云端指定类别最新快照恢复（覆盖 /data 镜像并触发自愈拉取）。"""
    from storage import s3_backup

    kind = (payload.get("kind") or "").strip()
    s3_cfg = config.get_s3()
    if not config.is_s3_configured():
        raise HTTPException(status_code=400, detail="S3 尚未配置")
    try:
        ok = s3_backup.restore_from_cloud("default", kind, s3_cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=500, detail="恢复失败或云端无归档，详见服务端日志")
    return {"ok": True}


# ───────────────────────────────────────────────
# 控制台面板 API (Task 6.2.1: 纯手工建点建边)
# ───────────────────────────────────────────────
# 约定：所有端点默认操作 "default" 脑（单脑部署阶段）；
# 手工面板录入的突触/规则断言均视为人工即时确认（status=CONFIRMED），
# 与 AI 提案链路（PENDING 待收纳）严格区分。

_DEFAULT_BRAIN = "default"


@router.get("/")
def console_page(request: Request):
    """渲染控制台主页（建点建边面板 + 待收纳列表宿主页）。"""
    return _templates.TemplateResponse(request, "index.html", {})


@router.get("/api/concepts")
def api_list_concepts(keyword: str = "", limit: int = 500):
    """概念列表（供表单 id/name 选择器与弱提示候选检索）。"""
    from core.brain_manager import get_or_create
    from core.graph_cortex import list_concepts

    session = get_or_create(_DEFAULT_BRAIN)
    items = list_concepts(session.get_kuzu(), keyword=keyword.strip(),
                          limit=max(1, min(int(limit), 2000)))
    return {"items": items, "total": len(items)}


@router.post("/api/concepts")
def api_create_concept(payload: dict):
    """创建概念节点。

    返回 created=true 表示已入库；created=false 表示命中相似度弱提示，
    携带 candidates 供前端弹「相近词确认」弹窗（红线：绝不自动合并）。
    强制落库须显式传 force=true 二次提交。
    """
    from core.brain_manager import get_or_create
    from core.graph_cortex import add_concept
    from core.similarity import check_concept_similarity
    from api.models import ConceptIn
    from storage import persistence

    try:
        body = ConceptIn(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    force = bool(payload.get("force"))
    if not force:
        sim = check_concept_similarity(_DEFAULT_BRAIN, body.name)
        if sim["candidates"]:
            return {"created": False, "similarity": sim}

    session = get_or_create(_DEFAULT_BRAIN)
    cid = add_concept(session.get_kuzu(), body.name, body.category, body.description)
    persistence.sync_to_storage()
    return {"created": True, "id": cid}


@router.post("/api/synapses")
def api_create_synapse(payload: dict):
    """手工连线录入即确认（status=CONFIRMED，人工面板场景铁律）。"""
    from core.brain_manager import get_or_create
    from core.graph_cortex import add_synapse
    from api.models import SynapseIn
    from storage import persistence

    try:
        body = SynapseIn(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    session = get_or_create(_DEFAULT_BRAIN)
    try:
        rid = add_synapse(session.get_kuzu(), body.src_id, body.dst_id,
                          body.relation, body.weight, body.evidence,
                          status="CONFIRMED")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    persistence.sync_to_storage()
    return {"created": True, "rel_id": rid}


@router.post("/api/rules")
def api_create_rule(payload: dict):
    """规则断言录入：'A + B 导致 C' 物化为每条前提→结论的 CONFIRMED 突触。

    断言本身追加进结论方向的 evidence（可回溯），全部前提概念必须已存在，
    任一缺失即 400 明确报错，绝不静默造点（与 models.py 约定对齐）。
    """
    from core.brain_manager import get_or_create
    from core.graph_cortex import add_synapse, list_concepts
    from api.models import RuleAssertion
    from storage import persistence

    try:
        body = RuleAssertion(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    session = get_or_create(_DEFAULT_BRAIN)
    conn = session.get_kuzu()
    by_name = {c["name"]: c["id"] for c in list_concepts(conn, limit=2000)}

    missing = [n for n in (*body.conditions, body.conclusion) if n not in by_name]
    if missing:
        raise HTTPException(status_code=400,
                            detail=f"概念不存在，请先建点: {', '.join(missing)}")

    conclusion_id = by_name[body.conclusion]
    rule_evidence = f"[规则断言] {' + '.join(body.conditions)} ⇒ {body.conclusion}"
    if body.evidence:
        rule_evidence += f"；出处: {body.evidence}"

    rel_ids = []
    for cond_name in body.conditions:
        rid = add_synapse(conn, by_name[cond_name], conclusion_id,
                          "causes", 0.8, rule_evidence, status="CONFIRMED")
        rel_ids.append(rid)
    persistence.sync_to_storage()
    return {"created": True, "rel_ids": rel_ids,
            "fanin": len(rel_ids)}


# ───────────────────────────────────────────────
# 待收纳与人机共审 (Task 6.2.2: index.html 下半区)
# ───────────────────────────────────────────────
# AI 提案链路产生的 PENDING 突触在此由人工批量处置：
# 确认（→CONFIRMED 参与推演）/ 拒绝（→REJECTED 留审计痕迹）/
# 编辑（权重滑块等）/ 删除。状态流转全部经 core 状态机闸口。


@router.get("/api/synapses")
def api_list_synapses(status: str = "PENDING", limit: int = 500):
    """突触列表（默认取待收纳 PENDING；status 空串=全部）。"""
    from core.brain_manager import get_or_create
    from core.graph_cortex import list_synapses

    try:
        items = list_synapses(get_or_create(_DEFAULT_BRAIN).get_kuzu(),
                              status=status.strip(), limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"items": items, "total": len(items)}


@router.post("/api/synapses/batch")
def api_batch_synapses(payload: dict):
    """批量处置：{rel_ids: [...], action: "confirm"|"reject"}。

    逐条走状态机闸口；单条非法不影响其余（结果逐条回传，主权在人工）。
    """
    from core.brain_manager import get_or_create
    from core.graph_cortex import set_synapse_status
    from storage import persistence

    rel_ids = payload.get("rel_ids") or []
    action = payload.get("action")
    if action not in ("confirm", "reject"):
        raise HTTPException(status_code=400, detail=f"非法 action: {action!r}")
    if not isinstance(rel_ids, list) or not rel_ids:
        raise HTTPException(status_code=400, detail="rel_ids 不得为空")

    target = "CONFIRMED" if action == "confirm" else "REJECTED"
    conn = get_or_create(_DEFAULT_BRAIN).get_kuzu()
    results = {}
    for rid in rel_ids:
        try:
            results[rid] = set_synapse_status(conn, str(rid), target)
        except ValueError as exc:
            results[rid] = str(exc)          # 非法流转逐条记录，不中断批量
    persistence.sync_to_storage()
    return {"action": action, "results": results}


@router.post("/api/synapses/{rel_id}")
def api_update_synapse(rel_id: str, payload: dict):
    """单条编辑（权重滑块/关系/证据）；不含状态流转（走 batch 或状态闸口）。"""
    from core.brain_manager import get_or_create
    from core.graph_cortex import update_synapse
    from storage import persistence

    allowed = {k: payload[k] for k in ("weight", "relation", "evidence")
               if k in payload}
    if not allowed:
        raise HTTPException(status_code=400, detail="无可编辑字段")
    try:
        ok = update_synapse(get_or_create(_DEFAULT_BRAIN).get_kuzu(),
                            rel_id, **allowed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=404, detail=f"突触不存在: {rel_id}")
    persistence.sync_to_storage()
    return {"updated": True, "rel_id": rel_id}


@router.delete("/api/synapses/{rel_id}")
def api_delete_synapse(rel_id: str):
    """删除突触（误录清理路径；REJECTED 亦删除而非改写终态）。"""
    from core.brain_manager import get_or_create
    from core.graph_cortex import delete_synapse
    from storage import persistence

    if not delete_synapse(get_or_create(_DEFAULT_BRAIN).get_kuzu(), rel_id):
        raise HTTPException(status_code=404, detail=f"突触不存在: {rel_id}")
    persistence.sync_to_storage()
    return {"deleted": True, "rel_id": rel_id}


# ───────────────────────────────────────────────
# 纯图推演看板 API (Task 6.3.2: 神经脉冲动画与证据链打印)
# ───────────────────────────────────────────────


@router.post("/api/reason")
def api_reason(payload: dict):
    """纯图推演（模式 1 闭环）：线索 → 能量扩散 → 结论与真实证据链。

    返回 core.inference.pure_reason 完整结果：
    - activations: 按峰值能量降序的激活节点序列，每项含 hop（扩散时序）、
      chain（种子→…→该节点 的真实突触链：rel_id/relation/evidence），
      供前端按跳序播放脉冲动画；
    - answer:      结论概念 + 多前提断言命中（conditions/evidence）+ 证据链；
    - status=insufficient 时 message 为固定话术，前端原样展示，绝不编造。
    """
    from core.inference import pure_reason

    clue = str((payload or {}).get("clue", "")).strip()
    if not clue:
        raise HTTPException(status_code=422, detail="clue 不得为空")
    if len(clue) > 256:
        raise HTTPException(status_code=422, detail="clue 超长（>256 字）")

    try:
        result = pure_reason(_DEFAULT_BRAIN, clue)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


# ───────────────────────────────────────────────
# 冒烟自检（python3 api/web_routes.py，零网络零真实密码）
# ───────────────────────────────────────────────

if __name__ == "__main__":
    import shutil
    import sys
    import tempfile

    logging.basicConfig(level=logging.WARNING)
    _self = sys.modules[__name__]

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # 打补丁：固定密码与签名密钥，避免依赖真实 config.json
    _orig_check = config.check_password
    _orig_load = config.load_config
    config.check_password = lambda pwd: pwd == "right_pwd"
    config.load_config = lambda: {"security": {"api_token": "sm_smoke_key"}}

    # Task 6.1.2 自检补丁：config 读写函数全部打桩，写入值捕获进 _captured 供断言
    _cfg_names = ("get_mode", "set_mode", "get_llm", "set_llm", "get_s3", "set_s3",
                  "get_backup_policy", "set_backup_policy", "is_s3_configured",
                  "set_password", "rotate_api_token")
    _orig_cfg = {n: getattr(config, n) for n in _cfg_names}
    _captured: dict = {}
    config.get_mode = lambda: "manual"

    def _fake_set_mode(m):
        if m not in ("manual", "assisted", "augmented"):
            raise ValueError(f"非法运行模式: {m!r}")
        _captured["mode"] = m

    config.set_mode = _fake_set_mode
    config.get_llm = lambda: {"base_url": "", "api_key": "", "model": ""}
    config.set_llm = lambda b, k, m: _captured.update(llm=(b, k, m))
    config.get_s3 = lambda: {"endpoint_url": "", "access_key_id": "",
                             "secret_access_key": "", "bucket": ""}
    config.set_s3 = lambda e, a, s, b: _captured.update(s3=(e, a, s, b))
    config.get_backup_policy = lambda: {"hourly_enabled": True, "daily_enabled": False}
    config.set_backup_policy = lambda hourly_enabled, daily_enabled: _captured.update(policy=(hourly_enabled, daily_enabled))
    config.is_s3_configured = lambda: False
    config.set_password = lambda p: True
    config.rotate_api_token = lambda: "sm_rotated_token"
    _self._list_snapshots = lambda kind: [f"backups/default/{kind}/snap_{kind}.tar.gz"]

    # s3_backup 备份/恢复打桩（零网络）
    from storage import s3_backup as _s3b
    _orig_backup_fn = _s3b.manual_backup
    _orig_restore_fn = _s3b.restore_from_cloud
    _s3b.manual_backup = lambda *a, **k: True

    def _fake_restore(*a, **k):
        # 合法 kind → 返回 False（模拟恢复失败走 500）；非法 kind → ValueError 走 400
        kind = a[1] if len(a) >= 2 else k.get("kind")
        if kind not in ("manual", "hourly", "daily"):
            raise ValueError("非法备份类别，仅允许 ('manual', 'hourly', 'daily')")
        return False

    _s3b.restore_from_cloud = _fake_restore  # 默认桩：合法 kind 走 500、非法 kind 走 400

    # Task 6.2.1 面板端点自检桩：临时 Kùzu 库替身 + 刷盘空转（零 /data 触碰）
    import kuzu as _kuzu_mod
    from core import brain_manager as _bm
    from core import graph_cortex as _gc
    from storage import persistence as _persistence_mod

    _kuzu_tmp = tempfile.mkdtemp(prefix="smoke_kuzu_")
    _kuzu_db = _kuzu_mod.Database(_kuzu_tmp + "/graph")   # 路径不得预先存在，由引擎自建
    _kuzu_conn = _kuzu_mod.Connection(_kuzu_db)

    class _FakeSession:
        def get_kuzu(self):
            return _kuzu_conn

    _gc.init_schema(_kuzu_conn)                      # 面板端点首次使用前建表
    _orig_get_or_create = _bm.get_or_create
    _orig_sync = _persistence_mod.sync_to_storage
    _bm.get_or_create = lambda brain_id: _FakeSession()   # 延迟导入均经模块属性取值
    _persistence_mod.sync_to_storage = lambda: {"synced": False}

    tmp = tempfile.mkdtemp(prefix="smoke_login_")
    try:
        app = FastAPI()
        app.include_router(router)            # GET / 即真实 console_page（受登录守卫保护）
        install_login_guard(app)
        client = TestClient(app, follow_redirects=False)

        # 1) 未登录访问管理页 → 302 /login（中间件拦截）
        r = client.get("/")
        assert r.status_code == 302 and r.headers["location"] == "/login", (r.status_code, r.headers)
        print("未登录访问管理页 → 302 /login ✓")

        # 2) 登录页可匿名访问
        r = client.get("/login")
        assert r.status_code == 200 and "Synapse" in r.text, r.status_code
        print("登录页渲染 ✓")

        # 3) 错误密码 → 401 错误回显
        r = client.post("/login", data={"password": "wrong"})
        assert r.status_code == 401 and "密码错误" in r.text, (r.status_code, r.text[:120])
        print("错误密码 401 ✓")

        # 4) 正确密码 → 302 / 且下发 HttpOnly 会话 Cookie
        r = client.post("/login", data={"password": "right_pwd"})
        assert r.status_code == 302 and r.headers["location"] == "/", r.status_code
        set_cookie = r.headers.get("set-cookie", "")
        assert SESSION_COOKIE in set_cookie and "HttpOnly" in set_cookie, set_cookie
        print("正确密码 → 302 + HttpOnly Cookie ✓")

        # 5) 携带合法 Cookie 访问管理页 → 200（真实 index.html 面板）
        cookie = set_cookie.split(";", 1)[0]
        r = client.get("/", headers={"Cookie": cookie})
        assert r.status_code == 200 and "Synapse" in r.text and "建立概念节点" in r.text, r.status_code
        print("携带合法 Cookie 访问管理页 ✓")

        # 6) 篡改签名的 Cookie → 再次 302
        bad = cookie[:-2] + ("xx" if not cookie.endswith("xx") else "yy")
        r = client.get("/", headers={"Cookie": bad})
        assert r.status_code == 302, r.status_code
        print("篡改签名 Cookie → 302 ✓")

        # 7) 过期令牌 → 未登录
        expired = sign_session(exp_epoch=int(time.time()) - 10)
        assert not verify_session(expired)
        print("过期令牌校验 ✓")

        # 8) /v1/models 不受 Cookie 守卫影响（Bearer 体系自管）
        r = client.get("/v1/models")
        assert r.status_code != 302, r.status_code      # 非 302 即未走登录重定向
        print("/v1/* 不受 Cookie 守卫拦截 ✓")

        # ── Task 6.1.2 设置端点自检 ──
        auth = {"headers": {"Cookie": cookie}}

        # 9) 设置页渲染（需登录态）
        r = client.get("/settings", **auth)
        assert r.status_code == 200 and "Synapse" in r.text, r.status_code
        print("设置页渲染 ✓")

        # 10) 未登录访问设置 API → 中间件 302 拦截（全新 client，无 Cookie）
        anon = TestClient(app, follow_redirects=False)
        r = anon.get("/settings/api")
        assert r.status_code == 302 and r.headers["location"].startswith("/login"), r.status_code
        print("未登录访问设置 API → 302 ✓")

        # 11) GET /settings/api 聚合配置（敏感键只回 has_key/has_secret）
        r = client.get("/settings/api", **auth)
        body = r.json()
        assert body["mode"] == "manual", body
        assert "api_key" not in body["llm"] and body["llm"]["has_key"] is False, body
        assert "secret_access_key" not in body["s3"] and body["s3"]["has_secret"] is False, body
        print("设置聚合 API ✓")

        # 12) 模式切换：合法值写入 + 非法值 400
        r = client.post("/settings/api/mode", json={"mode": "augmented"}, **auth)
        assert r.status_code == 200 and _captured["mode"] == "augmented", (r.status_code, _captured)
        r = client.post("/settings/api/mode", json={"mode": "bogus"}, **auth)
        assert r.status_code == 400, r.status_code
        print("模式切换 ✓（非法值 400）")

        # 13) LLM / S3 写入捕获
        client.post("/settings/api/llm", json={"base_url": "http://llm.local/v1",
                                               "api_key": "sk-x", "model": "gpt"}, **auth)
        assert _captured["llm"] == ("http://llm.local/v1", "sk-x", "gpt"), _captured
        client.post("/settings/api/s3", json={"endpoint_url": "http://s3.local",
                                              "access_key_id": "ak", "secret_access_key": "sk",
                                              "bucket": "bkt"}, **auth)
        assert _captured["s3"] == ("http://s3.local", "ak", "sk", "bkt"), _captured
        print("LLM / S3 配置写入 ✓")

        # 14) 备份策略开关
        client.post("/settings/api/backup_policy", json={"hourly_enabled": False,
                                                         "daily_enabled": True}, **auth)
        assert _captured["policy"] == (False, True), _captured
        print("备份策略开关 ✓")

        # 15) 密码修改：过短 400，正常 200
        r = client.post("/settings/api/password", json={"password": "123"}, **auth)
        assert r.status_code == 400, r.status_code
        r = client.post("/settings/api/password", json={"password": "abcdef"}, **auth)
        assert r.status_code == 200, r.status_code
        print("密码修改校验 ✓")

        # 16) Token 轮换：仅本次响应回明文
        r = client.post("/settings/api/token", **auth)
        assert r.json() == {"token": "sm_rotated_token"}, r.json()
        print("Token 轮换 ✓")

        # 17) 快照状态：S3 未配置 → 三类全空；恢复 S3 配置后 → 列表非空
        r = client.get("/settings/api/snapshots", **auth)
        assert r.json() == {"manual": [], "hourly": [], "daily": []}, r.json()
        config.is_s3_configured = lambda: True
        r = client.get("/settings/api/snapshots", **auth)
        snaps = r.json()
        assert all(len(snaps[k]) == 1 for k in snaps), snaps
        assert snaps["manual"][0].startswith("backups/default/manual/"), snaps
        print("快照状态列举 ✓")

        # 18) 手动备份触发 + 恢复：默认桩下 失败→500、非法 kind→400；成功桩下 200
        r = client.post("/settings/api/backup", **auth)
        assert r.status_code == 200, (r.status_code, r.text)
        r = client.post("/settings/api/restore", json={"kind": "hourly"}, **auth)
        assert r.status_code == 500, r.status_code        # 默认桩返回 False → 500
        r = client.post("/settings/api/restore", json={"kind": "weekly"}, **auth)
        assert r.status_code == 400, r.status_code        # 非法 kind → ValueError → 400
        _s3b.restore_from_cloud = lambda *a, **k: True    # 切换成功桩
        r = client.post("/settings/api/restore", json={"kind": "hourly"}, **auth)
        assert r.status_code == 200, r.status_code
        print("手动备份 / 云端恢复 ✓（非法 kind 400）")

        # ── Task 6.2.1 面板端点自检（临时 Kùzu 已建表）──
        # 19) 未登录访问面板 API → 302 拦截
        r = anon.get("/api/concepts")
        assert r.status_code == 302, r.status_code
        print("未登录访问面板 API → 302 ✓")

        # 20) 建点：无相近概念 → 直接入库
        r = client.post("/api/concepts", json={"name": "恶寒", "category": "症状",
                                               "description": "怕冷"}, **auth)
        assert r.status_code == 200 and r.json()["created"] is True, (r.status_code, r.text)
        cid_a = r.json()["id"]
        print("建点（无相似直接入库）✓")

        # 21) 相似度弱提示：录入「恶寒发热」→ created=false + candidates；红线：无副作用
        r = client.post("/api/concepts", json={"name": "恶寒发热", "category": "症状"}, **auth)
        body = r.json()
        assert body["created"] is False and body["similarity"]["candidates"], body
        assert body["similarity"]["candidates"][0]["name"] == "恶寒", body
        r = client.get("/api/concepts", **auth)
        assert r.json()["total"] == 1, r.json()           # 弱提示未产生任何写入
        print("相似度弱提示 ✓（仅提示无副作用）")

        # 22) force=true 强制落库 + 列表
        r = client.post("/api/concepts", json={"name": "恶寒发热", "category": "症状", "force": True}, **auth)
        assert r.json()["created"] is True, r.json()
        cid_b = r.json()["id"]
        r = client.post("/api/concepts", json={"name": "发热", "category": "症状", "force": True}, **auth)
        cid_c = r.json()["id"]
        r = client.get("/api/concepts", **auth)
        assert r.json()["total"] == 3, r.json()
        print("force 落库 + 概念列表 ✓")

        # 23) 连线：手工录入即 CONFIRMED；端点不存在 → 400；非法参数 → 422
        r = client.post("/api/synapses", json={"src_id": cid_b, "dst_id": cid_c,
                                               "relation": "relates", "weight": 0.9}, **auth)
        assert r.status_code == 200 and r.json()["created"], (r.status_code, r.text)
        r = client.post("/api/synapses", json={"src_id": "ghost", "dst_id": cid_c,
                                               "relation": "relates"}, **auth)
        assert r.status_code == 400, r.status_code
        r = client.post("/api/synapses", json={"src_id": cid_a, "dst_id": cid_c,
                                               "relation": "relates", "weight": 9.9}, **auth)
        assert r.status_code == 422, r.status_code        # weight 越界 → Pydantic 校验
        print("手工连线（CONFIRMED / 缺点 400 / 越界 422）✓")

        # 24) 规则断言：物化为 前提→结论 CONFIRMED 突触；缺概念 → 400
        r = client.post("/api/rules", json={"conditions": ["恶寒", "发热"],
                                            "conclusion": "太阳伤寒", "evidence": "伤寒论"}, **auth)
        body = r.json()
        assert r.status_code == 400, r.status_code        # 结论概念未建 → 明确报错
        r = client.post("/api/concepts", json={"name": "太阳伤寒", "category": "证型", "force": True}, **auth)
        cid_d = r.json()["id"]
        r = client.post("/api/rules", json={"conditions": ["恶寒", "发热"],
                                            "conclusion": "太阳伤寒", "evidence": "伤寒论"}, **auth)
        body = r.json()
        assert r.status_code == 200 and body["created"] and body["fanin"] == 2, (r.status_code, body)
        print("规则断言物化 ✓（缺概念 400 / 双前提 fanin=2）")

        print("web_routes 冒烟自检全部通过 ✓")
    finally:
        config.check_password = _orig_check
        config.load_config = _orig_load
        for _n, _f in _orig_cfg.items():
            setattr(config, _n, _f)
        _s3b.manual_backup = _orig_backup_fn
        _s3b.restore_from_cloud = _orig_restore_fn
        _bm.get_or_create = _orig_get_or_create
        _persistence_mod.sync_to_storage = _orig_sync
        del _self._list_snapshots
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(_kuzu_tmp, ignore_errors=True)

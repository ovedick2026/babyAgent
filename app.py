"""
SynapseMind - 单进程入口组装 (app.py)

Task 7.1: 集成启动入口。

装配内容：
- FastAPI 应用：include web_routes（Cookie 会话控制台）+ openai_routes（/v1 Bearer API）；
- /static 静态资源挂载（graph_view.js 等管理页资产，受登录守卫保护）；
- /health 轻量健康检查（登录守卫白名单前缀之一）；
- Jinja2 模板由 web_routes 内部 Jinja2Templates 承载（web/templates/），此处无需重复配置。

生命周期（lifespan）：
- 启动：init_persistence（探测 /data 可写性并建目录）→ restore_on_boot（/data 镜像
  自愈拉取回 /tmp 工作副本）→ 注册 APScheduler 定时备份（hourly/daily 开关经
  callable 实时读取 config，支持运行时改配置后下次触发即生效）。
- 关闭：先停调度器（防止备份与新写入竞争），再逐脑 close_brain（内部强制
  sync_to_storage 刷盘 /data），保证停机零数据丢失。

铁律：单 Worker Uvicorn。SQLite 与 Kùzu 均按单写者模型设计，
多进程并发写必然损坏数据，严禁调大 workers。
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import config
from api import openai_routes, web_routes
from core import brain_manager
from storage import persistence, s3_backup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("synapsemind.app")

WEB_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭钩子：自愈拉取 → 定时备份注册；退出时停表并安全落盘。"""
    # 1) 持久化初始化 + 启动自愈（/data 镜像 → /tmp 工作副本；无镜像则初始化空库）
    persistence.init_persistence()
    persistence.restore_on_boot()

    # 2) APScheduler 定时备份：hourly 整点 / daily 每日 03:00（可被 policy.daily_hour 覆盖），
    #    受 backup_policy 开关控制；s3_cfg 与 backup_policy 传 callable，触发时刻实时求值。
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler(timezone="UTC")
    registered = s3_backup.register_backup_jobs(
        scheduler,
        list_brain_ids=brain_manager.list_brains,
        s3_cfg=config.get_s3,
        backup_policy=config.get_backup_policy,
    )
    scheduler.start()
    logger.info("调度器已启动，已注册备份任务: %s", registered or "无（开关均关闭）")

    try:
        yield
    finally:
        # 3) 优雅停机：先停调度器防止备份与新写入竞争，再逐脑关闭（内含强制刷盘 /data）
        scheduler.shutdown(wait=False)
        for brain_id in brain_manager.list_brains():
            try:
                brain_manager.close_brain(brain_id)
            except Exception as exc:  # noqa: BLE001 停机清尾不让单脑故障中断全局退出
                logger.warning("关闭 brain=%s 异常: %s", brain_id, exc)
        logger.info("SynapseMind 已优雅停机")


app = FastAPI(title="SynapseMind", lifespan=lifespan)


@app.get("/health", include_in_schema=False)
def health():
    """轻量健康检查（web_routes 登录守卫白名单前缀之一，匿名可探活）。"""
    return {"status": "ok"}


# ── 路由装配 ──
# Web 控制台（Cookie 会话）在前，OpenAI 兼容 API（自带 /v1 前缀与 Bearer 鉴权）在后。
app.include_router(web_routes.router)
app.include_router(openai_routes.router)

# 登录守卫中间件：/login、/v1/*、/health 放行，其余未登录一律 302 /login。
# /static 同样被守卫拦截——管理页 CSS/JS 属受保护资产（见 web_routes 设计注释）。
web_routes.install_login_guard(app)

# 静态资源（graph_view.js 等前端资产）。mount 为路由级分发，中间件照样先拦截，
# 故未登录拿不到任何静态文件，与「中间件拦截未登录访问管理页」语义一致。
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("SM_HOST", "0.0.0.0")
    port = int(os.environ.get("SM_PORT", "8000"))
    # 铁律：单 Worker。SQLite 与 Kùzu 均按单写者设计，多进程并发写必损坏。
    uvicorn.run(app, host=host, port=port, workers=1, log_level="info")

"""
SynapseMind - OpenAI 兼容 API (api/openai_routes.py)

Task 5.1:   GET  /v1/models            模型发现（brain → model id）。
Task 5.2.1: require_bearer            Bearer 凭证校验依赖项（恒时比较）。
Task 5.2.2: POST /v1/chat/completions 推演执行与无路径兜底。
Task 5.2.3: stream=true / false       双形态响应（SSE 增量 + 普通 JSON）。

chat/completions 链路定位：
    解析 model(=brain_id) 与 messages 末条 user 线索
      → 按运行模式路由：
          manual / assisted → core.inference.pure_reason（纯图能量扩散，零 LLM）
          augmented         → 同上；纯图无路径且 LLM 可用时叠加
                              core.reasoner.plan_for_brain 产出辨证计划
      → 无有效激活路径 / LLM 不可用 / 推演异常
        → 固定话术兜底：「当前脑尚未学习到足以回答该问题的知识」
      → stream=true 时以 OpenAI chat.completion.chunk 的 SSE 增量下发
        （首帧 role → 证据链逐段 content 帧 → stop 终帧 → [DONE]）；
        stream=false 时返回普通 JSON。两种形态正文完全一致，
        均内嵌真实已确认证据链（标注来源概念 id），绝不编造

核心铁律：
- 只读发现：/v1/models 绝不创建/物化任何脑；
- 推演只消费 CONFIRMED 子图（inference/reasoner 数据源头保证），
  PENDING / REJECTED 知识对回答端点天然不可见；
- 无路径兜底话术固定，严禁在无证据时给出模棱两可的猜测；
- 流式与非流式正文逐字一致（自检强制校验），杜绝两套文本漂移；
- 响应结构严格对齐 OpenAI（含 error 对象形态），鉴权失败统一 401。
"""

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

import config
from config import verify_api_token
from core import brain_manager, inference, reasoner

logger = logging.getLogger("synapsemind.openai_routes")

router = APIRouter(prefix="/v1", tags=["openai"])

MODEL_OWNER = "synapsemind"

# Task 5.2.2: 无有效激活路径时的固定兜底话术（不得替换为含糊表述）
NO_PATH_MESSAGE = "当前脑尚未学习到足以回答该问题的知识"

# auto_error=False：缺失 Authorization 时由依赖自行返回 401，错误语义统一可控
_bearer_scheme = HTTPBearer(auto_error=False)


async def require_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> str:
    """
    Task 5.2.1: Bearer 凭证校验依赖项。

    从 Authorization: Bearer <token> 提取凭证，交由 config.verify_api_token
    与 security.api_token 恒时比较（hmac.compare_digest，防时序侧信道）。
    缺失 / 无效 / 服务端未配置 Token 一律 401，并携带标准
    WWW-Authenticate: Bearer 响应头。

    Returns:
        校验通过的明文 token（供后续路由做请求级日志/审计）。
    """
    candidate = credentials.credentials if credentials is not None else None
    if not verify_api_token(candidate):
        logger.warning("API 鉴权失败: %s", "缺失凭证" if candidate is None else "无效凭证")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return candidate


# ───────────────────────────────────────────────
# 请求模型（Task 5.2.2）
# ───────────────────────────────────────────────

class _MessageIn(BaseModel):
    """OpenAI 兼容消息项：仅消费 role=user 的内容作为推演线索。"""

    role: str = Field(..., description="消息角色（system/user/assistant）")
    content: str = Field("", max_length=8000, description="消息内容")

    @field_validator("role")
    @classmethod
    def _valid_role(cls, v: str) -> str:
        if v not in ("system", "user", "assistant"):
            raise ValueError(f"非法 role: {v!r}")
        return v


class ChatCompletionIn(BaseModel):
    """POST /v1/chat/completions 请求体（OpenAI 兼容字段子集）。"""

    model: str = Field(..., min_length=1, max_length=64, description="目标脑 id")
    messages: List[_MessageIn] = Field(..., min_length=1, description="对话消息列表")
    stream: bool = Field(False, description="是否流式（SSE 增量输出，Task 5.2.3）")

    @field_validator("model")
    @classmethod
    def _valid_model(cls, v: str) -> str:
        v = v.strip()
        try:
            brain_manager.validate_brain_id(v)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return v


# ───────────────────────────────────────────────
# 模型发现 (Task 5.1)
# ───────────────────────────────────────────────

def _brain_created_epoch(brain_id: str) -> int:
    """脑目录 mtime → OpenAI created 字段（UTC 秒）；取不到时回退当前时间。"""
    try:
        # BrainSession 实例化零 IO（惰性原则），仅借用其路径计算
        brain_dir = Path(brain_manager.BrainSession(brain_id).sqlite_path).parent
        return int(brain_dir.stat().st_mtime)
    except (ValueError, OSError):
        return int(time.time())


@router.get("/models", dependencies=[Depends(require_bearer)])
def list_models():
    """OpenAI 兼容模型发现：动态列出全部已物化 brain（id 即 model id）。"""
    data = [
        {
            "id": brain_id,
            "object": "model",
            "created": _brain_created_epoch(brain_id),
            "owned_by": MODEL_OWNER,
        }
        for brain_id in brain_manager.list_brains()
    ]
    logger.debug("/v1/models 返回 %d 个模型", len(data))
    return {"object": "list", "data": data}


# ───────────────────────────────────────────────
# 推演执行 (Task 5.2.2)
# ───────────────────────────────────────────────

def _extract_clue(messages: List[_MessageIn]) -> str:
    """取最后一条 user 消息内容作为推演线索（strip 后可为空，由调用方兜底）。"""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content.strip()
    return ""


def _render_evidence_chain(answer: dict) -> str:
    """推演答案 → 证据链文本块（每跳标注 关系(权重) + evidence + 源/目标概念 id）。"""
    conclusion = answer.get("conclusion", {})
    lines = [f"结论：{conclusion.get('name', '')}（id={conclusion.get('id', '')}）"]

    assertion = answer.get("assertion")
    if assertion and assertion.get("conditions"):
        conds = "、".join(
            f"{c.get('name', '')}(id={c.get('id', '')})" for c in assertion["conditions"]
        )
        lines.append(f"命中规则断言：{conds} → {conclusion.get('name', '')}")
        for ev in assertion.get("evidence", []):
            lines.append(f"  断言依据：{ev}")

    chain = answer.get("chain", [])
    if chain:
        lines.append("推演路径（真实已确认突触）：")
        for hop in chain:
            lines.append(
                f"  {hop.get('src_id', '')} --{hop.get('relation', '')}--> "
                f"{hop.get('dst_id', '')} 依据：{hop.get('evidence', '') or '（未标注）'}"
            )
    return "\n".join(lines)


def _plan_fallback_content(brain_id: str, clue: str) -> Optional[str]:
    """augmented 模式专用：纯图无路径时叠加 LLM 辨证规划。

    LLM 未配置 / 调用失败 / 未产出有效计划 → 返回 None（调用方走固定话术）。
    计划只是「计划」而非结论 —— 文本中明确指引到控制台完成人工判定。
    """
    if not config.is_llm_configured():
        return None
    try:
        plan = reasoner.plan_for_brain(brain_id, clue)
    except Exception as exc:  # MetabolismError/ValueError 等一律兜底，绝不 5xx
        logger.warning("augmented 规划失败（走固定话术）: brain=%s err=%s", brain_id, exc)
        return None
    lines = ["纯图推演未命中路径，以下为 AI 增强模式产出的分步辨证计划（尚未形成结论）："]
    for i, step in enumerate(plan["steps"], 1):
        lines.append(f"第 {i} 步：{step['goal']}｜所需线索：{'；'.join(step['clues'])}")
    lines.append("请前往 Web 控制台对该计划逐步人工判定后固化结论。")
    return "\n".join(lines)


def _resolve_reply(brain_id: str, clue: str, mode: str) -> str:
    """Task 5.2.2 推演路由：产出最终回复正文（流式/非流式共用同一份）。"""
    # manual / assisted：推演永远是纯图确定性（assisted 仅录入环节用 LLM 提案）
    # augmented：先纯图，未命中再叠加 reasoner 辨证规划
    result = inference.pure_reason(brain_id, clue)
    if result.get("status") == "ok" and result.get("answer"):
        return _render_evidence_chain(result["answer"])

    if mode == "augmented" and clue:
        plan_content = _plan_fallback_content(brain_id, clue)
        if plan_content:
            return plan_content

    logger.info("无有效激活路径，固定话术兜底: brain=%s clue=%r", brain_id, clue[:32])
    return NO_PATH_MESSAGE


def _openai_error(status: int, message: str, code: str) -> HTTPException:
    """构造 OpenAI 风格错误（detail 为 error 对象形态）。"""
    return HTTPException(
        status_code=status,
        detail={"error": {"message": message, "type": "invalid_request_error", "code": code}},
    )


def _rough_tokens(text: str) -> int:
    """粗略 token 估算（非 LLM 推演无真实 usage，按字符数近似仅供计费参考）。"""
    return max(1, len(text) // 2)


# ───────────────────────────────────────────────
# 流式响应 (Task 5.2.3)
# ───────────────────────────────────────────────

def _content_slices(content: str, max_len: int = 48) -> List[str]:
    """完整回复 → 增量片段序列。

    证据链文本天然逐行成段（结论行 / 断言行 / 每跳路径行），故首切按行；
    超长行再按 max_len 二次切分，保证前端可见渐进式打印效果。
    """
    lines = content.split("\n")
    slices: List[str] = []
    for i, line in enumerate(lines):
        piece = line + ("\n" if i < len(lines) - 1 else "")  # 保留行间换行
        for j in range(0, len(piece), max_len):
            part = piece[j:j + max_len]
            if part:
                slices.append(part)
    return slices or [""]


def _sse_chunk(
    completion_id: str,
    created: int,
    brain_id: str,
    delta: dict,
    finish_reason: Optional[str] = None,
) -> str:
    """单个 OpenAI 兼容 chat.completion.chunk 的 SSE 帧文本（data: 前缀 + 空行结尾）。"""
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": brain_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_stream(completion_id: str, created: int, brain_id: str, content: str):
    """SSE 事件生成器：首帧 role → 证据链逐段 content 帧 → stop 终帧 → [DONE]。"""
    # 首帧：OpenAI 惯例先声明 assistant 角色
    yield _sse_chunk(completion_id, created, brain_id, {"role": "assistant", "content": ""})
    # 增量帧：与非流式完全相同的正文，逐段下发
    for piece in _content_slices(content):
        yield _sse_chunk(completion_id, created, brain_id, {"content": piece})
    # 终帧：finish_reason=stop 后随 [DONE] 哨兵
    yield _sse_chunk(completion_id, created, brain_id, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


@router.post("/chat/completions", dependencies=[Depends(require_bearer)])
def chat_completions(req: ChatCompletionIn):
    """OpenAI 兼容推演端点：按模式路由推演，流式/非流式双形态（Task 5.2.2/5.2.3）。"""
    brain_id = req.model
    if brain_id not in brain_manager.list_brains():
        raise _openai_error(
            404, f"The model '{brain_id}' does not exist", "model_not_found"
        )

    clue = _extract_clue(req.messages)
    mode = config.get_mode()
    logger.info(
        "chat/completions: brain=%s mode=%s stream=%s clue=%r",
        brain_id, mode, req.stream, clue[:32],
    )

    content = _resolve_reply(brain_id, clue, mode)
    completion_id = f"chatcmpl_{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _sse_stream(completion_id, created, brain_id, content),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 防 nginx 等反代缓冲打乱 SSE 时序
            },
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": brain_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": _rough_tokens(clue),
            "completion_tokens": _rough_tokens(content),
            "total_tokens": _rough_tokens(clue) + _rough_tokens(content),
        },
    }


if __name__ == "__main__":
    # 零网络冒烟自检：
    #   A. /v1/models 鉴权缺失401 / 凭证错误401 / 正确凭证200；
    #   B. /v1/chat/completions 非流式 JSON 证据链 + 流式 SSE 增量
    #      （Task 5.2.2/5.2.3，推演函数打补丁避免依赖真实图谱与 LLM）。
    import json as _json
    import logging as _logging
    import shutil
    import tempfile

    _logging.basicConfig(level=logging.INFO)
    _self = sys.modules[__name__]

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from storage import persistence as _persistence

    # 固定补丁：Token 校验 / 运行模式 / 纯图推演返回（同 reasoner 自检手法）
    _TOKEN = "sm_smoke_token"
    _orig_verify = _self.verify_api_token
    _orig_get_mode = _self.config.get_mode
    _orig_pure_reason = _self.inference.pure_reason
    _self.verify_api_token = lambda tok: tok == _TOKEN
    _self.config.get_mode = lambda: "manual"

    # 模拟一次多前提汇流命中的推演答案（A+B⇒C，双入边证据链）
    _FAKE_ANSWER = {
        "conclusion": {"id": "c", "name": "表实证"},
        "energy": 0.40,
        "peak": 0.81,
        "hop": 1,
        "chain": [
            {"src_id": "a", "rel_id": "s1", "relation": "rule",
             "evidence": "《伤寒论》第1条", "dst_id": "c"},
            {"src_id": "b", "rel_id": "s2", "relation": "rule",
             "evidence": "《伤寒论》第1条", "dst_id": "c"},
        ],
        "assertion": {
            "conditions": [{"id": "a", "name": "恶寒"}, {"id": "b", "name": "发热"}],
            "evidence": ["《伤寒论》第1条"],
        },
    }

    def _fake_pure_reason(brain_id, clue, **kw):
        if (clue or "").strip():
            return {"status": "ok", "message": "ok", "answer": dict(_FAKE_ANSWER)}
        return {"status": "insufficient", "message": "x", "answer": None}

    _self.inference.pure_reason = _fake_pure_reason

    tmp = Path(tempfile.mkdtemp(prefix="smoke_openai_"))
    _orig_runtime = _persistence.RUNTIME_DIR
    _persistence.RUNTIME_DIR = tmp  # brain_manager 运行期动态读取，patch 生效
    try:
        d = tmp / "brain_alpha"
        d.mkdir(parents=True)
        (d / "brain.db").write_bytes(b"")  # 物化判定标准：含 brain.db

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        auth = {"Authorization": f"Bearer {_TOKEN}"}

        # A) 模型发现三路径
        resp = client.get("/v1/models")
        assert resp.status_code == 401, resp.status_code
        assert resp.headers.get("www-authenticate") == "Bearer", resp.headers
        print("缺失凭证 401 ✓")

        resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401, resp.status_code
        print("无效凭证 401 ✓")

        resp = client.get("/v1/models", headers=auth)
        assert resp.status_code == 200, resp.text
        assert [m["id"] for m in resp.json()["data"]] == ["brain_alpha"], resp.text
        print("模型发现 ✓")

        payload = {
            "model": "brain_alpha",
            "messages": [{"role": "user", "content": "恶寒 发热"}],
        }

        # B1) 非流式 JSON：正文内嵌真实证据链并标注来源概念 id
        resp = client.post("/v1/chat/completions", headers=auth, json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["object"] == "chat.completion" and body["model"] == "brain_alpha", body
        assert body["choices"][0]["finish_reason"] == "stop", body
        content = body["choices"][0]["message"]["content"]
        assert "结论：表实证（id=c）" in content, content
        assert "恶寒(id=a)、发热(id=b)" in content, content
        assert "a --rule--> c" in content and "《伤寒论》第1条" in content, content
        assert body["usage"]["total_tokens"] > 0, body
        print("非流式 JSON 证据链 ✓")

        # B2) 流式 SSE：多帧增量 + 拼接必须与非流式逐字一致
        resp = client.post(
            "/v1/chat/completions", headers=auth, json={**payload, "stream": True}
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/event-stream"), resp.headers
        frames = [f for f in resp.text.split("\n\n") if f.startswith("data: ")]
        assert frames[-1] == "data: [DONE]", frames[-1]
        chunks = [_json.loads(f[len("data: "):]) for f in frames[:-1]]
        assert all(c["object"] == "chat.completion.chunk" for c in chunks), chunks[0]
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant", chunks[0]
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop", chunks[-1]
        assert len(chunks) > 3, f"应多帧增量而非整段一帧: {len(chunks)}"
        streamed = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        assert streamed == content, "流式拼接必须与非流式正文逐字一致"
        print(f"流式 SSE 增量 ✓ 共 {len(chunks)} 帧，拼接与非流式正文一致")

        # B3) 无路径兜底（空线索 → insufficient），流式下同样输出固定话术
        resp = client.post(
            "/v1/chat/completions",
            headers=auth,
            json={"model": "brain_alpha",
                  "messages": [{"role": "user", "content": "   "}], "stream": True},
        )
        assert resp.status_code == 200, resp.text
        assert NO_PATH_MESSAGE in resp.text, resp.text
        assert resp.text.rstrip().endswith("data: [DONE]"), resp.text[-80:]
        print("无路径固定话术兜底（流式）✓")
        print("openai_routes 冒烟自检全部通过 ✓")
    finally:
        _self.verify_api_token = _orig_verify
        _self.config.get_mode = _orig_get_mode
        _self.inference.pure_reason = _orig_pure_reason
        _persistence.RUNTIME_DIR = _orig_runtime
        shutil.rmtree(tmp, ignore_errors=True)

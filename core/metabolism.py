"""
SynapseMind - LLM 代谢器客户端 (core/metabolism.py)

Task 4.2.1: OpenAI 兼容 /chat/completions 的统一调用封装。

定位与边界：
- 本模块只负责「把请求安全地送到 LLM 并把响应完整拿回来」这一件事；
  Prompt 设计与结构化解析属 Task 4.2.2，规划器属 Task 4.3.1，均不在此处；
- 配置来源：config.get_llm() 的 {base_url, api_key, model} 三元组；
  手工模式下字段允许为空 —— 是否可用由 is_llm_configured() 预检，
  未配置时调用 chat_completion 会直接抛 MetabolismError，绝不发出残缺请求；
- 错误包装铁律：网络层/协议层的任何异常都必须翻译为带中文语义的
  MetabolismError，绝不让 httpx 原生异常裸抛到路由层（防栈泄漏），
  同时保留原始异常链（raise ... from exc）供日志诊断。

URL 拼接规则（OpenAI 兼容端点惯例）：
- base_url 允许三种形态：`https://host`、`https://host/v1`、`https://host/v1/`；
- 统一规范化为 `<base>/chat/completions`：自动补 /v1（若无）并去尾斜杠；
  这样用户在设置页只需粘贴厂商控制台给的根地址即可，无需关心路径细节。

本文件实现：
- MetabolismError      统一异常类型
- is_llm_configured()  三元组预检
- chat_completion()    同步调用（脚本/自检用）
- achat_completion()   异步调用（FastAPI 路由用，不阻塞事件循环）

传输层韧性（Bugfix: SSL UNEXPECTED_EOF_WHILE_READING）：
- 网关/代理在 TLS 握手或长连接上偶发切断（SSL EOF / 连接重置 / 读中断）
  属瞬时故障，一次失败即抛会误伤「配置完全正确」的用户；
- 因此对传输层瞬时错误做「至多 3 次尝试 + 指数退避」自动重试；
- 客户端显式 trust_env=False：容器内 HTTP(S)_PROXY 环境变量常把
  https 流量劫持到不支持 CONNECT 隧道的代理，正是 SSL EOF 高发源头；
- 4xx（鉴权/参数错误）语义明确，绝不重试，避免无意义打点。
"""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from config import get_llm

logger = logging.getLogger("synapsemind.metabolism")

# 默认请求超时（秒）：LLM 生成普遍较慢，取宽裕值；调用方可按场景覆盖
DEFAULT_TIMEOUT = 120.0
# 传输层瞬时故障自动重试：总尝试次数与指数退避基数秒
LLM_MAX_ATTEMPTS = 3
LLM_RETRY_BACKOFF = 1.5
# 默认生成参数：代谢类任务（速读/拆解）要求稳定，temperature 取低值
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2048


class MetabolismError(Exception):
    """LLM 代谢链路统一业务异常：message 面向用户可读，cause 保留原始异常链。"""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None,
                 status_code: Optional[int] = None):
        super().__init__(message)
        self.cause = cause
        self.status_code = status_code  # 上游 HTTP 状态码（网络层错误时为 None）


def is_llm_configured() -> bool:
    """三元组预检：base_url 与 model 至少齐备才认为可用（api_key 允许本地网关免鉴权）。"""
    llm = get_llm()
    return bool(llm.get("base_url")) and bool(llm.get("model"))


def _endpoint_url() -> str:
    """从 config 读取 base_url 并规范化为完整 chat/completions 端点。"""
    base = (get_llm().get("base_url") or "").strip().rstrip("/")
    if not base:
        raise MetabolismError("LLM 未配置：请先在设置页填写 base_url 与 model")
    if not base.endswith("/v1"):
        base = f"{base}/v1"  # 兼容用户只填 https://host 的形态
    return f"{base}/chat/completions"


def _build_payload(messages: List[Dict[str, str]], *, temperature: float,
                   max_tokens: int, stream: bool = False) -> Dict[str, Any]:
    """构造 OpenAI 兼容请求体（model 从配置注入，消息形态原样透传）。"""
    model = (get_llm().get("model") or "").strip()
    if not model:
        raise MetabolismError("LLM 未配置：模型名（model）为空")
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def _auth_headers() -> Dict[str, str]:
    """鉴权头：api_key 为空时不带 Authorization 头（本地 Ollama/LM Studio 场景）。"""
    headers = {"Content-Type": "application/json"}
    api_key = (get_llm().get("api_key") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _unwrap_response(data: Dict[str, Any]) -> str:
    """从 OpenAI 兼容响应中取出首个 choice 的文本；结构异常一律翻译为业务错误。"""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise MetabolismError(f"LLM 响应缺少 choices：{str(data)[:300]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        # 兼容某些网关把内容放在 text 字段的情况
        content = choices[0].get("text")
    if not isinstance(content, str):
        raise MetabolismError("LLM 响应的 content 不是文本")
    return content


def _wrap_http_error(exc: httpx.HTTPError) -> MetabolismError:
    """把 httpx 异常族翻译为用户可读的 MetabolismError（保留原始异常链）。"""
    if isinstance(exc, httpx.TimeoutException):
        return MetabolismError(f"LLM 请求超时（>{DEFAULT_TIMEOUT:.0f}s）", cause=exc)
    if isinstance(exc, httpx.ConnectError):
        return MetabolismError(f"无法连接 LLM 服务: {exc}", cause=exc)
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:300]
        return MetabolismError(
            f"LLM 服务返回 {exc.response.status_code}: {body}",
            cause=exc, status_code=exc.response.status_code,
        )
    return MetabolismError(f"LLM 请求失败: {exc}", cause=exc)


def _is_transient_error(exc: BaseException) -> bool:
    """判定是否为值得重试的传输层瞬时故障。

    覆盖（Bugfix 实测）：SSL UNEXPECTED_EOF_WHILE_READING / ConnectionReset /
    RemoteProtocolError(读中断) / ConnectError / 超时 / 5xx。
    4xx 属语义明确的客户端错误（鉴权/参数），重试无意义，不在此列。
    """
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True  # 含 SSL EOF、连接重置、读写中断、ConnectError 全族
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


def _safe_client(timeout: float) -> httpx.Client:
    """构造出站 HTTP 客户端。

    trust_env=False（核心修复）：忽略 HTTP(S)_PROXY/NO_PROXY 等环境变量。
    容器环境的代理变量常把 https 请求劫持到仅支持明文转发的代理，
    TLS 握手被中途掐断即报 SSL: UNEXPECTED_EOF_WHILE_READING。
    LLM 端点为用户显式配置的可信地址，直连即可，无需环境代理。
    """
    return httpx.Client(timeout=timeout, trust_env=False)


def _async_safe_client(timeout: float) -> httpx.AsyncClient:
    """异步版安全客户端，语义与 _safe_client 严格一致。"""
    return httpx.AsyncClient(timeout=timeout, trust_env=False)


def _retry_sleep(attempt: int) -> None:
    """指数退避：第 1 次失败后 1.5s，第 2 次后 3s（attempt 从 1 计）。"""
    time.sleep(LLM_RETRY_BACKOFF * attempt)


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """同步版补全调用：返回首个 choice 文本；任何失败抛 MetabolismError。

    传输层瞬时故障（SSL EOF/连接重置/超时/5xx）自动重试至多
    LLM_MAX_ATTEMPTS 次（指数退避）；4xx 等语义错误立即抛出。
    """
    url = _endpoint_url()
    payload = _build_payload(messages, temperature=temperature, max_tokens=max_tokens)
    last_exc: Exception = MetabolismError("未执行")
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            with _safe_client(timeout) as client:
                resp = client.post(url, json=payload, headers=_auth_headers())
                resp.raise_for_status()
                return _unwrap_response(resp.json())
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < LLM_MAX_ATTEMPTS and _is_transient_error(exc):
                logger.warning("LLM 传输层瞬时失败（第 %d/%d 次），退避后重试: %s",
                               attempt, LLM_MAX_ATTEMPTS, exc)
                _retry_sleep(attempt)
                continue
            raise _wrap_http_error(exc) from exc
        except ValueError as exc:  # resp.json() 解析失败
            raise MetabolismError("LLM 响应不是合法 JSON", cause=exc) from exc
    raise _wrap_http_error(last_exc)  # 理论不可达，防御性兜底


async def achat_completion(
    messages: List[Dict[str, str]],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """异步版补全调用（FastAPI 路由专用，不阻塞事件循环）；语义与同步版严格一致。

    同样具备传输层瞬时故障自动重试（asyncio.sleep 退避，不阻塞事件循环）。
    """
    import asyncio  # 就近引入：仅异步路径需要事件循环级 sleep
    url = _endpoint_url()
    payload = _build_payload(messages, temperature=temperature, max_tokens=max_tokens)
    last_exc: Exception = MetabolismError("未执行")
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            async with _async_safe_client(timeout) as client:
                resp = await client.post(url, json=payload, headers=_auth_headers())
                resp.raise_for_status()
                return _unwrap_response(resp.json())
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < LLM_MAX_ATTEMPTS and _is_transient_error(exc):
                logger.warning("LLM 传输层瞬时失败（第 %d/%d 次），退避后重试: %s",
                               attempt, LLM_MAX_ATTEMPTS, exc)
                await asyncio.sleep(LLM_RETRY_BACKOFF * attempt)
                continue
            raise _wrap_http_error(exc) from exc
        except ValueError as exc:
            raise MetabolismError("LLM 响应不是合法 JSON", cause=exc) from exc
    raise _wrap_http_error(last_exc)  # 理论不可达，防御性兜底


# ───────────────────────────────────────────────
# 速读秘书：用户资料 → 突触提案 (Task 4.2.2)
# ───────────────────────────────────────────────

import json  # noqa: E402  速读秘书 JSON 解析所需（随功能区块就近引入）

# 提案字段白名单：LLM 返回中的任何额外字段（confidence/id/merge 等）
# 一律视为越权直接丢弃 —— 尤其警惕「合并概念」「改名」类建议，
# 红线：速读秘书只产出提案数据，绝不允许任何自动合并/改名动作。
PROPOSAL_KEYS = ("source", "target", "relation", "weight", "evidence")

_SKIM_SYSTEM_PROMPT = (
    "你是知识图谱的「速读秘书」。阅读用户资料，抽取概念之间的有向关系，"
    "输出突触提案 JSON 数组。规则：\n"
    "1. 仅输出 JSON 数组本身，禁止任何解释文字、前言或 markdown 代码块标记。\n"
    '2. 数组每项形如 {"source": "概念A", "target": "概念B", '
    '"relation": "导致", "weight": 0.8, "evidence": "原文依据片段"}。\n'
    "3. source/target/relation 一律使用中文：source/target 取原文关键概念短语"
    "（≤32 字）；relation 用简短中文关系词（如 导致/抑制/提示/属于/主治/表现为/"
    "包含/依赖，优先沿用原文措辞）；weight 为 0~1 小数表示关系确定度；"
    "evidence 为支撑该关系的原文片段（≤100 字，可为空字符串）。\n"
    "4. 严禁提出概念合并、改名、删除等任何修改既有图谱的建议。\n"
    "5. 资料中无任何可抽取关系时输出 []。"
)


def _extract_json_array(text: str) -> list:
    """从 LLM 原始返回中尽力提取 JSON 数组（容忍 markdown 包裹与前后废话）。"""
    cleaned = (text or "").strip()
    if "```" in cleaned:
        for seg in cleaned.split("```"):
            seg = seg.strip()
            if seg.startswith("json"):
                seg = seg[4:].strip()
            if seg.startswith("["):
                cleaned = seg
                break
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # 兜底：截取首个 '[' 到最后一个 ']' 之间的片段重试
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start == -1 or end <= start:
            raise
        data = json.loads(cleaned[start:end + 1])
    if isinstance(data, dict):          # 某些模型包一层 {"proposals": [...]}
        data = data.get("proposals")
    if not isinstance(data, list):
        raise ValueError("顶层结构不是 JSON 数组")
    return data


def _sanitize_proposal(item) -> Optional[dict]:
    """单条提案净化：白名单取值 + 类型/范围校验；非法条目丢弃并记日志。"""
    if not isinstance(item, dict):
        logger.warning("速读提案丢弃（非对象）: %r", str(item)[:200])
        return None

    # 越权字段检测：字段绝不采纳，含 merge/rename 等危险建议时明确告警
    extras = set(item) - set(PROPOSAL_KEYS)
    if extras:
        logger.warning("速读提案越权字段丢弃 %s（合并/改名类建议一律忽略）", sorted(extras))

    source = str(item.get("source") or "").strip()
    target = str(item.get("target") or "").strip()
    relation = str(item.get("relation") or "").strip()
    if not source or not target or not relation:
        logger.warning("速读提案丢弃（source/target/relation 存在空值）: %r", str(item)[:200])
        return None

    try:
        weight = float(item.get("weight", 0.5))
    except (TypeError, ValueError):
        logger.warning("速读提案丢弃（weight 非数值）: %r", str(item)[:200])
        return None
    weight = min(1.0, max(0.0, weight))  # 越界 clamp 回 [0,1]（与突触权重同域）

    evidence = item.get("evidence")
    evidence = str(evidence).strip() if evidence is not None else ""

    return {
        "source": source[:64],
        "target": target[:64],
        "relation": relation[:64],
        "weight": round(weight, 4),
        "evidence": evidence[:1000],
    }


def _parse_skim_raw(raw: str) -> dict:
    """LLM 原始返回 → 净化提案列表；整体解析失败抛 MetabolismError。

    铁律：整体无法解析时记录原始返回供人工检查并抛错，绝不把半截数据
    交给下游入库（入库动作本身属 Task 4.2.3，且必须整批人工确认）。
    """
    logger.debug("速读秘书原始返回: %s", (raw or "")[:2000])
    try:
        data = _extract_json_array(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("速读秘书返回无法解析为 JSON 数组，原始返回已记录: %s", (raw or "")[:2000])
        raise MetabolismError(f"速读秘书返回无法解析: {exc}") from exc

    proposals: List[dict] = []
    dropped = 0
    for item in data:
        clean = _sanitize_proposal(item)
        if clean is None:
            dropped += 1
        else:
            proposals.append(clean)
    logger.info("速读秘书产出 %d 条提案（丢弃 %d 条非法项）", len(proposals), dropped)
    return {"proposals": proposals, "raw": raw, "dropped": dropped}


def _skim_messages(text: str) -> List[Dict[str, str]]:
    """构造速读秘书消息序列（system 规则 + user 资料定界包裹）。"""
    return [
        {"role": "system", "content": _SKIM_SYSTEM_PROMPT},
        {"role": "user", "content": f"用户资料：\n<<<\n{text.strip()}\n>>>"},
    ]


def skim_extract_proposals(text: str, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """同步版速读秘书（脚本/自检用）：返回 {proposals, raw, dropped}。"""
    if not text or not text.strip():
        raise MetabolismError("待速读的资料文本为空")
    raw = chat_completion(_skim_messages(text), timeout=timeout)
    return _parse_skim_raw(raw)


async def askim_extract_proposals(text: str, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """异步版速读秘书（FastAPI 路由用）；行为与同步版严格一致。"""
    if not text or not text.strip():
        raise MetabolismError("待速读的资料文本为空")
    raw = await achat_completion(_skim_messages(text), timeout=timeout)
    return _parse_skim_raw(raw)


# ───────────────────────────────────────────────
# 对话回流秘书：用户话语 → 突触提案 (升级C)
# ───────────────────────────────────────────────

_REFLECT_SYSTEM_PROMPT = (
    "你是知识图谱的「对话回流秘书」。用户刚与推演助手完成一轮对话，"
    "你的任务是从用户话语中识别「新增知识」或「纠正既有结论」的陈述，"
    "输出突触提案 JSON 数组供人工审核入库。规则：\n"
    "1. 仅输出 JSON 数组本身，禁止任何解释文字、前言或 markdown 代码块标记。\n"
    '2. 数组每项形如 {"source": "概念A", "target": "概念B", '
    '"relation": "导致", "weight": 0.8, "evidence": "用户原话片段"}。\n'
    "3. source/target/relation 一律使用中文：source/target 取用户话语中的关键"
    "概念短语（≤32 字）；relation 用简短中文关系词（如 导致/抑制/提示/属于/主治/"
    "表现为/包含/依赖，优先沿用用户措辞）；weight 为 0~1 小数表示表述确定度；"
    "evidence 必须摘录用户原话片段（≤100 字），保留口语原貌供审核者溯源。\n"
    "4. 仅抽取陈述性知识（经验、事实、纠正意见）；纯提问、闲聊、确认性应答"
    "（如「好的」「明白了」）一律忽略，无知识可抽时输出 []。\n"
    "5. 严禁提出概念合并、改名、删除等任何修改既有图谱的建议；用户对既有结论的"
    "否定（如「不对，A 不是 B 的病因」）应输出为修正方向的关系提案，"
    "新旧冲突的裁决权完全归人工审核。"
)


def _reflect_messages(clue: str, answer: str = "") -> List[Dict[str, str]]:
    """构造对话回流秘书消息序列（system 规则 + 话语/语境定界包裹）。

    answer 仅作纠正语境参考（用户说「不对」时秘书需知道在否定什么），
    prompt 层已明令绝不从 answer 中抽取 —— 回流只沉淀用户亲口所言。
    """
    user_content = f"用户话语：\n<<<\n{clue.strip()}\n>>>"
    if answer and answer.strip():
        user_content += (
            "\n\n（供参考的助手上一轮回答，仅作语境，绝不从中抽取）：\n<<<\n"
            + answer.strip()[:2000] + "\n>>>")
    return [
        {"role": "system", "content": _REFLECT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def reflect_extract_proposals(clue: str, answer: str = "", *,
                              timeout: float = DEFAULT_TIMEOUT) -> dict:
    """同步版对话回流秘书（脚本/自检用）：返回 {proposals, raw, dropped}。"""
    if not clue or not clue.strip():
        raise MetabolismError("对话回流解析的用户话语为空")
    raw = chat_completion(_reflect_messages(clue, answer), timeout=timeout)
    return _parse_skim_raw(raw)


async def areflect_extract_proposals(clue: str, answer: str = "", *,
                                     timeout: float = DEFAULT_TIMEOUT) -> dict:
    """异步版对话回流秘书（FastAPI 路由用）；行为与同步版严格一致。"""
    if not clue or not clue.strip():
        raise MetabolismError("对话回流解析的用户话语为空")
    raw = await achat_completion(_reflect_messages(clue, answer), timeout=timeout)
    return _parse_skim_raw(raw)


if __name__ == "__main__":
    # 冒烟自检（零网络）：
    # 1) 配置预检与错误包装路径
    import logging as _logging
    _logging.basicConfig(level=logging.INFO)

    print("is_llm_configured →", is_llm_configured())
    try:
        chat_completion([{"role": "user", "content": "ping"}], timeout=3.0)
    except MetabolismError as exc:
        print("MetabolismError（预期）:", exc)

    # 2) 速读秘书解析：monkeypatch 同步客户端，喂入典型脏返回
    # 注意：`python -m core.metabolism` 运行时本模块身份是 __main__，
    # 直接取 sys.modules[__name__] 打补丁，避免双模块实例导致 patch 落空。
    import sys as _sys
    _self = _sys.modules[__name__]

    def _fake_chat(messages, **kw):
        return (
            "好的，以下是提取结果：\n"
            "```json\n"
            "[{\"source\":\"风寒\",\"target\":\"恶寒\",\"relation\":\"导致\","
            "\"weight\":0.9,\"evidence\":\"风寒束表则恶寒\"},"
            "{\"source\":\"风寒\",\"target\":\"发热\",\"relation\":\"导致\","
            "\"weight\":1.5,\"evidence\":\"郁而发热\"},"
            "{\"source\":\"\",\"target\":\"空源头\",\"relation\":\"导致\","
            "\"weight\":0.5,\"evidence\":\"\"},"
            "{\"source\":\"恶寒\",\"target\":\"发热\",\"relation\":\"并见\","
            "\"weight\":0.7,\"evidence\":\"二症并见\",\"confidence\":0.99,"
            "\"merge\":\"建议合并恶寒与发热\"}]\n"
            "```"
        )

    _self.chat_completion = _fake_chat
    result = skim_extract_proposals("风寒束表，恶寒发热。")
    print("proposals =", result["proposals"])
    print("dropped   =", result["dropped"])
    assert len(result["proposals"]) == 3, "应为 3 条合法提案（空源头丢弃，越界 weight clamp 保留）"
    assert result["dropped"] == 1
    assert all(set(p) == set(PROPOSAL_KEYS) for p in result["proposals"]), "越权字段必须被剔除"
    assert result["proposals"][1]["weight"] == 1.0, "weight 1.5 应 clamp 为 1.0"

    # 3) 整体解析失败 → MetabolismError 包装
    _self.chat_completion = lambda m, **kw: "抱歉，我无法输出 JSON 格式。"
    try:
        skim_extract_proposals("测试资料")
    except MetabolismError as exc:
        print("解析失败包装（预期）:", str(exc)[:60])

    # 4) 升级C 对话回流秘书：monkeypatch 喂典型话语
    #    修正性话语 → 提取纠正提案；纯闲聊 → 零提案；越权字段剔除。
    def _fake_reflect(messages, **kw):
        assert "用户话语" in messages[1]["content"], "user 消息应含话语定界包裹"
        if "明白" in messages[1]["content"]:
            return "[]"
        return (
            "```json\n"
            "[{\"source\":\"桂枝\",\"target\":\"风寒表虚证\",\"relation\":\"主治\","
            "\"weight\":0.8,\"evidence\":\"我临床上桂枝治这个很管用\"},"
            "{\"source\":\"麻黄\",\"target\":\"表虚证\",\"relation\":\"主治\","
            "\"weight\":0.9,\"evidence\":\"麻黄也行\",\"confidence\":0.99,"
            "\"merge\":\"建议合并麻黄桂枝\"}]\n"
            "```"
        )

    _self.chat_completion = _fake_reflect
    refl = reflect_extract_proposals("不对，应该是桂枝主治风寒表虚证，我临床上用着很管用。",
                                     answer="（参考语境）麻黄主治表实证……")
    assert len(refl["proposals"]) == 2 and refl["dropped"] == 0, refl
    assert refl["proposals"][1]["weight"] == 0.9, "越界 weight 应 clamp"
    assert all(set(p) == set(PROPOSAL_KEYS) for p in refl["proposals"]), \
        "越权字段（merge/confidence）必须被剔除"
    chatter = reflect_extract_proposals("好的，明白了，谢谢")
    assert chatter["proposals"] == [], "纯闲聊应零提案"

    try:
        reflect_extract_proposals("   ")
        raise AssertionError("空话语应被拦截")
    except MetabolismError:
        pass

    # 5) 回流消息构造：answer 语境携带 + 空 answer 不携带
    assert "参考" in _reflect_messages("问", answer="答")[1]["content"]
    assert "参考" not in _reflect_messages("问")[1]["content"]

    print("metabolism 冒烟自检全部通过 ✓（含升级C 对话回流秘书）")

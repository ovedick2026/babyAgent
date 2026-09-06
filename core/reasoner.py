"""
SynapseMind - 增强模式推演规划器 (core/reasoner.py)

Task 4.3.1: 模式 3（AI 增强）第一环节 —— 任务拆解规划器。

链路定位：
    用户复杂诉求 + CONFIRMED 图谱上下文
      → build_graph_context()   组装受控大小的已知知识摘要
      → plan_steps()            LLM 产出分步辨证计划（每步目标 + 所需线索）
      → 严格净化：越权字段丢弃、非法步骤整条剔除、零有效步骤即报错
    （分步人工判定状态机属 Task 4.3.2，本模块不涉及会话状态）

核心铁律：
- 规划产物只是「计划」，不是结论 —— 任何步骤都不直接写入图谱；
- 图谱上下文只来自 get_confirmed_subgraph（严格已确认知识），
  PENDING / REJECTED 对规划器天然不可见；
- 解析失败 / 空计划一律整体报错并记录原始返回，绝不部分采用 ——
  与速读秘书（Task 4.2.2）同一净化哲学。
"""

import json
import logging
import re
import sys
from typing import List

from core.metabolism import MetabolismError, chat_completion

logger = logging.getLogger("synapsemind.reasoner")

# 计划形状约束：步骤数与每步线索数均设上限，防止 LLM 失控产出超长计划
MAX_STEPS = 8
MAX_CLUES_PER_STEP = 6
MAX_GOAL_LEN = 200
MAX_CLUE_LEN = 100

# 图谱上下体量约束：概念/关系条数截断，防止 token 爆炸（与局部视口原则对齐）
CONTEXT_NODE_LIMIT = 60
CONTEXT_EDGE_LIMIT = 120

_PLANNER_SYSTEM_PROMPT = (
    "你是 SynapseMind 的辨证规划师。依据用户诉求与「已知图谱知识」，"
    "把复杂诉求拆解为有序的分步推演计划。只输出一个 JSON 对象，禁止输出任何其他文字。\n"
    '格式：{"steps": [{"goal": "本步要厘清的目标", "clues": ["所需线索1", "所需线索2"]}]}\n'
    "要求：\n"
    f"1. 步骤数量 1~{MAX_STEPS} 步，每步目标单一、明确、可判定；\n"
    f"2. 每步列 1~{MAX_CLUES_PER_STEP} 条验证所需的具体线索（可引用已知概念名，或需用户补充的信息）；\n"
    "3. 优先利用已知图谱知识，禁止臆造图谱中不存在的概念名。"
)


# ───────────────────────────────────────────────
# 图谱上下文构建
# ───────────────────────────────────────────────

def build_graph_context(
    brain_id: str,
    clue: str = "",
    *,
    node_limit: int = CONTEXT_NODE_LIMIT,
    edge_limit: int = CONTEXT_EDGE_LIMIT,
) -> dict:
    """
    拉取指定脑的 CONFIRMED 子图并组装为受控大小的规划上下文。

    相关性策略（确定性规则，无 LLM 参与）：
      - 线索非空时：名称与线索互为子串的概念优先排前，其余按名称序殿后；
      - 截断至 node_limit 个概念后，仅保留两端均在保留集内的边，
        再截断至 edge_limit 条（按 rel_id 稳定序，保证两次构建一致）。

    Returns:
        {"nodes": [...], "edges": [...]}（结构与 get_confirmed_subgraph 一致）。
    """
    from core.graph_cortex import get_brain_confirmed_subgraph  # 延迟导入防环

    subgraph = get_brain_confirmed_subgraph(brain_id)
    nodes = list(subgraph.get("nodes", []))
    edges = list(subgraph.get("edges", []))

    clue_norm = (clue or "").strip().lower()
    if clue_norm:
        def _relevance(n: dict) -> int:
            name = str(n.get("name", "")).lower()
            return 0 if (name in clue_norm or clue_norm in name) else 1

        nodes.sort(key=lambda n: (_relevance(n), str(n.get("name", ""))))

    nodes = nodes[: max(1, int(node_limit))]
    keep_ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["src_id"] in keep_ids and e["dst_id"] in keep_ids]
    edges.sort(key=lambda e: e.get("rel_id", ""))
    edges = edges[: max(1, int(edge_limit))]

    logger.debug("规划上下文: %d 概念 / %d 关系", len(nodes), len(edges))
    return {"nodes": nodes, "edges": edges}


def _format_context(ctx: dict) -> dict:
    """上下文 → LLM 友好的紧凑摘要（概念用名称+分类，关系用自然语言箭头）。"""
    id2name = {n["id"]: n.get("name", "") for n in ctx.get("nodes", [])}
    relations = []
    for e in ctx.get("edges", []):
        src = id2name.get(e.get("src_id"), e.get("src_id", ""))
        dst = id2name.get(e.get("dst_id"), e.get("dst_id", ""))
        relations.append(
            f'{src} --{e.get("relation", "")}({float(e.get("weight", 0)):.2f})--> {dst}'
        )
    return {
        "概念": [f'{n.get("name", "")}({n.get("category", "")})' for n in ctx.get("nodes", [])],
        "已确认关系": relations,
    }


# ───────────────────────────────────────────────
# 规划执行与结构化净化
# ───────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    """剥除 markdown 代码围栏（```json ... ```）与首尾空白。"""
    return _FENCE_RE.sub("", text.strip()).strip()


def _sanitize_steps(raw_steps) -> List[dict]:
    """步骤数组净化：白名单字段、逐条校验，非法步骤整条丢弃。"""
    cleaned: List[dict] = []
    for item in raw_steps[:MAX_STEPS] if isinstance(raw_steps, list) else []:
        if not isinstance(item, dict):
            continue
        goal = item.get("goal")
        clues = item.get("clues")
        if not isinstance(goal, str) or not goal.strip():
            continue
        if not isinstance(clues, list):
            continue
        clean_clues = []
        for c in clues:
            if isinstance(c, str) and c.strip():
                clean_clues.append(c.strip()[:MAX_CLUE_LEN])
            if len(clean_clues) >= MAX_CLUES_PER_STEP:
                break
        if not clean_clues:
            continue  # 无线索的步骤不可执行，整条剔除
        cleaned.append({"goal": goal.strip()[:MAX_GOAL_LEN], "clues": clean_clues})
        if len(cleaned) >= MAX_STEPS:
            break
    return cleaned


def plan_steps(clue: str, graph_context: dict, *, timeout: float = 60.0) -> dict:
    """
    Task 4.3.1 主入口：诉求 + 图谱上下文 → 分步辨证计划。

    Args:
        clue:          用户复杂诉求原文（非空）。
        graph_context: build_graph_context() 的产出。
        timeout:       LLM 调用超时秒数。

    Returns:
        {"steps": [{"goal": str, "clues": [str, ...]}, ...], "raw": 原始返回文本}

    Raises:
        ValueError:      clue 为空。
        MetabolismError: LLM 未配置 / 调用失败 / 返回无法解析为有效计划。
    """
    if not isinstance(clue, str) or not clue.strip():
        raise ValueError("诉求线索不得为空")

    user_payload = json.dumps(
        {"用户诉求": clue.strip(), "已知图谱知识": _format_context(graph_context)},
        ensure_ascii=False,
    )
    raw = chat_completion(
        [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        timeout=timeout,
    )

    text = _strip_fences(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # 容错：LLM 常在 JSON 前后混入闲聊，且可能返回裸数组而非
        # {"steps": [...]} 包裹形态 —— 依次尝试截取 {...} / [...] 平衡块。
        parsed = None
        starts = sorted(i for i in (text.find("{"), text.find("[")) if i >= 0)
        ends = sorted(
            (i for i in (text.rfind("}"), text.rfind("]")) if i >= 0),
            reverse=True,
        )
        for start in starts:
            for end in ends:
                if start >= end:
                    continue
                try:
                    parsed = json.loads(text[start:end + 1])
                    break
                except json.JSONDecodeError:
                    continue
            if parsed is not None:
                break
        if parsed is None:
            logger.error("规划器返回无法解析为 JSON，原始返回已记录: %s", raw)
            raise MetabolismError(f"规划器返回无法解析: {raw[:200]}") from None

    raw_steps = parsed.get("steps") if isinstance(parsed, dict) else parsed
    steps = _sanitize_steps(raw_steps)
    if not steps:
        logger.error("规划器未产出任何有效步骤，原始返回已记录: %s", raw)
        raise MetabolismError("规划器未产出有效计划，请检查 LLM 配置或重试")

    logger.info("规划完成: 诉求=%r 有效步骤=%d", clue.strip()[:32], len(steps))
    return {"steps": steps, "raw": raw}


def plan_for_brain(brain_id: str, clue: str, *, timeout: float = 60.0) -> dict:
    """便捷入口：自动构建 brain 的 CONFIRMED 上下文后执行规划。"""
    return plan_steps(clue, build_graph_context(brain_id, clue), timeout=timeout)


if __name__ == "__main__":
    # 零网络冒烟自检：monkeypatch 同步客户端，覆盖净化/容错/失败三条路径。
    # 直接取 sys.modules[__name__] 打补丁，避免双模块实例导致 patch 落空。
    import logging as _logging

    _logging.basicConfig(level=logging.INFO)
    _self = sys.modules[__name__]

    # 1) 脏返回净化：围栏包裹 + 越权字段 + 非法步骤混杂
    def _fake_chat_ok(messages, **kw):
        return (
            '好的，计划如下：\n'
            '```json\n'
            '[{"goal": "判断表虚表实", "clues": ["恶寒", "汗出与否"], "confidence": 0.9},\n'
            ' {"goal": "", "clues": ["无目标应剔除"]},\n'
            ' {"goal": "无线索步骤应剔除", "clues": []},\n'
            ' {"goal": "定方", "clues": ["证型"]}]\n'
            '```\n以上。'
        )

    _orig = _self.chat_completion
    _self.chat_completion = _fake_chat_ok
    ctx = {
        "nodes": [
            {"id": "a", "name": "恶寒", "category": "症状", "description": ""},
            {"id": "b", "name": "表实证", "category": "证型", "description": ""},
        ],
        "edges": [
            {"rel_id": "s1", "src_id": "a", "dst_id": "b",
             "weight": 0.9, "relation": "suggests", "evidence": "e"},
        ],
    }
    result = plan_steps("恶寒发热无汗", ctx)
    assert result["steps"] == [
        {"goal": "判断表虚表实", "clues": ["恶寒", "汗出与否"]},
        {"goal": "定方", "clues": ["证型"]},
    ], f"净化结果异常: {result['steps']}"
    print("净化路径 ✓  steps =", result["steps"])

    # 2) 彻底垃圾返回 → 整体报错（绝不部分采用）
    def _fake_chat_bad(messages, **kw):
        return "抱歉，我无法按格式输出。"

    _self.chat_completion = _fake_chat_bad
    try:
        plan_steps("x", ctx)
        raise AssertionError("垃圾返回应报错")
    except MetabolismError as exc:
        print("失败包装 ✓  ", exc)
    finally:
        _self.chat_completion = _orig

    print("reasoner 冒烟自检全部通过 ✓")

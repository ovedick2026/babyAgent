"""
SynapseMind - 概念相近度弱提示引擎 (core/similarity.py)

Task 3.3.1：录入新概念时用 rapidfuzz 检索已存在概念，相似度超阈值（默认 >80%）
时返回提示结构，供前端弹窗让用户裁决「确认为不同概念 / 建立关联」。

═══════════════════════════════════════════════
红线（Roadmap 明文铁律）：
  本模块仅生成提示数据，后端绝无任何合并、改名、自动建边等副作用；
  是否视为同一概念、是否连线，主权 100% 在人工操作。
═══════════════════════════════════════════════

实现分层：
- rank_similar()            纯函数：给定候选概念 dict 列表做 rapidfuzz 打分排序，
                            不触碰任何存储，可直接离线单测；
- check_concept_similarity() 取数封装：按 brain_id 拉取全量概念后调用纯函数，
                            只读（list_concepts 为纯查询），无写入路径。
"""

import logging
from typing import List

from rapidfuzz import fuzz, process

logger = logging.getLogger("synapsemind.similarity")

# Roadmap 规范：相似度 >80% 触发弱提示
DEFAULT_THRESHOLD = 80.0
# 最多返回的相近候选数（提示弹窗不需要长列表）
DEFAULT_LIMIT = 5


def rank_similar(
    name: str,
    concepts: List[dict],
    threshold: float = DEFAULT_THRESHOLD,
    limit: int = DEFAULT_LIMIT,
) -> List[dict]:
    """纯函数：对新概念名与既有概念列表做相近度打分排序。

    采用复合打分：max(fuzz.ratio, fuzz.partial_ratio)。
    - ratio: 整体字符级相似度，覆盖「恶寒/恶寒证」这类小改动的对称场景；
    - partial_ratio: 部分匹配，覆盖「新名完整包含旧名」的最典型近似重复
      场景（如已有「恶寒」，录入「恶寒发热」ratio 仅 ~67 但 partial=100）；
      弱提示宁可多提示、由人工裁决，故取两者较大值防漏报。
    阈值判定为「严格大于」（score > threshold 才入选），不依赖
    rapidfuzz score_cutoff 在不同版本间的 >= / > 边界语义，由本函数
    自行过滤，保证行为一致。结果按分数降序截断前 limit 个。

    Args:
        name:      待录入的新概念名。
        concepts:  既有概念列表，元素含 id/name/category/description 键。
        threshold: 相似度阈值（不含边界，严格 >）。
        limit:     返回上限。

    Returns:
        [{"id", "name", "category", "description", "score"}, ...]，
        score ∈ (threshold, 100]，按降序排列。无命中返回空列表。
    """
    name = (name or "").strip()
    if not name or not concepts:
        return []

    # choices 以概念 id 为键；extract 返回 (匹配文本, 分数, id) 三元组。
    # 不传 score_cutoff（规避版本间边界语义差异），全量打分后自行严格过滤。
    # 复合打分见 docstring：ratio 与 partial_ratio 取 max，防包含型重复漏报。
    # **_kw 兼容 rapidfuzz 各版本向 scorer 注入的 score_cutoff 等额外参数。
    def _composite_score(query: str, choice: str, **_kw) -> float:
        return max(fuzz.ratio(query, choice), fuzz.partial_ratio(query, choice))

    choices = {c["id"]: c.get("name", "") for c in concepts}
    hits = process.extract(name, choices, scorer=_composite_score)

    by_id = {c["id"]: c for c in concepts}
    ranked = []
    for _text, score, cid in hits:
        if score <= threshold:  # 红线判定：严格大于阈值才提示
            continue
        src = by_id[cid]
        ranked.append({
            "id": cid,
            "name": src.get("name", ""),
            "category": src.get("category", ""),
            "description": src.get("description", ""),
            "score": round(float(score), 2),
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:max(1, int(limit))]


def check_concept_similarity(
    brain_id: str,
    name: str,
    threshold: float = DEFAULT_THRESHOLD,
    limit: int = DEFAULT_LIMIT,
) -> dict:
    """取数封装：检查 brain 内与待录入概念名相近的既有概念。

    只读操作：经 brain_manager 会话取 Kùzu 连接 → graph_cortex.list_concepts
    纯查询 → rank_similar 打分。无任何写入/合并/改名副作用（模块红线）。

    Returns:
        {
          "query":      {"name": 待录入名},
          "threshold":  阈值,
          "candidates": [{id, name, category, description, score}, ...],
          "hint":       供前端直接展示的提示文案（无命中为空串），
        }
    """
    from core.brain_manager import get_or_create  # 延迟导入，避免环依赖
    from core.graph_cortex import list_concepts

    session = get_or_create(brain_id)
    conn = session.get_kuzu()
    concepts = list_concepts(conn, limit=2000)  # 上限兜底，防超大图谱拖垮打分

    candidates = rank_similar(name, concepts, threshold=threshold, limit=limit)
    if candidates:
        top = candidates[0]
        hint = (
            f"已存在相近概念「{top['name']}」（相似度 {top['score']}%），"
            "请确认是不同概念、还是重复录入，或应建立关联。"  # 仅提示，不代做决定
        )
        logger.info("相似度提示: brain=%s query=%r 命中 %d 条", brain_id, name, len(candidates))
    else:
        hint = ""

    return {
        "query": {"name": name},
        "threshold": threshold,
        "candidates": candidates,
        "hint": hint,
    }

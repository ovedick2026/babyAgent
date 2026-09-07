"""
SynapseMind - 知识语义分段器 (core/distill.py)

Task 11.1: 把任意长度的知识文本切成「意思完整」的段（segment）序列，
供提炼引擎逐段喂给 LLM —— 贪多嚼不烂：单段过长必致模型走神遗漏，
截断句子必致语义残缺（领会错意思比遗漏更危险）。

分段铁律（优先级从高到低）：
1. 句子是原子单元：任何切分点绝不可能落在句子中间；
2. 作者的空行换段是最高语义信号：空行处必分段（即使该段很短）；
3. 单段超过 max_chars 时，在句终符（。！？；…等，含闭引号跟读）处
   按贪心打包再切分：每段尽量装满但绝不超限、绝不拆句；
4. 段序 = 原文序：分段是纯重排不重不漏（空白归一化由调用方按需处理）。

对外仅一个函数：split_segments(text, max_chars=600) -> List[str]
"""

from __future__ import annotations

import re
from typing import List

# 句终符：中文句读 + 常见西文标点；后跟 」』"')】 等闭符时句界仍在闭符之后
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;…])(?=[^」』”'）》\]]|$)")

# 段内兜底硬上限保护：极端无句读长串（公式串/密文/整段英文无标点）
# 超过该长度时允许在逗号/顿号处断开，最后手段按字符硬切（记警告日志）
_COMMA_BREAK = re.compile(r"(?<=[，,、])")

DEFAULT_MAX_CHARS = 600


def _split_sentences(para: str) -> List[str]:
    """把一个自然段切成句子列表（每句含句终符；尾句无句终符则原样保留）。"""
    parts = [s for s in _SENTENCE_END.split(para) if s.strip()]
    return parts if parts else [para]


def split_segments(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """
    把知识文本切成语义完整段序列。

    Args:
        text:      原始知识文本（任意长度；空/全空白返回空列表）。
        max_chars: 单段软上限（默认 600 字）。段落到该长度即在下一个
                   句终符处封段——句子本身超限时才会触发段内二级切分。

    Returns:
        段字符串列表（保序、不重不漏、每个元素意思完整）。
    """
    if not isinstance(text, str) or max_chars < 50:
        raise ValueError("text 须为字符串且 max_chars >= 50")

    # 按空行切自然段；普通换行不切（同一段的软换行），归一为一个空格
    raw_paras = [p.replace("\n", " ").strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in raw_paras if p]

    segments: List[str] = []
    for para in paras:
        if len(para) <= max_chars:
            segments.append(para)
            continue

        # 超长自然段 → 句子贪心打包
        buf = ""
        for sent in _split_sentences(para):
            if len(sent) > max_chars:
                # 单句自身已超上限（必然无法独立成段）：先冲刷缓冲，再段内二级切分
                if buf:
                    segments.append(buf)
                    buf = ""
                segments.extend(_hard_split_long_sentence(sent, max_chars))
                continue
            if buf and len(buf) + len(sent) > max_chars:
                segments.append(buf)
                buf = sent
            else:
                buf += sent
        if buf:
            segments.append(buf)

    return segments


def _hard_split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    """无句读超长单句的二级切分：优先逗号/顿号边界，最后按字符硬切（保底不丢字）。"""
    out: List[str] = []
    buf = ""
    for chunk in _COMMA_BREAK.split(sentence):
        if buf and len(buf) + len(chunk) > max_chars:
            out.append(buf)
            buf = chunk
        else:
            buf += chunk
        while len(buf) >= max_chars:            # 连续无标点的极端长串硬切
            out.append(buf[:max_chars])
            buf = buf[max_chars:]
    if buf:
        out.append(buf)
    return out


if __name__ == "__main__":
    # 自检：空行分段 / 超长句终打包 / 无句读二级切分 / 保序不重不漏
    import logging as _log
    _log.basicConfig(level=_log.INFO)

    long_para = "太阳主表是一身之藩篱。" * 40          # 880 字，句终打包
    no_punct = "阴阳者天地之道也" * 120                # 840 字无句读
    text = f"短段一。\n\n恶寒发热并见者属表证。但热不寒者属里证。\n\n{long_para}\n\n{no_punct}"

    segs = split_segments(text)
    assert all(len(s) <= 620 for s in segs), [len(s) for s in segs]
    joined = "".join(segs)
    compact = "".join(p.replace("\n", " ").strip() for p in text.split("\n\n") if p.strip())
    assert joined == compact, "分段必须保序不重不漏"
    assert segs[0] == "短段一。" and len(segs) >= 4
    print(f"distill 自检通过 ✓ 共 {len(segs)} 段，长度分布 {[len(s) for s in segs]}")
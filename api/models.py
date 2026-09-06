"""
SynapseMind - 手工录入 Pydantic 请求模型 (api/models.py)

Task 4.1.1：模式 1（纯手工）录入链路的请求体定义与字段校验。

模型清单：
- ConceptIn      概念节点录入（name/category/description）
- SynapseIn      突触连线录入（有向 src→dst，weight ∈ [0,1]）
- RuleAssertion  "A + B 导致 C" 型规则断言（多条件 → 单结论）

校验对齐原则：字段约束与 core 层硬约束严格一致 ——
- 概念/突触 id 格式同 graph_cortex._ID_RE（[A-Za-z0-9_-]{1,64}），
  保证前端录入的 id 能直接通过 core 层二次校验，不在 API 层产生"合法差异"；
- weight 与能量扩散传导系数同域 [0,1]；
- 所有文本字段 strip 后校验，拒绝纯空白输入。

本模块只做"数据形状"定义，零业务副作用；入库动作由路由层调用 core 完成。
"""

import re
from typing import List

from pydantic import BaseModel, Field, field_validator

# 与 core/graph_cortex._ID_RE 保持一致的 id 约束（避免跨模块私有引用）
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class ConceptIn(BaseModel):
    """概念节点录入请求。"""

    name: str = Field(..., min_length=1, max_length=64, description="概念名（同脑内允许重名，id 由后端生成）")
    category: str = Field(..., min_length=1, max_length=32, description="分类标签（如 症状/证型/方剂）")
    description: str = Field("", max_length=500, description="概念释义（可选）")

    @field_validator("name", "category")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("字段不能为纯空白")
        return v

    @field_validator("description")
    @classmethod
    def _strip_description(cls, v: str) -> str:
        return v.strip()


class SynapseIn(BaseModel):
    """突触连线录入请求（有向：src → dst）。"""

    src_id: str = Field(..., description="起点概念 id")
    dst_id: str = Field(..., description="终点概念 id")
    relation: str = Field(..., min_length=1, max_length=64, description="关系标签（causes/inhibits/relates 等）")
    weight: float = Field(0.5, ge=0.0, le=1.0, description="传导权重 ∈ [0,1]，能量扩散的唯一系数")
    evidence: str = Field("", max_length=1000, description="录入依据/出处（可选，回溯证据链用）")

    @field_validator("src_id", "dst_id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        v = v.strip()
        if not _ID_RE.match(v):
            raise ValueError(f"非法概念 id: {v!r}（仅允许字母/数字/下划线/连字符，长度 1~64）")
        return v

    @field_validator("relation")
    @classmethod
    def _reject_blank_relation(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("关系标签不能为纯空白")
        return v

    @field_validator("evidence")
    @classmethod
    def _strip_evidence(cls, v: str) -> str:
        return v.strip()


class RuleAssertion(BaseModel):
    """"A + B 导致 C" 型规则断言请求：多条件组合 → 单一结论。

    conditions / conclusion 存概念名（人工录入的自然语言形态），
    由路由层负责解析/匹配到具体概念 id（找不到时明确报错，绝不静默造点）。
    """

    conditions: List[str] = Field(..., min_length=1, max_length=8, description="前提条件概念名列表（1~8 个）")
    conclusion: str = Field(..., min_length=1, max_length=64, description="结论概念名")
    evidence: str = Field("", max_length=1000, description="规则出处（经典条文/医案等，可选）")

    @field_validator("conditions")
    @classmethod
    def _clean_conditions(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip() for s in v if s and s.strip()]
        if not cleaned:
            raise ValueError("前提条件不能全部为空白")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("前提条件存在重复项")
        return cleaned

    @field_validator("conclusion")
    @classmethod
    def _reject_blank_conclusion(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("结论不能为纯空白")
        return v

    @field_validator("evidence")
    @classmethod
    def _strip_rule_evidence(cls, v: str) -> str:
        return v.strip()

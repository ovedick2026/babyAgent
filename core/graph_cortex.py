"""
SynapseMind - 认知拓扑皮层 (core/graph_cortex.py)

知识图谱的物化载体：Kùzu 图数据库中的概念节点（Concept）与突触边（SYNAPSE）。

核心铁律（与 Readme.md 对齐，后续所有 CRUD 必须遵守）：
- 只有 status='CONFIRMED' 的突触才代表「已学知识」，推演引擎
  （activation / reasoner）严禁读取 PENDING / REJECTED 边；
- 状态机：PENDING → CONFIRMED / REJECTED，CONFIRMED 仅由人工触发，
  任何自动流程不得越权修改突触状态；
- name 在同脑内允许重复（「恶寒」与「怕冷」病机不同，绝不静默合并），
  概念身份以 id 为唯一准绳。

本文件当前实现：
- Task 3.1.1  init_schema(conn): Kùzu Schema 幂等初始化

后续任务将在此文件追加：
- Task 3.1.2  概念节点 CRUD: add_concept / get_concept / update_concept /
              delete_concept / list_concepts
- Task 3.1.3  突触边 CRUD 与状态流转: add_synapse / update_synapse / delete_synapse
- Task 3.1.4  CONFIRMED 子图只读查询: get_confirmed_subgraph
"""

import logging

logger = logging.getLogger("synapsemind.graph_cortex")

# 突触合法状态取值域（状态机常量，供 Task 3.1.3 与前端审核面板复用）
SYNAPSE_STATUSES = ("PENDING", "CONFIRMED", "REJECTED")

# ───────────────────────────────────────────────
# Schema DDL（Task 3.1.1 基线 + Task 3.1.3 增补 rel_id 稳定寻址）
# ───────────────────────────────────────────────

# 知识来源标记（升级D·来源分级权限）。
# 分级语义：intake=书籍类（文档投喂提炼，禁止经对话修改，仅人工可改）；
# manual/dialogue=经验类（人工手录/对话回流，可经对话修改）；
# legacy=加字段前的存量数据（从严按书籍类管理，人工可手动改标记）。
KNOWLEDGE_SOURCES = ("intake", "manual", "dialogue", "legacy")
DEFAULT_KNOWLEDGE_SOURCE = "intake"   # 缺省从严：未显式标记的一律按书籍类

# 知识类别分级（升级D 主体）：把四种来源归并为两大权限类别。
# book（书籍类）= intake/legacy：投喂提炼与存量迁移，权威性强，
#   禁止经对话修改（对话回流提案命中即拒），仅限人工手动修改；
# experience（经验类）= manual/dialogue：用户亲笔手录与对话回流，
#   允许经对话纠正/补充（回流提案照常入池，由人工裁决落定）。
KNOWLEDGE_CLASSES = {
    "book": ("intake", "legacy"),
    "experience": ("manual", "dialogue"),
}


def knowledge_class(source: str) -> str:
    """知识来源 → 权限类别（'book' | 'experience'）。

    铁律：未知/非法来源一律从严判为书籍类（book）—— 分级权限的
    安全取向是「拿不准就禁止对话修改」，绝不给未知来源放行。
    """
    src = (source or "").strip()
    for cls, members in KNOWLEDGE_CLASSES.items():
        if src in members:
            return cls
    return "book"


def is_book_class(source: str) -> bool:
    """是否书籍类（禁止对话修改）；未知来源从严判 True。"""
    return knowledge_class(source) == "book"

_CONCEPT_DDL = (
    "CREATE NODE TABLE Concept("
    "id STRING, "
    "name STRING, "
    "category STRING, "
    "description STRING, "
    "knowledge_source STRING, "
    "PRIMARY KEY(id))"
)

_SYNAPSE_DDL = (
    # rel_id：用户自定义稳定地址（s_ 前缀 + uuid），Task 3.1.3 增补。
    # 理由：Kùzu REL 表内部 ID 在不同版本间取值/参数回传方式不稳定，
    # 以自定义 rel_id 寻址可保证边 CRUD、证据链引用跨版本可移植且人类可读。
    "CREATE REL TABLE SYNAPSE(FROM Concept TO Concept, "
    "rel_id STRING, "
    "weight DOUBLE, "
    "relation STRING, "
    "evidence STRING, "
    "status STRING, "
    "knowledge_source STRING)"
)


def _existing_tables(conn) -> dict:
    """
    查询当前图库已有表，返回 {表名: 类型}（类型为 NODE / REL）。

    兼容性说明：不同 Kùzu 版本 SHOW TABLES 的列序不同（有的 name 在前，
    有的 id 在前），因此按列名动态定位索引，避免硬编码列位置踩坑。
    """
    result = conn.execute("CALL show_tables() RETURN *")
    cols = list(result.get_column_names())
    name_i = cols.index("name") if "name" in cols else 0
    type_i = cols.index("type") if "type" in cols else 1

    tables: dict = {}
    while result.has_next():
        row = result.get_next()
        tables[row[name_i]] = row[type_i]
    return tables


def _table_columns(conn, table: str) -> list:
    """查指定表的列名列表（table_info 动态取列索引，兼容不同版本列序差异）。"""
    result = conn.execute(f"CALL table_info('{table}') RETURN *")
    cols = result.get_column_names()
    name_i = cols.index("name") if "name" in cols else 1
    out: list = []
    while result.has_next():
        out.append(result.get_next()[name_i])
    return out


def _migrate_add_knowledge_source(conn) -> None:
    """老库换芯：为 Concept/SYNAPSE 补 knowledge_source 列（升级D·前置）。

    Kùzu 0.11.3 不支持 ALTER TABLE ADD PROPERTY（实测 Parser 直接拒绝），
    采用已在同版本临时库演练通过的五步迁移：
      RENAME 两表为 *_legacy → 建新表（带 knowledge_source）→
      拷点 → 按 id 关联拷边 → DROP 旧表（先 REL 后 NODE）。

    存量数据统一记 knowledge_source='legacy'（从严按书籍类管理，
    人工可在面板手动改标记）。中断恢复：本迁移在单写者模型下瞬时完成，
    若极端情况中断遗留 *_legacy 表，由 /data 镜像 restore_on_boot 自愈
    拉回旧结构后重跑迁移（与项目既有自愈哲学一致）。

    Raises:
        RuntimeError: 检测到新旧表并存的含混中间态（拒绝猜测，交由镜像自愈）。
    """
    tables = _existing_tables(conn)
    has_legacy = "Concept_legacy" in tables or "SYNAPSE_legacy" in tables
    has_new = "Concept" in tables or "SYNAPSE" in tables
    if has_legacy and has_new:
        raise RuntimeError(
            "knowledge_source 迁移中间态（新旧表并存）：请从 /data 镜像恢复后重试"
        )

    if not has_legacy:
        # 正常路径：旧结构原位改名后换芯
        conn.execute("ALTER TABLE SYNAPSE RENAME TO SYNAPSE_legacy")
        conn.execute("ALTER TABLE Concept RENAME TO Concept_legacy")

    conn.execute(_CONCEPT_DDL)
    conn.execute(_SYNAPSE_DDL)
    conn.execute(
        "MATCH (n:Concept_legacy) "
        "CREATE (m:Concept {id: n.id, name: n.name, category: n.category, "
        "description: n.description, knowledge_source: 'legacy'})"
    )
    conn.execute(
        "MATCH (a:Concept_legacy)-[s:SYNAPSE_legacy]->(b:Concept_legacy), "
        "(na:Concept), (nb:Concept) "
        "WHERE na.id = a.id AND nb.id = b.id "
        "CREATE (na)-[:SYNAPSE {rel_id: s.rel_id, weight: s.weight, "
        "relation: s.relation, evidence: s.evidence, status: s.status, "
        "knowledge_source: 'legacy'}]->(nb)"
    )
    conn.execute("DROP TABLE SYNAPSE_legacy")   # 先 REL（端点依赖 NODE）
    conn.execute("DROP TABLE Concept_legacy")
    logger.info("Schema 迁移: knowledge_source 换芯完成（存量数据标记 legacy）")


def init_schema(conn) -> bool:
    """
    Task 3.1.1: Kùzu Schema 幂等初始化。

    建 Node 表 Concept(id, name, category, description) 与
    Rel 表 SYNAPSE(FROM Concept TO Concept, weight, relation, evidence, status)。

    幂等保证：先 SHOW TABLES 探测，已存在的表跳过建表 ——
    因此本函数可在每次 get_or_create 脑会话后放心调用，重复执行零副作用。
    建表顺序先 Concept 后 SYNAPSE（REL 表的端点依赖 NODE 表存在）。

    Args:
        conn: Kùzu Connection（通常来自 BrainSession.get_kuzu()）。

    Returns:
        True=Schema 就绪（新建或本就完整）。

    Raises:
        RuntimeError: Kùzu 建表失败（交由调用方决定是否降级/中止）。
    """
    try:
        tables = _existing_tables(conn)

        if "Concept" not in tables:
            conn.execute(_CONCEPT_DDL)
            logger.info("Kùzu Schema: 已创建 NODE 表 Concept")
        elif "knowledge_source" not in _table_columns(conn, "Concept"):
            # 老结构（升级D 前无 knowledge_source）→ 换芯迁移（一并重建 REL 表）
            logger.info("Kùzu Schema: 检测到旧结构，启动 knowledge_source 迁移")
            _migrate_add_knowledge_source(conn)
        else:
            logger.info("Kùzu Schema: NODE 表 Concept 已存在，跳过建表")

        if "SYNAPSE" not in tables:
            conn.execute(_SYNAPSE_DDL)
            logger.info("Kùzu Schema: 已创建 REL 表 SYNAPSE")
        elif "knowledge_source" not in _table_columns(conn, "SYNAPSE") and \
                "SYNAPSE_legacy" not in _existing_tables(conn):
            # 单独缺列（Concept 已带列而 SYNAPSE 不带的异常态）也走换芯兜底
            logger.info("Kùzu Schema: SYNAPSE 缺 knowledge_source，启动迁移")
            _migrate_add_knowledge_source(conn)
        else:
            logger.info("Kùzu Schema: REL 表 SYNAPSE 已存在，跳过建表")

        return True
    except Exception as exc:
        logger.error("Kùzu Schema 初始化失败: %s", exc)
        raise RuntimeError(f"Kùzu Schema 初始化失败: {exc}") from exc


def ensure_brain_schema(brain_id: str) -> bool:
    """
    便捷入口：取指定脑的 Kùzu 连接并确保 Schema 就绪。

    供多脑注册（Task 2.2.2 get_or_create）之后、任何图谱读写之前调用，
    保证「脑已注册 → 图表可用」这一前置不变式。
    """
    from core.brain_manager import get_or_create  # 延迟导入，避免环依赖

    conn = get_or_create(brain_id).get_kuzu()
    return init_schema(conn)


# ───────────────────────────────────────────────
# 概念节点 CRUD (Task 3.1.2)
# ───────────────────────────────────────────────

import re
import uuid
from typing import Optional

# 概念分类白名单之外的分类一律原样放行（领域无关设计，仅做长度约束）
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _new_concept_id() -> str:
    """生成全局唯一概念 id（c_ 前缀 + 32 位十六进制，杜绝自增 id 在多脑间的碰撞）。"""
    return f"c_{uuid.uuid4().hex}"


def _validate_id(concept_id: str) -> None:
    if not isinstance(concept_id, str) or not _ID_RE.match(concept_id):
        raise ValueError(f"非法概念 id: {concept_id!r}")


def add_concept(
    conn,
    name: str,
    category: str = "general",
    description: str = "",
    concept_id: Optional[str] = None,
    knowledge_source: str = DEFAULT_KNOWLEDGE_SOURCE,
) -> str:
    """
    创建概念节点，返回其唯一 id。

    铁律回顾：name 在同脑内允许重复（近义词不自动合并，主权归人类），
    因此本函数不做任何同名查重/拦截；若调用方（Web 面板）检测到相近词，
    由 similarity 弱提示引擎提示用户自行决断。

    Args:
        conn:        Kùzu Connection。
        name:        概念名称（非空，≤128 字）。
        category:    分类标签（≤64 字，默认 general）。
        description: 概念释义（≤2000 字）。
        concept_id:  显式指定 id（批量导入/恢复场景用）；缺省自动生成。
        knowledge_source: 知识来源标记（intake/manual/dialogue/legacy；
                升级D 分级权限依据：intake=书籍类禁对话修改，其余=经验类）。

    Returns:
        新建概念的 id。

    Raises:
        ValueError: name 为空 / 字段超长 / concept_id 非法。
        RuntimeError: Kùzu 写入失败。
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("概念 name 不得为空")
    name = name.strip()
    if len(name) > 128:
        raise ValueError(f"概念 name 超长（>{128} 字）")
    category = (category or "general").strip() or "general"
    if len(category) > 64:
        raise ValueError(f"category 超长（>{64} 字）")
    description = description or ""
    if len(description) > 2000:
        raise ValueError(f"description 超长（>{2000} 字）")
    if knowledge_source not in KNOWLEDGE_SOURCES:
        raise ValueError(
            f"非法 knowledge_source={knowledge_source!r}，仅允许 {KNOWLEDGE_SOURCES}"
        )

    cid = concept_id or _new_concept_id()
    _validate_id(cid)

    try:
        conn.execute(
            "CREATE (:Concept {id: $id, name: $name, category: $cat, "
            "description: $dsc, knowledge_source: $ks})",
            {"id": cid, "name": name, "cat": category, "dsc": description,
             "ks": knowledge_source},
        )
    except Exception as exc:
        # id 主键冲突（显式指定重复 id）也在此统一包装
        raise RuntimeError(f"概念节点写入失败 (id={cid}): {exc}") from exc

    logger.info("概念已创建: id=%s name=%r category=%s", cid, name, category)
    return cid


def get_concept(conn, concept_id: str) -> Optional[dict]:
    """按 id 精确读取单个概念；不存在返回 None。"""
    _validate_id(concept_id)
    result = conn.execute(
        "MATCH (n:Concept) WHERE n.id = $id "
        "RETURN n.id, n.name, n.category, n.description, n.knowledge_source",
        {"id": concept_id},
    )
    if not result.has_next():
        return None
    row = result.get_next()
    return {"id": row[0], "name": row[1], "category": row[2],
            "description": row[3], "knowledge_source": row[4]}


def update_concept(
    conn,
    concept_id: str,
    name: Optional[str] = None,
    category: Optional[str] = None,
    description: Optional[str] = None,
) -> bool:
    """
    按 id 更新概念字段（仅更新显式传入的字段，None 表示保持不变）。

    Returns:
        True=已更新；False=目标概念不存在。

    Raises:
        ValueError: 字段非法（空 name / 超长）。
    """
    _validate_id(concept_id)

    if name is not None:
        name = name.strip()
        if not name:
            raise ValueError("概念 name 不得为空")
        if len(name) > 128:
            raise ValueError(f"概念 name 超长（>{128} 字）")
    if category is not None:
        category = category.strip() or "general"
        if len(category) > 64:
            raise ValueError(f"category 超长（>{64} 字）")
    if description is not None and len(description) > 2000:
        raise ValueError(f"description 超长（>{2000} 字）")

    set_parts = []
    params = {"id": concept_id}
    if name is not None:
        set_parts.append("n.name = $name")
        params["name"] = name
    if category is not None:
        set_parts.append("n.category = $cat")
        params["cat"] = category
    if description is not None:
        set_parts.append("n.description = $dsc")
        params["dsc"] = description

    if not set_parts:
        return True  # 无字段可改，视为幂等成功

    try:
        conn.execute(
            f"MATCH (n:Concept) WHERE n.id = $id SET {', '.join(set_parts)}",
            params,
        )
    except Exception as exc:
        raise RuntimeError(f"概念更新失败 (id={concept_id}): {exc}") from exc

    # SET 对不存在的节点是空操作，需回查确认目标真实存在
    return get_concept(conn, concept_id) is not None


def delete_concept(conn, concept_id: str) -> bool:
    """
    按 id 删除概念节点及其关联突触（DETACH 语义：端点消失，边不可悬空）。

    Returns:
        True=已删除；False=目标概念不存在（幂等友好）。
    """
    _validate_id(concept_id)
    try:
        conn.execute(
            "MATCH (n:Concept) WHERE n.id = $id DETACH DELETE n",
            {"id": concept_id},
        )
    except Exception as exc:
        raise RuntimeError(f"概念删除失败 (id={concept_id}): {exc}") from exc
    return get_concept(conn, concept_id) is None


def list_concepts(
    conn,
    keyword: str = "",
    category: str = "",
    limit: int = 500,
) -> list[dict]:
    """
    列出概念（供前端面板与 similarity 引擎检索用）。

    Args:
        keyword:  name 子串模糊过滤（大小写不敏感，空串=不过滤）。
        category: 精确分类过滤（空串=不过滤）。
        limit:    返回上限（默认 500，防全量渲染卡顿，与局部视口加载原则对齐）。

    Returns:
        [{id, name, category, description, knowledge_source}, ...]，
        按 name 字典序排序。
    """
    cypher = "MATCH (n:Concept)"
    where = []
    params: dict = {"lim": max(1, min(int(limit), 2000))}

    if keyword:
        where.append("toLower(n.name) CONTAINS toLower($kw)")
        params["kw"] = keyword
    if category:
        where.append("n.category = $cat")
        params["cat"] = category
    if where:
        cypher += " WHERE " + " AND ".join(where)

    cypher += (" RETURN n.id, n.name, n.category, n.description, n.knowledge_source"
               " ORDER BY n.name LIMIT $lim")

    result = conn.execute(cypher, params)
    concepts = []
    while result.has_next():
        row = result.get_next()
        concepts.append({"id": row[0], "name": row[1], "category": row[2],
                         "description": row[3], "knowledge_source": row[4]})
    return concepts


# ───────────────────────────────────────────────
# 突触边 CRUD 与状态流转 (Task 3.1.3)
# ───────────────────────────────────────────────

# 状态机合法流转表：PENDING → CONFIRMED / REJECTED。
# CONFIRMED 与 REJECTED 均为终态：已确认知识不可被自动流程暗中改写，
# 误判的补救路径是「删除后重建」，主权始终在人。
_ALLOWED_TRANSITIONS = {
    "PENDING": {"CONFIRMED", "REJECTED"},
    "CONFIRMED": set(),
    "REJECTED": set(),
}


def _new_rel_id() -> str:
    """生成全局唯一突触地址（s_ 前缀 + 32 位十六进制）。"""
    return f"s_{uuid.uuid4().hex}"


def _validate_rel_id(rel_id: str) -> None:
    if not isinstance(rel_id, str) or not _ID_RE.match(rel_id):
        raise ValueError(f"非法突触 rel_id: {rel_id!r}")


def _validate_weight(weight: float) -> float:
    try:
        w = float(weight)
    except (TypeError, ValueError):
        raise ValueError(f"weight 必须为数值，收到 {weight!r}")
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"weight 必须落在 [0.0, 1.0]，收到 {w}")
    return w


def add_synapse(
    conn,
    src_id: str,
    dst_id: str,
    relation: str,
    weight: float = 0.5,
    evidence: str = "",
    status: str = "PENDING",
    rel_id: Optional[str] = None,
    knowledge_source: str = DEFAULT_KNOWLEDGE_SOURCE,
) -> str:
    """
    创建突触边（有向：src → dst），返回其稳定地址 rel_id。

    铁律：
      - 新建突触默认 status='PENDING'，未经人工确认绝不视为已学知识；
      - status 允许显式传入 CONFIRMED，但该入口仅供「手工面板录入即确认」
        与「云端恢复重放」两类人工场景，API 提案链路（Task 4.2.3）必须保持默认；
      - 两端概念必须真实存在，杜绝悬空边；
      - 平行边（同 src/dst/relation 多条）不拦截 —— 是否冗余由人审决断，
        后端不做任何自动去重/合并。

    Args:
        conn:     Kùzu Connection。
        src_id:   起点概念 id。
        dst_id:   终点概念 id。
        relation: 关系标签（非空，≤64 字，中文短词如 导致/抑制/提示/属于）。
        weight:   权重 [0.0, 1.0]，默认 0.5。
        evidence: 证据原文片段（≤2000 字），推演回溯与前端证据链打印复用。
        status:   初始状态，默认 PENDING。
        rel_id:   显式指定（恢复/导入场景），缺省自动生成。
        knowledge_source: 知识来源标记（intake/manual/dialogue/legacy；
            升级D 分级权限依据：dialogue 来源可经对话修改，intake/legacy
            按书籍类从严管理）。

    Returns:
        新建突触的 rel_id。

    Raises:
        ValueError:   参数非法 / 端点概念不存在。
        RuntimeError: Kùzu 写入失败。
    """
    _validate_id(src_id)
    _validate_id(dst_id)
    if get_concept(conn, src_id) is None:
        raise ValueError(f"起点概念不存在: {src_id}")
    if get_concept(conn, dst_id) is None:
        raise ValueError(f"终点概念不存在: {dst_id}")

    if not isinstance(relation, str) or not relation.strip():
        raise ValueError("relation 不得为空")
    relation = relation.strip()
    if len(relation) > 64:
        raise ValueError(f"relation 超长（>{64} 字）")

    w = _validate_weight(weight)
    evidence = evidence or ""
    if len(evidence) > 2000:
        raise ValueError(f"evidence 超长（>{2000} 字）")
    if status not in SYNAPSE_STATUSES:
        raise ValueError(f"非法初始 status={status!r}，仅允许 {SYNAPSE_STATUSES}")
    if knowledge_source not in KNOWLEDGE_SOURCES:
        raise ValueError(
            f"非法 knowledge_source={knowledge_source!r}，仅允许 {KNOWLEDGE_SOURCES}")

    rid = rel_id or _new_rel_id()
    _validate_rel_id(rid)

    try:
        conn.execute(
            "MATCH (a:Concept), (b:Concept) "
            "WHERE a.id = $src AND b.id = $dst "
            "CREATE (a)-[:SYNAPSE {rel_id: $rid, weight: $w, relation: $rel, "
            "evidence: $ev, status: $st, knowledge_source: $ks}]->(b)",
            {"src": src_id, "dst": dst_id, "rid": rid, "w": w,
             "rel": relation, "ev": evidence, "st": status,
             "ks": knowledge_source},
        )
    except Exception as exc:
        raise RuntimeError(f"突触写入失败 ({src_id}→{dst_id}): {exc}") from exc

    logger.info("突触已创建: rel_id=%s %s→%s rel=%s w=%.2f status=%s",
                rid, src_id, dst_id, relation, w, status)
    return rid


def get_confirmed_edge(conn, src_id: str, dst_id: str,
                       relation: str) -> Optional[dict]:
    """按有向端点对 + 关系词精确查既有 CONFIRMED 突触（升级D 审查用）。

    - 仅认 CONFIRMED：PENDING/REJECTED 不是已学知识，不构成修改对象；
    - 有向语义：src→dst 与 dst→src 是两条不同的知识，不互查反向；
    - relation 精确匹配（ relation 本就是自由标签，不做同义词归并——
      归并属概念工程，超出分级权限的审查职责）；
    - 同端点对同关系可能多条（历史多次确认），任取一条（分级判定只需
      类别，不关心具体权重）。

    Returns:
        {rel_id, knowledge_source, weight, evidence} 或 None。
    """
    result = conn.execute(
        "MATCH (a:Concept)-[s:SYNAPSE]->(b:Concept) "
        "WHERE a.id = $src AND b.id = $dst AND s.relation = $rel "
        "AND s.status = 'CONFIRMED' "
        "RETURN s.rel_id, s.knowledge_source, s.weight, s.evidence LIMIT 1",
        {"src": src_id, "dst": dst_id, "rel": relation},
    )
    if not result.has_next():
        return None
    row = result.get_next()
    return {"rel_id": row[0], "knowledge_source": row[1],
            "weight": row[2], "evidence": row[3]}


def get_synapse(conn, rel_id: str) -> Optional[dict]:
    """按 rel_id 精确读取单条突触；不存在返回 None。"""
    _validate_rel_id(rel_id)
    result = conn.execute(
        "MATCH (a:Concept)-[s:SYNAPSE]->(b:Concept) WHERE s.rel_id = $rid "
        "RETURN a.id, b.id, s.weight, s.relation, s.evidence, s.status, "
        "s.knowledge_source LIMIT 1",
        {"rid": rel_id},
    )
    if not result.has_next():
        return None
    row = result.get_next()
    return {
        "rel_id": rel_id,
        "src_id": row[0],
        "dst_id": row[1],
        "weight": row[2],
        "relation": row[3],
        "evidence": row[4],
        "status": row[5],
        "knowledge_source": row[6],
    }


def update_synapse(
    conn,
    rel_id: str,
    weight: Optional[float] = None,
    relation: Optional[str] = None,
    evidence: Optional[str] = None,
) -> bool:
    """
    按 rel_id 编辑突触的 weight / relation / evidence（None 表示保持不变）。

    注意：本函数刻意不提供 status 参数 —— 状态流转必须走 set_synapse_status
    的状态机闸口，禁止在普通编辑里顺手改状态。

    Returns:
        True=目标存在（含幂等无改动）；False=突触不存在。

    Raises:
        ValueError: 字段非法。
    """
    _validate_rel_id(rel_id)
    if get_synapse(conn, rel_id) is None:
        return False

    set_parts = []
    params: dict = {"rid": rel_id}
    if weight is not None:
        set_parts.append("s.weight = $w")
        params["w"] = _validate_weight(weight)
    if relation is not None:
        relation = relation.strip()
        if not relation:
            raise ValueError("relation 不得为空")
        if len(relation) > 64:
            raise ValueError(f"relation 超长（>{64} 字）")
        set_parts.append("s.relation = $rel")
        params["rel"] = relation
    if evidence is not None:
        if len(evidence) > 2000:
            raise ValueError(f"evidence 超长（>{2000} 字）")
        set_parts.append("s.evidence = $ev")
        params["ev"] = evidence

    if not set_parts:
        return True

    try:
        conn.execute(
            "MATCH (a:Concept)-[s:SYNAPSE]->(b:Concept) WHERE s.rel_id = $rid "
            f"SET {', '.join(set_parts)}",
            params,
        )
    except Exception as exc:
        raise RuntimeError(f"突触更新失败 (rel_id={rel_id}): {exc}") from exc

    logger.info("突触已更新: rel_id=%s 字段=%s", rel_id, list(params.keys()))
    return True


def set_synapse_status(conn, rel_id: str, status: str) -> bool:
    """
    状态机闸口：仅允许 PENDING → CONFIRMED / REJECTED。

    CONFIRMED 仅可由人工处置路径触发（confirm_synapse / Web 审核面板）；
    本函数本身不区分调用方身份，身份鉴权由上层路由（登录态 / API Token）负责，
    但任何自动流程（相似度提示、定时任务、LLM 提案）均不得调用本函数。

    Returns:
        True=流转成功或幂等（已是目标状态）；False=突触不存在。

    Raises:
        ValueError: 非法状态或非法流转（如 CONFIRMED → PENDING）。
    """
    _validate_rel_id(rel_id)
    current = get_synapse(conn, rel_id)
    if current is None:
        return False

    if status not in SYNAPSE_STATUSES:
        raise ValueError(f"非法目标 status={status!r}，仅允许 {SYNAPSE_STATUSES}")

    cur = current["status"]
    if cur == status:
        return True  # 幂等
    if status not in _ALLOWED_TRANSITIONS.get(cur, set()):
        raise ValueError(f"非法状态流转: {cur} → {status}（终态不可改，误判请删除重建）")

    try:
        conn.execute(
            "MATCH (a:Concept)-[s:SYNAPSE]->(b:Concept) WHERE s.rel_id = $rid "
            "SET s.status = $st",
            {"rid": rel_id, "st": status},
        )
    except Exception as exc:
        raise RuntimeError(f"突触状态流转失败 (rel_id={rel_id}): {exc}") from exc

    logger.info("突触状态流转: rel_id=%s %s → %s", rel_id, cur, status)
    return True


def list_synapses(
    conn,
    status: str = "",
    limit: int = 500,
) -> list[dict]:
    """
    列举突触（供 Web 审核面板与图视图取数）。

    Args:
        status: 按状态过滤（空串=全部；非法值抛 ValueError）。
        limit:  返回上限（默认 500，与局部视口加载原则对齐）。

    Returns:
        [{rel_id, src_id, src_name, dst_id, dst_name, relation,
          weight, evidence, status, knowledge_source}, ...]，
        按 rel_id 字典序稳定排序。升级D：附 knowledge_source 供前端
        渲染来源徽标（书籍/经验/手工/迁移），分级权限可视可查。
    """
    if status and status not in SYNAPSE_STATUSES:
        raise ValueError(f"非法 status 过滤值: {status!r}，仅允许 {SYNAPSE_STATUSES}")

    # Cypher 子句合法顺序：MATCH → WHERE → RETURN → ORDER BY → LIMIT。
    # Task 9.2 修复：原实现把 WHERE 拼在 RETURN 之后，Kùzu Parser 直接报错。
    cypher = "MATCH (a:Concept)-[s:SYNAPSE]->(b:Concept)"
    params: dict = {"lim": max(1, min(int(limit), 2000))}
    if status:
        cypher += " WHERE s.status = $st"
        params["st"] = status
    cypher += (
        " RETURN s.rel_id, a.id, a.name, b.id, b.name,"
        " s.relation, s.weight, s.evidence, s.status, s.knowledge_source"
        " ORDER BY s.rel_id LIMIT $lim"
    )

    result = conn.execute(cypher, params)
    rows = []
    while result.has_next():
        r = result.get_next()
        rows.append({
            "rel_id": r[0], "src_id": r[1], "src_name": r[2],
            "dst_id": r[3], "dst_name": r[4], "relation": r[5],
            "weight": float(r[6]), "evidence": r[7], "status": r[8],
            "knowledge_source": r[9],
        })
    return rows


def confirm_synapse(conn, rel_id: str) -> bool:
    """人工确认：PENDING → CONFIRMED（仅 Web 审核面板/手工接口可触达）。"""
    return set_synapse_status(conn, rel_id, "CONFIRMED")


def reject_synapse(conn, rel_id: str) -> bool:
    """人工否决：PENDING → REJECTED（被否决提案保留审计痕迹，不再参与推演）。"""
    return set_synapse_status(conn, rel_id, "REJECTED")


def delete_synapse(conn, rel_id: str) -> bool:
    """
    按 rel_id 删除突触边。

    Returns:
        True=已删除；False=突触不存在（幂等友好）。
    """
    _validate_rel_id(rel_id)
    try:
        conn.execute(
            "MATCH (a:Concept)-[s:SYNAPSE]->(b:Concept) WHERE s.rel_id = $rid DELETE s",
            {"rid": rel_id},
        )
    except Exception as exc:
        raise RuntimeError(f"突触删除失败 (rel_id={rel_id}): {exc}") from exc
    return get_synapse(conn, rel_id) is None


# ───────────────────────────────────────────────
# CONFIRMED 子图只读查询 (Task 3.1.4)
# ───────────────────────────────────────────────

def get_confirmed_subgraph(conn) -> dict:
    """
    拉取 CONFIRMED 突触诱导子图（只读，供推演引擎与前端渲染复用）。

    铁律：
      - 仅遍历 status='CONFIRMED' 的边；PENDING / REJECTED 一概不可见，
        从数据源头保证「严格依据已学知识」，杜绝未确认提案混入推演；
      - 节点集合 = 参与 ≥1 条 CONFIRMED 边的概念（无确认连线的孤立概念
        不进入推演子图，但其仍可经 list_concepts 供录入联想使用）；
      - 本函数纯只读、零写副作用，可被 activation / reasoner / Web API
        高频安全调用。

    Returns:
        {"nodes": [{id, name, category, description, knowledge_source}, ...],
         "edges": [{rel_id, src_id, dst_id, weight, relation, evidence,
                    knowledge_source}, ...]}
        节点按首次出现去重，边按 rel_id 稳定排序，
        便于前端增量 diff 与推演引擎的确定性遍历。
    """
    result = conn.execute(
        "MATCH (a:Concept)-[s:SYNAPSE]->(b:Concept) WHERE s.status = 'CONFIRMED' "
        "RETURN a.id, a.name, a.category, a.description, a.knowledge_source, "
        "b.id, b.name, b.category, b.description, b.knowledge_source, "
        "s.rel_id, s.weight, s.relation, s.evidence, s.knowledge_source "
        "ORDER BY s.rel_id"
    )
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    while result.has_next():
        row = result.get_next()
        src = {"id": row[0], "name": row[1], "category": row[2],
               "description": row[3], "knowledge_source": row[4]}
        dst = {"id": row[5], "name": row[6], "category": row[7],
               "description": row[8], "knowledge_source": row[9]}
        nodes.setdefault(src["id"], src)
        nodes.setdefault(dst["id"], dst)
        edges.append({
            "rel_id": row[10],
            "src_id": row[0],
            "dst_id": row[5],
            "weight": row[11],
            "relation": row[12],
            "evidence": row[13],
            "knowledge_source": row[14],
        })
    logger.debug("CONFIRMED 子图: %d 节点 / %d 边", len(nodes), len(edges))
    return {"nodes": list(nodes.values()), "edges": edges}


def get_brain_confirmed_subgraph(brain_id: str) -> dict:
    """
    便捷入口：按 brain_id 解析该脑连接后拉取 CONFIRMED 子图。

    供推演引擎（activation / reasoner）与前端看板按脑调用，
    内部复用多脑注册表单例连接，不产生新句柄。
    """
    from core.brain_manager import get_or_create  # 延迟导入，避免环依赖

    conn = get_or_create(brain_id).get_kuzu()
    return get_confirmed_subgraph(conn)


if __name__ == "__main__":
    # 自检：对临时测试脑执行 Schema 初始化并打印表清单
    import logging as _logging
    import os
    import tempfile

    _logging.basicConfig(level=logging.INFO)

    os.environ.setdefault("SYNAPSE_RUNTIME_DIR", tempfile.mkdtemp(prefix="synapse_schema_test_"))
    from core.brain_manager import get_or_create

    session = get_or_create("_schema_selftest")
    init_schema(session.get_kuzu())
    result = session.get_kuzu().execute("CALL show_tables() RETURN *")
    while result.has_next():
        print(result.get_next())

    # ── 升级D·前置：knowledge_source 端到端读写断言 ──
    conn = session.get_kuzu()
    ca = add_concept(conn, "ks_src_test", "测试源点", "症状",
                     knowledge_source="manual")
    cb = add_concept(conn, "ks_dst_test", "测试汇点", "证候")  # 缺省=intake
    assert get_concept(conn, ca)["knowledge_source"] == "manual"
    assert get_concept(conn, cb)["knowledge_source"] == "intake"

    rid = add_synapse(conn, ca, cb, "causes", 0.8, "自检证据",
                      status="CONFIRMED", knowledge_source="dialogue")
    # 三个读取口逐口核验来源标记贯通
    assert get_synapse(conn, rid)["knowledge_source"] == "dialogue"
    listed = {r["rel_id"]: r for r in list_synapses(conn)}
    assert listed[rid]["knowledge_source"] == "dialogue"
    sub = get_confirmed_subgraph(conn)
    edge = next(e for e in sub["edges"] if e["rel_id"] == rid)
    assert edge["knowledge_source"] == "dialogue"
    # 非法来源拦截（概念/突触双侧）
    for bad_call in (
        lambda: add_concept(conn, "ks_bad", "坏来源", "症状", knowledge_source="chat"),
        lambda: add_synapse(conn, ca, cb, "causes", knowledge_source="chat"),
    ):
        try:
            bad_call()
            raise AssertionError("非法 knowledge_source 应拦截")
        except ValueError as exc:
            print("非法来源拦截 ✓ ", exc)
    print("knowledge_source 端到端读写 ✓  concept/synapse/子图三口贯通")

    # ── 升级D：知识类别分级助手 ──
    # 映射正确性：intake/legacy→book，manual/dialogue→experience
    assert knowledge_class("intake") == "book"
    assert knowledge_class("legacy") == "book"
    assert knowledge_class("manual") == "experience"
    assert knowledge_class("dialogue") == "experience"
    # 未知/空值从严判书籍类（分级权限安全取向：拿不准就禁改）
    assert knowledge_class("") == "book"
    assert knowledge_class("bogus") == "book"
    assert knowledge_class(None) == "book"
    assert is_book_class("intake") and is_book_class("legacy")
    assert not is_book_class("dialogue") and not is_book_class("manual")
    assert is_book_class("unknown_xyz"), "未知来源必须从严判书籍类"
    print("知识类别分级 ✓  book={intake,legacy} / experience={manual,dialogue} / 未知从严")

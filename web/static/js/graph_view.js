/**
 * SynapseMind - 神经图谱局部视口渲染 (web/static/js/graph_view.js)
 *
 * Task 6.3.1: Cytoscape.js 局部视口加载。
 *
 * 局部视口铁律（与 Readme/Todo 对齐）：
 * - 本模块绝不默认全量渲染整张图谱；
 * - 默认仅渲染「中心节点 2-hop 邻域子图」（或由推演链路注入的激活子图，
 *   供 Task 6.3.2 脉冲动画复用）；
 * - 邻域节点数超过视口上限（VIEWPORT_MAX_NODES）时，拒绝渲染并提示「聚焦」，
 *   绝不降级为全量加载；
 * - 只渲染 status='CONFIRMED' 的突触（客户端二次过滤，与推演铁律一致：
 *   PENDING / REJECTED 提案不得出现在已学知识视图中）。
 *
 * 数据源（复用既有面板端点，零后端改动）：
 * - GET /api/concepts?limit=2000                    概念字典（搜索联想 + id→name 映射）
 * - GET /api/synapses?status=CONFIRMED&limit=2000   已确认突触（邻域计算原料）
 * 2-hop 邻域在客户端计算，仅把局部子图元素送入 Cytoscape 渲染。
 *
 * 对外接口（window.SynapseGraph）：
 * - mount(hostEl)               挂载画布（自动构建聚焦工具栏与图例）
 * - focus(idOrName)             以指定概念为中心拉取并渲染 2-hop 邻域
 * - renderActivation(subgraph)  渲染推演激活子图（Task 6.3.2 复用入口）
 * - refresh()                   重新拉取数据并按当前中心重渲染
 * 同时暴露 window.refreshGraphView = refresh，供 index.html 审核流
 * （Task 6.2.2 批量确认后）联动刷新，并监听 "synapses:changed" 事件。
 *
 * 集成（宿主页面仅需：/static 挂载于 Task 7.1）：
 *   <div id="graphCanvas"></div>
 *   <script src="/static/js/graph_view.js" defer></script>
 */

(function () {
  "use strict";

  // ── 局部视口常量 ──
  const HOPS = 2;                  // 默认邻域深度：中心节点 2-hop
  const VIEWPORT_MAX_NODES = 120;  // 视口节点上限：超限提示聚焦，绝不全量加载
  const FETCH_LIMIT = 2000;        // 端点拉取上限（core 侧同样钳制）
  const CYTO_CDN = "https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js";

  // ── 模块状态 ──
  let cy = null;               // Cytoscape 实例（惰性创建）
  let concepts = [];           // [{id,name,category,description}]
  let conceptById = new Map();
  let confirmedEdges = [];     // 仅 CONFIRMED（双保险过滤后）
  let truncated = false;       // 确认边是否触达拉取上限（截断子集提示）
  let centerId = null;         // 当前聚焦中心概念 id
  let ui = null;               // 工具栏/画布/状态行引用
  let cytoPromise = null;      // CDN 脚本加载 Promise

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

  async function api(path) {
    const resp = await fetch(path);
    let data = {};
    try { data = await resp.json(); } catch (_) { /* 非 JSON（如被 302 到登录页） */ }
    if (!resp.ok) {
      const d = data.detail;
      throw new Error(Array.isArray(d)
        ? d.map((e) => e.msg).join("；")
        : (d || `HTTP ${resp.status}`));
    }
    return data;
  }

  function setStatus(text, cls) {
    if (!ui) return;
    ui.status.textContent = text;   // textContent 防注入
    ui.status.className = "gstatus " + (cls || "");
  }

  // ── 模块私有样式（自包含，宿主页面零侵入）──
  const CSS = `
    .gvHost { width: 100%; }
    .gvHost .gbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
    .gvHost .gFocus {
      flex: 1 1 280px; padding: 8px 10px; font-size: 13px; color: #eef2ff;
      background: #0d1425; border: 1px solid #2a3a5e; border-radius: 8px; outline: none;
    }
    .gvHost .gFocus:focus { border-color: #6f9bff; }
    .gvHost .gBtn {
      padding: 8px 14px; font-size: 13px; font-weight: 600; color: #fff;
      background: linear-gradient(135deg, #3b6cf0, #6f9bff);
      border: none; border-radius: 8px; cursor: pointer;
    }
    .gvHost .gBtn:hover { filter: brightness(1.12); }
    .gvHost .gBtn:disabled { opacity: .45; cursor: not-allowed; }
    .gvHost .gstatus { font-size: 12px; color: #8fa3c8; }
    .gvHost .gstatus.ok { color: #7fe3a0; }
    .gvHost .gstatus.err { color: #ff9b9b; }
    .gvHost .gstatus.warn { color: #ffd28a; }
    .gvHost .gcanvas {
      height: 460px; border: 1px solid #22304f; border-radius: 10px;
      background: #0a101f; overflow: hidden;
    }
    .gvHost .glegend { font-size: 11px; color: #5d6782; margin-top: 8px; }
  `;

  function injectCss() {
    if (document.getElementById("gvCss")) return;
    const st = document.createElement("style");
    st.id = "gvCss";
    st.textContent = CSS;
    document.head.appendChild(st);
  }

  // ── Cytoscape.js CDN 惰性加载（离线环境给出可读指引）──
  function loadCytoscape() {
    if (window.cytoscape) return Promise.resolve();
    if (!cytoPromise) {
      cytoPromise = new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = CYTO_CDN;
        s.onload = () => (window.cytoscape
          ? resolve()
          : reject(new Error("Cytoscape.js 加载异常")));
        s.onerror = () => {
          cytoPromise = null;
          reject(new Error("Cytoscape.js CDN 加载失败（离线部署请将 dist/cytoscape.min.js 放至 /static 并调整 CYTO_CDN）"));
        };
        document.head.appendChild(s);
      });
    }
    return cytoPromise;
  }

  // ── 数据拉取 ──
  async function fetchConcepts() {
    const d = await api(`/api/concepts?limit=${FETCH_LIMIT}`);
    if (!Array.isArray(d.items)) throw new Error("概念接口返回异常（未登录或服务不可用）");
    concepts = d.items;
    conceptById = new Map(concepts.map((n) => [n.id, n]));
  }

  async function fetchEdges() {
    const d = await api(`/api/synapses?status=CONFIRMED&limit=${FETCH_LIMIT}`);
    if (!Array.isArray(d.items)) throw new Error("突触接口返回异常（未登录或服务不可用）");
    confirmedEdges = d.items.filter((e) => e.status === "CONFIRMED");  // 双保险：仅已确认边
    truncated = d.items.length >= FETCH_LIMIT;
  }

  // ── 2-hop 无向邻域（渲染视口用；边方向保留在元素数据中）──
  function neighborhood(center) {
    const adj = new Map();
    for (const e of confirmedEdges) {
      if (!adj.has(e.src_id)) adj.set(e.src_id, new Set());
      if (!adj.has(e.dst_id)) adj.set(e.dst_id, new Set());
      adj.get(e.src_id).add(e.dst_id);
      adj.get(e.dst_id).add(e.src_id);
    }
    const seen = new Set([center]);
    let frontier = [center];
    for (let h = 0; h < HOPS; h++) {
      const next = [];
      for (const id of frontier) {
        for (const nb of adj.get(id) || []) {
          if (!seen.has(nb)) { seen.add(nb); next.push(nb); }
        }
      }
      frontier = next;
    }
    return seen;
  }

  // ── Cytoscape 元素构建（仅局部子图进入渲染管线）──
  function catColor(cat) {
    let h = 0;
    for (const ch of String(cat || "")) h = (h * 31 + ch.codePointAt(0)) % 360;
    return `hsl(${h}, 52%, 60%)`;
  }

  function buildElements(nodeIds, opts = {}) {
    const ids = nodeIds instanceof Set ? nodeIds : new Set(nodeIds);
    const edges = confirmedEdges.filter((e) => ids.has(e.src_id) && ids.has(e.dst_id));
    const deg = new Map();
    for (const e of edges) {
      deg.set(e.src_id, (deg.get(e.src_id) || 0) + 1);
      deg.set(e.dst_id, (deg.get(e.dst_id) || 0) + 1);
    }
    const els = [];
    for (const id of ids) {
      const c = conceptById.get(id) || { id, name: id, category: "general" };
      els.push({
        group: "nodes",
        data: {
          id,
          label: c.name,
          category: c.category,
          color: catColor(c.category),
          size: 22 + Math.min(deg.get(id) || 0, 12) * 3,
        },
        classes: opts.activation ? "activation" : (id === centerId ? "center" : ""),
      });
    }
    for (const e of edges) {
      els.push({
        group: "edges",
        data: {
          id: e.rel_id,
          source: e.src_id,
          target: e.dst_id,
          label: e.relation,
          weight: e.weight,
          evidence: e.evidence,
          width: 1 + Number(e.weight || 0) * 5,
        },
      });
    }
    return els;
  }

  const CY_STYLE = [
    {
      selector: "node",
      style: {
        "background-color": "data(color)",
        "border-width": 1.5,
        "border-color": "rgba(111,155,255,.5)",
        label: "data(label)",
        color: "#dbe4f7",
        "font-size": 11,
        "text-valign": "bottom",
        "text-margin-y": 6,
        "text-background-color": "#0a101f",
        "text-background-opacity": 0.75,
        "text-background-padding": 2,
        width: "data(size)",
        height: "data(size)",
      },
    },
    {
      selector: "node.center",
      style: { "border-width": 3.5, "border-color": "#ffd28a" },
    },
    {
      selector: "node.activation",
      style: { "border-width": 3, "border-color": "#7fe3a0" },
    },
    {
      selector: "edge",
      style: {
        width: "data(width)",
        "line-color": "rgba(122,160,255,.4)",
        "target-arrow-shape": "triangle",
        "target-arrow-color": "rgba(122,160,255,.4)",
        "curve-style": "bezier",
        label: "data(label)",
        "font-size": 9,
        color: "#7f8db0",
        "text-background-color": "#0a101f",
        "text-background-opacity": 0.8,
        "text-background-padding": 2,
      },
    },
    {
      // 神经脉冲态（Task 6.3.2）：按扩散时序被点亮的传导边
      selector: "edge.pulse",
      style: {
        width: 6,
        "line-color": "#ffd28a",
        "target-arrow-color": "#ffd28a",
        "z-index": 99,
      },
    },
    {
      // 种子 / 当前脉冲层节点的辉光态
      selector: "node.pulse",
      style: {
        "border-width": 4,
        "border-color": "#ffd28a",
        "overlay-color": "rgba(255, 210, 138, .35)",
        "overlay-opacity": 0.45,
        "overlay-padding": 6,
      },
    },
  ];

  function ensureCy() {
    if (!cy) {
      cy = window.cytoscape({
        container: ui.canvas,
        style: CY_STYLE,
        wheelSensitivity: 0.25,
      });
      // 点击节点 → 以其为新中心重新聚焦；点击边 → 状态行展示证据片段
      cy.on("tap", "node", (ev) => {
        focus(ev.target.id()).catch((e) => setStatus("✗ " + e.message, "err"));
      });
      cy.on("tap", "edge", (ev) => {
        const d = ev.target.data();
        setStatus(`边 ${d.label} · w=${Number(d.weight).toFixed(2)} · 证据：${d.evidence || "（无）"}`);
      });
    }
    return cy;
  }

  function render(els) {
    const g = ensureCy();
    g.elements().remove();
    if (els.length) g.add(els);
    g.layout({
      name: "cose", animate: true, animationDuration: 260,
      randomize: true, padding: 30, fit: true,
    }).run();
  }

  // ── 聚焦渲染：中心概念 2-hop 局部视口 ──
  async function focus(ref) {
    if (!ui) throw new Error("画布尚未挂载");
    if (!conceptById.size) await fetchConcepts();
    await fetchEdges();

    const target = conceptById.get(ref) || concepts.find((c) => c.name === ref);
    if (!target) {
      setStatus(`✗ 未找到概念：${ref}（请先在上方面板建点）`, "err");
      return;
    }
    await loadCytoscape();

    centerId = target.id;
    const ids = neighborhood(centerId);

    // 局部视口铁律：超限提示聚焦，绝不降级为全量加载
    if (ids.size > VIEWPORT_MAX_NODES) {
      ensureCy().elements().remove();
      setStatus(`⚠ 「${target.name}」的 ${HOPS}-hop 邻域含 ${ids.size} 个节点，超出视口上限 ${VIEWPORT_MAX_NODES}。请改选更具体的中心概念聚焦；本视图不提供全量加载。`, "warn");
      return;
    }

    const els = buildElements(ids);
    render(els);
    const nE = els.filter((x) => x.group === "edges").length;
    const note = truncated ? "（确认边已达拉取上限，当前为截断子集）" : "";
    if (ids.size <= 1) {
      setStatus(`中心「${target.name}」暂无 CONFIRMED 连线（孤立概念）${note}`, "warn");
    } else {
      setStatus(`✓ 已渲染「${target.name}」${HOPS}-hop 邻域：${ids.size} 节点 / ${nE} 条已确认边${note}（点击节点可重新聚焦）`, "ok");
    }
  }

  // ── 激活子图渲染入口（Task 6.3.2 推演链路复用）──
  async function renderActivation(subgraph) {
    if (!ui) throw new Error("画布尚未挂载");
    await loadCytoscape();
    await fetchEdges();
    if (!conceptById.size) await fetchConcepts();

    const ids = subgraph.node_ids instanceof Set
      ? subgraph.node_ids
      : new Set(subgraph.node_ids || []);
    if (!ids.size) { setStatus("激活子图为空（无高能节点）", "warn"); return; }
    if (ids.size > VIEWPORT_MAX_NODES) {
      setStatus(`⚠ 激活子图含 ${ids.size} 个节点，超出视口上限 ${VIEWPORT_MAX_NODES}，请提高能量阈值聚焦。`, "warn");
      return;
    }
    centerId = null;
    render(buildElements(ids, { activation: true }));
    setStatus(`✓ 已渲染${subgraph.title || "推演激活"}子图：${ids.size} 节点（绿描边 = 激活序列）`, "ok");
  }

  // ── 神经脉冲动画（Task 6.3.2）：按能量扩散时序点亮传导路径 ──
  // activations: /api/reason 返回的激活序列，每项 {id, hop, chain}；
  // 调度规则：hop=0 的种子节点立即辉光，此后逐跳点亮传导边与到达节点，
  // 复现能量波前推进的真实时序；边的高亮依据是激活链中的真实 rel_id，
  // 绝不对图中无关边做任何装饰性点亮（证据可视化零编造铁律）。
  let pulseTimers = [];

  function clearPulse() {
    pulseTimers.forEach(clearTimeout);
    pulseTimers = [];
    if (cy) cy.elements().removeClass("pulse");
  }

  function playPulse(activations, opts = {}) {
    if (!cy) return Promise.reject(new Error("画布尚未挂载"));
    clearPulse();

    const hopOf = new Map();          // concept_id → 最早激活跳数
    const edgeSeq = [];               // [{rel_id, hop}] 按传导时序去重
    const seenEdges = new Set();
    for (const a of activations || []) {
      const hop = Number.isFinite(a.hop) ? a.hop : 0;
      if (!hopOf.has(a.id) || hopOf.get(a.id) > hop) hopOf.set(a.id, hop);
      for (const step of a.chain || []) {
        if (step.rel_id && !seenEdges.has(step.rel_id)) {
          seenEdges.add(step.rel_id);
          edgeSeq.push({ rel_id: step.rel_id, hop });
        }
      }
    }
    if (!hopOf.size) return Promise.resolve();

    let maxHop = 0;
    for (const h of hopOf.values()) maxHop = Math.max(maxHop, h);
    const LAYER_MS = opts.layerMs || 520;   // 每一跳的点亮节奏
    const byId = (id) => cy.getElementById(id);

    // 种子层立即辉光
    for (const [id, hop] of hopOf) {
      if (hop === 0) {
        pulseTimers.push(setTimeout(() => byId(id).addClass("pulse"), 60));
      }
    }
    // 逐跳点亮传导边（波前推进），随后点亮到达节点
    for (const e of edgeSeq) {
      pulseTimers.push(setTimeout(() => {
        byId(e.rel_id).addClass("pulse");
      }, 120 + e.hop * LAYER_MS));
    }
    for (const [id, hop] of hopOf) {
      if (hop > 0) {
        pulseTimers.push(setTimeout(() => byId(id).addClass("pulse"),
                                   120 + hop * LAYER_MS + 160));
      }
    }

    return new Promise((resolve) => {
      pulseTimers.push(setTimeout(resolve, 120 + (maxHop + 1) * LAYER_MS));
    });
  }

  // ── 审核联动刷新（Task 6.2.2 批量确认后调用 / synapses:changed 事件）──
  async function refresh() {
    if (!ui) return;
    if (centerId) {
      await focus(centerId);   // 按当前中心重拉数据重渲染
    } else {
      await fetchConcepts();
      if (ui.dl) {
        ui.dl.innerHTML = concepts.map(
          (c) => `<option value="${esc(c.name)}">${esc(c.category)}</option>`
        ).join("");
      }
    }
  }

  window.refreshGraphView = refresh;
  window.addEventListener("synapses:changed", () => {
    refresh().catch(() => { /* 联动失败不打断主流程 */ });
  });

  // ── 挂载：构建工具栏 + 画布 + 图例 ──
  function mount(host) {
    if (!host || ui) return;
    injectCss();
    host.classList.add("gvHost");
    host.innerHTML = `
      <div class="gbar">
        <input class="gFocus" type="text" list="gvDl" maxlength="64"
               placeholder="输入中心概念聚焦渲染（默认 ${HOPS}-hop 局部视口，不自动全量加载）">
        <datalist id="gvDl"></datalist>
        <button class="gBtn" type="button">聚焦渲染</button>
        <span class="gstatus">局部视口模式：输入中心概念后点击「聚焦渲染」。</span>
      </div>
      <div class="gcanvas"></div>
      <div class="glegend">● 节点颜色 = 分类 · 大小 = 局部连接度 · 边宽 = 传导权重 · 点击节点重新聚焦 · 点击边查看证据（仅渲染 CONFIRMED 已学知识）</div>
    `;
    ui = {
      input: host.querySelector(".gFocus"),
      btn: host.querySelector(".gBtn"),
      status: host.querySelector(".gstatus"),
      canvas: host.querySelector(".gcanvas"),
      dl: host.querySelector("#gvDl"),
    };

    const doFocus = async () => {
      const ref = ui.input.value.trim();
      if (!ref) { setStatus("请先输入中心概念名或 id", "warn"); return; }
      ui.btn.disabled = true;
      try { await focus(ref); }
      catch (e) { setStatus("✗ " + e.message, "err"); }
      finally { ui.btn.disabled = false; }
    };
    ui.btn.addEventListener("click", doFocus);
    ui.input.addEventListener("keydown", (ev) => { if (ev.key === "Enter") doFocus(); });

    fetchConcepts().then(() => {
      ui.dl.innerHTML = concepts.map(
        (c) => `<option value="${esc(c.name)}">${esc(c.category)}</option>`
      ).join("");
    }).catch((e) => setStatus("✗ 概念字典加载失败：" + e.message, "err"));
  }

  // ── 对外接口 ──
  window.SynapseGraph = {
    mount,
    focus,
    renderActivation,
    playPulse,
    clearPulse,
    refresh,
    HOPS,
    VIEWPORT_MAX_NODES,
  };

  // ── 自动初始化：宿主页面含 #graphCanvas 即挂载（Task 7.1 静态挂载后生效）──
  function autoInit() {
    const host = document.getElementById("graphCanvas");
    if (host) mount(host);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", autoInit);
  } else {
    autoInit();
  }
})();

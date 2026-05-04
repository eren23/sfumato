/* Sfumato showcase — vanilla JS, no framework, no build step.
 *
 * Pages: browse, detail, about. URL hash drives navigation:
 *   #browse                 — list view
 *   #problem/<source>/<idx> — detail view
 *   #about                  — explainer
 */

const PAGE_SIZE = 20;
let DATA = null;
let FILTERED = [];
let CURRENT_PAGE = 0;
let ACTIVE_TAGS = new Set();
let ACTIVE_SOURCE = "";
let ACTIVE_CORRECT = "";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

const TAG_CLASSES = {
  "near_tie_wrong": "warn",
  "unanimous_wrong": "warn",
  "clear_majority_wrong": "warn",
  "no_extractable_answer": "warn",
  "commit_lora_repair": "ok",
  "redundancy_save": "ok",
};
const tagClass = (t) => TAG_CLASSES[t] || "";

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  const resp = await fetch("examples.json");
  DATA = await resp.json();

  $("#generated-at").textContent = DATA.generated_at;

  renderStats();
  renderTagPills();
  renderSourceFilter();

  $("#search").addEventListener("input", () => {
    CURRENT_PAGE = 0;
    applyFilters();
  });
  $("#correct-filter").addEventListener("change", (e) => {
    ACTIVE_CORRECT = e.target.value;
    CURRENT_PAGE = 0;
    applyFilters();
  });
  $("#source-filter").addEventListener("change", (e) => {
    ACTIVE_SOURCE = e.target.value;
    CURRENT_PAGE = 0;
    applyFilters();
  });
  $("#random-btn").addEventListener("click", () => {
    if (FILTERED.length === 0) return;
    const r = FILTERED[Math.floor(Math.random() * FILTERED.length)];
    location.hash = `problem/${r.source}/${r.idx}`;
  });
  $("#prev-page").addEventListener("click", () => {
    if (CURRENT_PAGE > 0) { CURRENT_PAGE--; renderList(); }
  });
  $("#next-page").addEventListener("click", () => {
    const nPages = Math.max(1, Math.ceil(FILTERED.length / PAGE_SIZE));
    if (CURRENT_PAGE < nPages - 1) { CURRENT_PAGE++; renderList(); }
  });
  $("#back-btn").addEventListener("click", () => { location.hash = "browse"; });

  $$(".nav-link[data-target]").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      location.hash = a.dataset.target;
    });
  });

  window.addEventListener("hashchange", route);
  applyFilters();
  route();
}

function renderStats() {
  const wrap = $("#stats");
  clear(wrap);
  const stat = (...children) => el("span", { class: "stat" }, ...children);
  const num = (s) => el("span", { class: "num" }, String(s));
  for (const [src, s] of Object.entries(DATA.stats.by_source)) {
    wrap.appendChild(stat(
      num(s.n),
      " records · accuracy ",
      num((s.accuracy * 100).toFixed(1) + "%"),
      " · ",
      el("code", {}, src),
    ));
  }
  wrap.appendChild(stat(num(DATA.n_records), " total"));
}

function renderTagPills() {
  const wrap = $("#tag-pills");
  clear(wrap);
  const tags = Object.entries(DATA.stats.by_tag).sort((a, b) => b[1] - a[1]);
  for (const [tag, count] of tags) {
    const cls = `tag-pill ${tagClass(tag)}`;
    const btn = el("button", {
      class: cls,
      type: "button",
      onclick: () => {
        if (ACTIVE_TAGS.has(tag)) ACTIVE_TAGS.delete(tag);
        else ACTIVE_TAGS.add(tag);
        renderTagPills();
        CURRENT_PAGE = 0;
        applyFilters();
      },
    }, `${tag} (${count})`);
    if (ACTIVE_TAGS.has(tag)) btn.classList.add("active");
    wrap.appendChild(btn);
  }
}

function renderSourceFilter() {
  const sel = $("#source-filter");
  for (const src of Object.keys(DATA.stats.by_source)) {
    sel.appendChild(el("option", { value: src }, src));
  }
}

// ── Filtering ──────────────────────────────────────────────────────────────
function applyFilters() {
  const q = $("#search").value.trim().toLowerCase();
  FILTERED = DATA.records.filter((r) => {
    if (q && !r.question.toLowerCase().includes(q)) return false;
    if (ACTIVE_SOURCE && r.source !== ACTIVE_SOURCE) return false;
    if (ACTIVE_CORRECT === "true" && !r.correct) return false;
    if (ACTIVE_CORRECT === "false" && r.correct) return false;
    if (ACTIVE_TAGS.size > 0) {
      for (const t of ACTIVE_TAGS) if (!r.tags.includes(t)) return false;
    }
    return true;
  });
  $("#results-count").textContent = `${FILTERED.length} match${FILTERED.length === 1 ? "" : "es"}`;
  renderList();
}

function renderList() {
  const wrap = $("#problem-list");
  clear(wrap);
  const start = CURRENT_PAGE * PAGE_SIZE;
  const slice = FILTERED.slice(start, start + PAGE_SIZE);
  for (const r of slice) {
    wrap.appendChild(renderCard(r));
  }
  const nPages = Math.max(1, Math.ceil(FILTERED.length / PAGE_SIZE));
  $("#page-info").textContent = `page ${CURRENT_PAGE + 1} / ${nPages}`;
  $("#prev-page").disabled = CURRENT_PAGE === 0;
  $("#next-page").disabled = CURRENT_PAGE >= nPages - 1;
}

function renderCard(r) {
  const head = el("div", { class: "problem-head" },
    el("span", { class: "problem-id" }, `#${r.idx}`),
    el("span", { class: `problem-condition ${r.condition}` }, r.condition),
    el("span", { class: `verdict ${r.correct ? "correct" : "wrong"}` },
      `${r.correct ? "✓" : "✗"} pred=${r.pred || "—"}, gold=${r.gold}`),
  );
  const q = el("div", { class: "problem-q" }, r.question);
  const meta = el("div", { class: "problem-meta" },
    el("span", {}, `seed ${r.seed}`),
    el("span", {}, `votes: ${r.votes_str || "(none)"}`),
  );
  const tags = el("div", { class: "problem-tags" });
  for (const t of r.tags) {
    tags.appendChild(el("span", { class: `tag-pill ${tagClass(t)}` }, t));
  }
  return el("li", {
    class: "problem-card",
    onclick: () => { location.hash = `problem/${r.source}/${r.idx}`; },
  }, head, q, meta, tags);
}

// ── Detail ─────────────────────────────────────────────────────────────────
function renderDetail(source, idxStr) {
  const idx = parseInt(idxStr, 10);
  const r = DATA.records.find((x) => x.source === source && x.idx === idx);
  const wrap = $("#detail-content");
  clear(wrap);
  if (!r) {
    wrap.appendChild(el("p", {}, `Problem #${idx} not found in source "${source}".`));
    return;
  }

  // Title row
  wrap.appendChild(el("div", { class: "problem-head" },
    el("span", { class: "problem-id" }, `#${r.idx} · seed ${r.seed} · k=${r.k_steps}`),
    el("span", { class: `problem-condition ${r.condition}` }, r.condition),
    el("span", { class: `verdict ${r.correct ? "correct" : "wrong"}` },
      r.correct ? "✓ correct" : "✗ wrong"),
  ));

  // Question
  wrap.appendChild(el("div", { class: "detail-q" }, r.question));

  // Tags
  const tagsDiv = el("div", { class: "problem-tags-detail" });
  for (const t of r.tags) {
    tagsDiv.appendChild(el("span", { class: `tag-pill ${tagClass(t)}` }, t));
  }
  wrap.appendChild(tagsDiv);

  // Summary cards
  const summary = el("div", { class: "detail-summary" });
  const pair = (label, value, cls = "") =>
    el("div", { class: "pair" }, label, el("span", { class: `v ${cls}` }, value));
  summary.appendChild(pair("Predicted", r.pred || "—", r.correct ? "correct" : "wrong"));
  summary.appendChild(pair("Gold", r.gold || "—"));
  summary.appendChild(pair("Voting", r.winner ? `winner: ${r.winner}` : "(no votes)"));
  if (r.flops) {
    const tflops = (r.flops / 1e12).toFixed(1);
    summary.appendChild(pair("FLOPs (LLaDA)", `${tflops} T`));
  }
  if (r.wallclock_ms != null) {
    summary.appendChild(pair("Wallclock", `${(r.wallclock_ms / 1000).toFixed(2)} s`));
  }
  if (r.esc_trigger_block != null) {
    summary.appendChild(pair("ESC fired", `block ${r.esc_trigger_block} (pruned ${r.esc_branches_pruned})`, "ok"));
  }
  wrap.appendChild(summary);

  // Speed panel (showcase v1) — only renders when a post-spike paired
  // wallclock is present on the record (build_examples.py SPEED_PAIRS).
  if (r.wallclock_ms != null && r.wallclock_ms_post != null) {
    wrap.appendChild(renderSpeedPanel(r));
  }

  // Vote bar (gold-aware)
  const votesArr = (r.votes_str || "").split("|").map((s) => s.trim());
  const tally = el("div", { class: "vote-tally" },
    el("h3", {}, `5-branch vote tally`),
  );
  const bar = el("div", { class: "vote-bar" });
  for (const v of votesArr) {
    let cls = "vote-cell";
    if (!v) cls += " empty";
    else if (v === r.gold) cls += " correct";
    else cls += " wrong";
    bar.appendChild(el("div", { class: cls }, v || "·"));
  }
  tally.appendChild(bar);
  wrap.appendChild(tally);

  // Branches
  const branches = el("div", { class: "branches" });
  for (let i = 0; i < r.branches.length; i++) {
    const text = r.branches[i] || "";
    const voted = votesArr[i] || "";
    const matchesGold = voted === r.gold;
    const isWinner = voted === r.winner && r.winner !== "";
    const card = el("div", {
      class: `branch-card${isWinner ? " winner" : ""} collapsed`,
    });
    const toggle = el("span", { class: "branch-toggle" }, "expand ▾");
    toggle.addEventListener("click", () => {
      card.classList.toggle("collapsed");
      toggle.textContent = card.classList.contains("collapsed")
        ? "expand ▾" : "collapse ▴";
    });
    const head = el("div", { class: "branch-head" },
      el("span", { class: "branch-num" }, `Branch ${i}${isWinner ? " · winner" : ""}`),
      el("span", {
        class: `branch-vote ${voted ? (matchesGold ? "match-gold" : "no-match") : ""}`,
      }, `vote: ${voted || "—"}`),
      toggle,
    );
    const body = el("div", { class: "branch-body" }, text || "(empty)");
    card.appendChild(head);
    card.appendChild(body);
    branches.appendChild(card);
  }
  wrap.appendChild(el("h3", {}, "Reasoning traces"));
  wrap.appendChild(branches);
}

// ── Speed panel (showcase v1) ─────────────────────────────────────────────
function renderSpeedPanel(r) {
  const pre = r.wallclock_ms;
  const post = r.wallclock_ms_post;
  const speedup = r.speedup;
  // Bar dimensions: pre is always the longer (slower) one; scale to 100%.
  // If post >= pre (regression), still render but flip color.
  const denom = Math.max(pre, post, 1);
  const prePct = (pre / denom) * 100;
  const postPct = (post / denom) * 100;

  const wrap = el("div", { class: "speed-panel" });
  wrap.appendChild(el("h3", {}, "Speed comparison (pre vs post-spike)"));

  const bars = el("div", { class: "speed-bars" });
  const row = (label, ms, pct, klass) => el("div", { class: `speed-row ${klass}` },
    el("span", { class: "speed-label" }, label),
    el("div", { class: "speed-track" },
      el("div", {
        class: "speed-fill",
        style: `width:${pct}%`,
      }),
    ),
    el("span", { class: "speed-time" }, `${(ms / 1000).toFixed(2)} s`),
  );
  bars.appendChild(row("pre-spike (sequential)", pre, prePct, "pre"));
  bars.appendChild(row("post-spike (S0 + S1 batched)", post, postPct, "post"));
  wrap.appendChild(bars);

  if (speedup != null) {
    const cls = speedup >= 1.0 ? "ok" : "warn";
    wrap.appendChild(el("div", { class: `speedup-badge ${cls}` },
      `${speedup.toFixed(2)}× ${speedup >= 1.0 ? "faster" : "slower"}`,
    ));
  }
  if (r.pred_post != null && r.pred_post !== r.pred) {
    wrap.appendChild(el("p", { class: "speed-note" },
      `Note: post-spike prediction "${r.pred_post}" differs from pre-spike "${r.pred}". This is expected when seeds map differently across implementations.`,
    ));
  }
  return wrap;
}

// ── Routing ────────────────────────────────────────────────────────────────
function route() {
  const h = location.hash.slice(1) || "browse";
  $$(".page").forEach((p) => p.classList.remove("active"));
  $$(".nav-link").forEach((a) => a.classList.remove("active"));

  if (h.startsWith("problem/")) {
    const parts = h.split("/");
    const source = parts[1];
    const idx = parts[2];
    $("#detail").classList.add("active");
    renderDetail(source, idx);
    window.scrollTo(0, 0);
    return;
  }
  const target = h === "about" ? "about" : "browse";
  $(`#${target}`).classList.add("active");
  $$(`.nav-link[data-target="${target}"]`).forEach((a) => a.classList.add("active"));
  window.scrollTo(0, 0);
}

document.addEventListener("DOMContentLoaded", init);

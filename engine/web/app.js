"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) throw new Error((body && (body.detail?.[0]?.msg || body.detail)) || `${res.status} ${res.statusText}`);
  return body;
}

const SOURCE_NAMES = { hotpotqa: "HotpotQA", postmortems: "Post-mortems", northwind: "Northwind Pay", upload: "Upload" };
const VERDICTS = {
  answered: "Answered from the evidence",
  insufficient: "Not enough evidence",
  conflicting: "Sources disagree",
  llm_unavailable: "Language model unavailable",
};
const EXAMPLES = [
  "Which system changes caused the payment processing delays in March 2024, which services were affected, and which documents confirm this?",
  "How long did the 12 March 2024 payment incident last?",
  "Who approved change CHG-1190?",
  "Were Scott Derrickson and Ed Wood of the same nationality?",
  "What caused the AWS Seoul region DNS resolver outage, and how long did it last?",
];

/* ------------------------------------------------------------------ tabs */
document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
    document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.id !== `tab-${btn.dataset.tab}`));
    if (btn.dataset.tab === "documents" && !$("#doc-rows").children.length) loadDocs(true);
    if (btn.dataset.tab === "system") loadSystem();
  });
});

/* ------------------------------------------------------------------ header */
async function loadHeader() {
  try {
    const h = await api("/api/health");
    if (h.status === "starting") { $("#corpus-line").textContent = "Loading models — first start takes about a minute."; setTimeout(loadHeader, 4000); return; }
    const s = await api("/api/stats");
    $("#corpus-line").textContent = `${s.documents.toLocaleString()} documents, ${s.chunks.toLocaleString()} passages indexed` +
      (h.llm_configured ? "" : " — answers disabled: no language model key configured");
  } catch (e) {
    $("#corpus-line").textContent = `Index unreachable: ${e.message}`;
  }
}

/* ------------------------------------------------------------------ investigate */
EXAMPLES.forEach((q) => {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = q;
  b.addEventListener("click", () => { $("#question").value = q; $("#ask-form").requestSubmit(); });
  $("#examples").append(b);
});

function trailItem(ev) {
  switch (ev.type) {
    case "plan": {
      const qs = (ev.plan.sub_questions || []).map((q) => `<q>${esc(q)}</q>`).join("<br>");
      return `<strong>Planned ${ev.plan.sub_questions.length} sub-question${ev.plan.sub_questions.length === 1 ? "" : "s"}</strong>${qs}`;
    }
    case "search": return `<strong>Searched</strong><q>${esc(ev.query)}</q> — ${ev.new.length} new passage${ev.new.length === 1 ? "" : "s"}`;
    case "tool_call":
      return ev.tool === "read_document"
        ? `<strong>Read document</strong>${esc(ev.args.doc_id)} — ${ev.new.length} passages`
        : `<strong>Followed a lead</strong><q>${esc(ev.args.query)}</q> — ${ev.new.length} new`;
    case "finish": return `<strong>Stopped searching</strong>${esc(ev.reason || "")}`;
    case "synthesis": return `<strong>Writing the answer</strong>from ${ev.evidence} passages`;
    case "llm_error": return `<strong>Language model failed</strong>${esc(ev.kind)} — showing evidence only`;
    case "done": return `<strong>Done</strong>`;
    default: return null;
  }
}

function addTrail(ev) {
  const html = trailItem(ev);
  if (!html) return;
  $("#trail").querySelectorAll("li.active").forEach((li) => li.classList.remove("active"));
  const li = document.createElement("li");
  li.innerHTML = html + (ev.t_ms != null ? ` <span class="t">· ${(ev.t_ms / 1000).toFixed(1)}s</span>` : "");
  if (ev.type === "llm_error") li.classList.add("error");
  if (ev.type !== "done") li.classList.add("active");
  $("#trail").append(li);
}

function citeButtons(text) {
  return esc(text).replace(/\[(S\d+)\]/g, (_, l) => `<button type="button" class="cite" data-label="${l}" aria-label="Show evidence ${l}">${l}</button>`);
}

function renderExhibit(s, opts = {}) {
  const li = document.createElement("li");
  li.className = "exhibit" + (s.cited ? " cited" : "");
  li.id = opts.idPrefix ? `${opts.idPrefix}${s.label}` : "";
  const where = [SOURCE_NAMES[s.source] || s.source, s.section, s.page ? `page ${s.page}` : null].filter(Boolean).join(", ");
  const link = s.url ? ` <a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">original</a>` : "";
  li.innerHTML = `<span class="label">${esc(s.label || "")}</span>
    <div class="meta"><span class="title">${esc(s.title)}</span> — ${esc(where)}${link}
      ${s.suspicious ? '<br><span class="flag">Contains text that tries to instruct the AI — treated as data only</span>' : ""}</div>
    <p>${esc(s.text)}</p>
    ${opts.why ? `<div class="why">${opts.why}</div>` : ""}`;
  return li;
}

function renderAnswer(r) {
  const el = $("#answer");
  el.className = `answer status-${r.status}`;
  const conflicts = (r.conflicts || []).filter((c) => c.description);
  const missing = (r.missing_information || []).filter(Boolean);
  const claims = r.claims || [];
  el.innerHTML = `
    <span class="verdict">${esc(VERDICTS[r.status] || r.status)}</span>
    ${r.short_answer ? `<p class="short-answer">${esc(r.short_answer)}</p>` : ""}
    <p class="answer-body">${citeButtons(r.answer || "")}</p>
    ${conflicts.length ? `<h3>Where the sources disagree</h3><ul class="conflicts">${conflicts.map((c) =>
      `<li>${esc(c.description)} ${c.citations.map((l) => `<button type="button" class="cite" data-label="${l}">${l}</button>`).join("")}</li>`).join("")}</ul>` : ""}
    ${missing.length ? `<h3>Not found in the documents</h3><ul class="missing">${missing.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>` : ""}
    ${claims.length ? `<details><summary>Claim check (${claims.length - (r.unsupported_claims || 0)} of ${claims.length} supported by their citations)</summary>
      <ul class="claims">${claims.map((c) => `<li class="${c.supported ? "" : "unsupported"}">${esc(c.statement)} ${c.citations.map((l) =>
        `<button type="button" class="cite" data-label="${l}">${l}</button>`).join("")}${c.supported ? "" : " — weak support"}</li>`).join("")}</ul></details>` : ""}
    ${r.cached ? '<p class="hint">Served from cache.</p>' : ""}`;

  const list = $("#exhibits");
  list.innerHTML = "";
  const sources = [...(r.sources || [])].sort((a, b) => Number(b.cited) - Number(a.cited));
  sources.forEach((s) => list.append(renderExhibit(s, { idPrefix: "ex-", why: `found by ${s.found_by.startsWith("read:") ? "reading the document" : `"${esc(s.found_by)}"`}` })));
  if (!sources.length) list.innerHTML = '<li class="empty">No passages matched this question.</li>';

  const u = r.usage || {};
  $("#usage").textContent = `${(u.latency_ms / 1000 || 0).toFixed(1)} s · ${u.calls || 0} model calls · ${(u.input_tokens || 0).toLocaleString()} input / ${(u.output_tokens || 0).toLocaleString()} output tokens`;
}

$("#answer").addEventListener("pointerover", spotlight);
$("#answer").addEventListener("focusin", spotlight);
$("#answer").addEventListener("click", (e) => {
  const b = e.target.closest(".cite");
  if (b) document.getElementById(`ex-${b.dataset.label}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
});
function spotlight(e) {
  const b = e.target.closest(".cite");
  document.querySelectorAll(".exhibit.spotlight").forEach((x) => x.classList.remove("spotlight"));
  if (b) document.getElementById(`ex-${b.dataset.label}`)?.classList.add("spotlight");
}

async function streamAsk(body, onEvent) {
  const res = await fetch("/api/ask/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) {
    let msg = `${res.status}`;
    try { msg = (await res.json()).detail || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const data = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (data) onEvent(JSON.parse(data.slice(6)));
    }
  }
}

$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#question").value.trim();
  if (question.length < 3) return;
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const src = $("#source-filter").value;
  $("#workbench").hidden = false;
  $("#trail").innerHTML = "";
  $("#exhibits").innerHTML = "";
  $("#usage").textContent = "";
  $("#answer").className = "answer";
  $("#answer").innerHTML = '<p class="thinking">Investigating…</p>';
  const btn = $("#ask-btn");
  btn.disabled = true;
  try {
    await streamAsk({ question, mode, sources: src ? [src] : null }, (ev) => {
      if (ev.type === "result") renderAnswer(ev.result);
      else if (ev.type === "error") throw new Error(ev.message);
      else addTrail(ev);
    });
  } catch (err) {
    $("#answer").className = "answer status-llm_unavailable";
    $("#answer").innerHTML = `<span class="verdict">Request failed</span><p>${esc(err.message)}</p>`;
  } finally {
    btn.disabled = false;
  }
});

/* ------------------------------------------------------------------ documents */
let docCursor = null;
async function loadDocs(reset) {
  if (reset) { docCursor = null; $("#doc-rows").innerHTML = ""; }
  const p = new URLSearchParams({ limit: "50" });
  if ($("#doc-q").value.trim()) p.set("q", $("#doc-q").value.trim());
  if ($("#doc-source").value) p.set("source", $("#doc-source").value);
  if (docCursor) p.set("cursor", docCursor);
  try {
    const r = await api(`/api/documents?${p}`);
    r.documents.forEach((d) => {
      const tr = document.createElement("tr");
      tr.dataset.id = d.doc_id;
      tr.tabIndex = 0;
      tr.innerHTML = `<td>${esc(d.title)}</td><td>${esc(SOURCE_NAMES[d.source] || d.source)}</td><td>${esc(d.fmt)}</td><td class="num">${d.n_chunks}</td><td>${d.near_duplicate_of ? "near-duplicate" : ""}</td>`;
      $("#doc-rows").append(tr);
    });
    if (reset && !r.documents.length) $("#doc-rows").innerHTML = '<tr><td colspan="5" class="empty">No documents match. Add one above or clear the filter.</td></tr>';
    docCursor = r.next_cursor;
    $("#doc-more").hidden = !docCursor;
  } catch (e) {
    $("#doc-rows").innerHTML = `<tr><td colspan="5" class="empty">Could not load documents: ${esc(e.message)}</td></tr>`;
  }
}
$("#doc-filter").addEventListener("submit", (e) => { e.preventDefault(); loadDocs(true); });
$("#doc-more").addEventListener("click", () => loadDocs(false));
$("#doc-rows").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr) showDoc(tr.dataset.id); });
$("#doc-rows").addEventListener("keydown", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr && e.key === "Enter") showDoc(tr.dataset.id); });

async function showDoc(id) {
  const box = $("#doc-detail");
  box.hidden = false;
  box.innerHTML = '<p class="thinking">Loading…</p>';
  try {
    const d = await api(`/api/documents/${encodeURI(id)}`);
    const protectedDoc = d.source !== "upload";
    box.innerHTML = `<h3>${esc(d.title)}</h3>
      <p class="hint">${esc(SOURCE_NAMES[d.source] || d.source)}, ${esc(d.fmt)}, ${d.n_words} words, updated ${esc(d.updated_at || d.created_at)}
      ${d.url ? `<br><a href="${esc(d.url)}" target="_blank" rel="noopener noreferrer">${esc(d.url)}</a>` : ""}
      ${d.near_duplicate_of ? `<br>Near-duplicate of ${esc(d.near_duplicate_of)}` : ""}</p>
      ${protectedDoc ? '<p class="hint">Part of the seed corpus: changes need the admin token.</p>' : `
      <div class="doc-actions">
        <label class="quiet"><input type="file" id="replace-file" hidden><button type="button" id="replace-btn">Replace file</button></label>
        <button type="button" class="danger" id="delete-btn">Delete document</button>
      </div>`}
      ${d.chunks.map((c) => `<div class="chunk"><small>Passage ${c.index}${c.section ? ` — ${esc(c.section)}` : ""}${c.page ? `, page ${c.page}` : ""}${c.suspicious ? ' — <span class="flag">suspected injected instructions</span>' : ""}</small>${esc(c.text)}</div>`).join("")}`;
    if (!protectedDoc) {
      $("#replace-btn").addEventListener("click", () => $("#replace-file").click());
      $("#replace-file").addEventListener("change", async (e) => {
        const f = e.target.files[0];
        if (!f) return;
        const fd = new FormData();
        fd.append("file", f);
        try { const r = await api(`/api/documents/${encodeURI(id)}`, { method: "PUT", body: fd }); showDoc(id); loadDocs(true); flash(`${f.name}: ${r.status}`); }
        catch (err) { flash(`${f.name}: ${err.message}`, true); }
      });
      const del = $("#delete-btn");
      del.addEventListener("click", async () => {
        if (del.dataset.armed !== "1") { del.dataset.armed = "1"; del.textContent = "Click again to delete"; return; }
        try { await api(`/api/documents/${encodeURI(id)}`, { method: "DELETE" }); box.hidden = true; loadDocs(true); loadHeader(); flash(`Deleted ${d.title}`); }
        catch (err) { flash(err.message, true); }
      });
    }
  } catch (e) {
    box.innerHTML = `<p class="empty">${esc(e.message)}</p>`;
  }
}

function flash(msg, bad = false) {
  const li = document.createElement("li");
  li.textContent = msg;
  if (bad) li.className = "bad";
  $("#upload-results").prepend(li);
}

async function uploadFiles(files) {
  if (!files.length) return;
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f));
  flash(`Indexing ${files.length} file${files.length === 1 ? "" : "s"}…`);
  try {
    const r = await api("/api/documents", { method: "POST", body: fd });
    const words = { created: "added", updated: "updated", unchanged: "already indexed, no change", duplicate: "identical to an existing document, not added" };
    r.results.forEach((x) => flash(`${x.filename}: ${words[x.status] || x.status}${x.duplicate_of ? ` (${x.duplicate_of})` : ""}${x.near_duplicate_of ? ` — very similar to ${x.near_duplicate_of}` : ""}${x.n_chunks ? `, ${x.n_chunks} passages` : ""}`));
    r.errors.forEach((x) => flash(`${x.filename}: ${x.error}`, true));
    loadDocs(true);
    loadHeader();
  } catch (e) {
    flash(e.message, true);
  }
}
$("#files").addEventListener("change", (e) => { uploadFiles(e.target.files); e.target.value = ""; });
const drop = $("#drop");
["dragover", "dragenter"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); uploadFiles(e.dataTransfer.files); });

/* ------------------------------------------------------------------ search lab */
$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const list = $("#search-results");
  list.innerHTML = '<li class="thinking">Searching…</li>';
  try {
    const r = await api("/api/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: $("#search-q").value, mode: $("#search-mode").value, rerank: $("#search-rerank").checked, mmr: $("#search-mmr").checked, top_k: 10 }),
    });
    $("#search-timing").textContent = Object.entries(r.timings_ms).map(([k, v]) => `${k.replace("_ms", "")} ${v} ms`).join(" · ");
    list.innerHTML = "";
    r.hits.forEach((h, i) => {
      const d = h.debug || {};
      const parts = [`score ${h.score}`, d.dense_rank ? `semantic #${d.dense_rank}` : "not in semantic top", d.sparse_rank ? `keyword #${d.sparse_rank}` : "not in keyword top"];
      if (d.ce != null) parts.push(`cross-encoder ${d.ce}`);
      if (d.title_boost) parts.push("title named in query");
      const li = renderExhibit({ ...h, label: `#${i + 1}` }, { why: parts.join(" · ") });
      list.append(li);
    });
    if (!r.hits.length) list.innerHTML = '<li class="empty">Nothing matched. Try fewer or different words.</li>';
  } catch (err) {
    list.innerHTML = `<li class="empty">${esc(err.message)}</li>`;
  }
});

/* ------------------------------------------------------------------ system */
function panel(title, rows) {
  return `<section class="panel"><h2>${esc(title)}</h2><dl>${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl></section>`;
}
async function loadSystem() {
  try {
    const [h, s, m] = await Promise.all([api("/api/health"), api("/api/stats"), api("/api/metrics")]);
    const llm = Object.entries(m.llm || {});
    const lat = Object.entries(m.latency || {}).filter(([k]) => !k.startsWith("http.GET /static"));
    $("#system").innerHTML =
      panel("Services", [["Vector database", `${h.vector_db} (${h.vector_backend})`], ["Language model", h.llm_configured ? h.llm_model : "not configured"],
        ["Fallback model", h.fallback_model], ["Embeddings", h.embedding_model], ["Re-ranker", h.reranker || "off"]]) +
      panel("Index", [["Documents", s.documents.toLocaleString()], ["Passages", s.chunks.toLocaleString()],
        ...Object.entries(s.documents_by_source).map(([k, v]) => [SOURCE_NAMES[k] || k, v.toLocaleString()]),
        ["Chunking", `${s.chunking.strategy}, ${s.chunking.max_words} words, ${s.chunking.overlap} overlap`]]) +
      panel("Caches", [["Answers cached", s.answer_cache.items], ["Answer cache hits", s.answer_cache.hits],
        ["Query embedding hits", s.embedding_cache.hits], ["Query embedding misses", s.embedding_cache.misses]]) +
      panel("Language model usage", llm.length ? llm.flatMap(([model, u]) => [[model, `${u.calls} calls`], ["tokens in / out", `${u.input_tokens} / ${u.output_tokens}`],
        ["paid-tier equivalent", `$${u.cost_usd}`]]) : [["Calls", "none yet"]]) +
      panel("Latency (p50 / p95)", lat.length ? lat.map(([k, v]) => [k, `${v.p50_ms} / ${v.p95_ms} ms (${v.n})`]) : [["Requests", "none yet"]]) +
      panel("Counters", Object.entries(m.counters).map(([k, v]) => [k, v]).concat([["Uptime", `${Math.round(m.uptime_s / 60)} min`]]));
  } catch (e) {
    $("#system").innerHTML = `<p class="empty">${esc(e.message)}</p>`;
  }
}

loadHeader();

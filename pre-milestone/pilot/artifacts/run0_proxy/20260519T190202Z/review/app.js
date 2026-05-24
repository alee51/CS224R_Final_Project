(function () {
  "use strict";

  const data = window.REVIEW_DATA;
  if (!data || !Array.isArray(data.prompts)) {
    document.body.innerHTML =
      '<p class="empty-state">Missing data. Run <code>python build_review_dashboard.py</code> first.</p>';
    return;
  }

  const KATEX_OPTS = {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
    strict: "ignore",
    trust: true,
    ignoredTags: ["script", "noscript", "style", "textarea"],
  };

  const prompts = data.prompts;
  const listEl = document.getElementById("prompt-list");
  const countEl = document.getElementById("list-count");
  const filterEl = document.getElementById("filter");
  const searchEl = document.getElementById("search");
  const titleEl = document.getElementById("detail-title");
  const problemEl = document.getElementById("problem-text");
  const goldEl = document.getElementById("gold-answer");
  const rolloutsEl = document.getElementById("rollouts");
  const prevBtn = document.getElementById("btn-prev");
  const nextBtn = document.getElementById("btn-next");
  const panelAbout = document.getElementById("panel-about");
  const panelStats = document.getElementById("panel-stats");
  const expandAllEl = document.getElementById("expand-all");
  const sourcePillEl = document.getElementById("source-pill");

  let filteredIndices = prompts.map((_, i) => i);
  let selectedPromptIndex = 0;
  let activeListPos = 0;
  let expandAll = false;
  let activeSource = "raw";

  const dualSource = Boolean(data.dual_source);

  if (location.protocol === "file:") {
    showFileProtocolBanner();
  }

  function showFileProtocolBanner() {
    const banner = document.createElement("div");
    banner.className = "file-protocol-banner";
    banner.innerHTML =
      'LaTeX needs network access for KaTeX. If math does not render, run ' +
      '<code>./serve.sh</code> in <code>review/</code> and open ' +
      '<code>http://localhost:8765</code>.';
    document.body.prepend(banner);
  }

  function truncateId(pid) {
    if (!pid) return "";
    return pid.length <= 12 ? pid : pid.slice(0, 8) + "…";
  }

  function getPromptCounts(p) {
    if (activeSource === "clean") {
      return {
        n_correct: p.n_correct_clean ?? p.n_correct ?? 0,
        n_clusters: p.n_clusters_clean ?? p.n_clusters ?? 0,
      };
    }
    return {
      n_correct: p.n_correct_raw ?? p.n_correct ?? 0,
      n_clusters: p.n_clusters_raw ?? p.n_clusters ?? 0,
    };
  }

  function getRolloutDisplay(r) {
    if (r.raw && r.clean) {
      return activeSource === "clean" ? r.clean : r.raw;
    }
    return r;
  }

  function rolloutHasDelta(r) {
    if (!r.delta) return false;
    return r.delta.parsed || r.delta.correct;
  }

  function matchesFilter(p) {
    const { n_correct } = getPromptCounts(p);
    switch (filterEl.value) {
      case "has_correct":
        return n_correct > 0;
      case "no_correct":
        return n_correct === 0;
      case "partial":
        return n_correct >= 1 && n_correct <= 7;
      default:
        return true;
    }
  }

  function matchesSearch(p) {
    const q = searchEl.value.trim().toLowerCase();
    if (!q) return true;
    return p.prompt_id.toLowerCase().includes(q);
  }

  function correctBadgeClass(n) {
    if (n === 0) return "none";
    if (n < 8) return "partial";
    return "";
  }

  function typesetMath(el) {
    if (!el) return;
    const render = window.renderMathInElement;
    if (typeof render !== "function") {
      console.warn("KaTeX auto-render not loaded; math will show as plain text.");
      return;
    }
    try {
      render(el, KATEX_OPTS);
    } catch (err) {
      console.warn("KaTeX render failed:", err);
    }
  }

  function setMathText(el, text) {
    if (!el) return;
    el.textContent = text == null ? "" : String(text);
    typesetMath(el);
  }

  function updateSourcePill() {
    if (!sourcePillEl) return;
    sourcePillEl.textContent = activeSource === "clean" ? "CLEAN" : "RAW";
    sourcePillEl.className =
      "source-pill " + (activeSource === "clean" ? "clean" : "raw");
    sourcePillEl.title = dualSource
      ? "Press ; to switch raw ↔ cleaned labels"
      : "Single-source build";
  }

  function toggleSource() {
    if (!dualSource) return;
    activeSource = activeSource === "raw" ? "clean" : "raw";
    updateSourcePill();
    renderList();
    renderDetail();
  }

  function applyFilters() {
    filteredIndices = [];
    for (let i = 0; i < prompts.length; i++) {
      const p = prompts[i];
      if (matchesFilter(p) && matchesSearch(p)) filteredIndices.push(i);
    }
    countEl.textContent = `${filteredIndices.length} / ${prompts.length}`;

    const stillVisible = filteredIndices.includes(selectedPromptIndex);
    if (!stillVisible && filteredIndices.length) {
      selectedPromptIndex = filteredIndices[0];
    }
    activeListPos = Math.max(0, filteredIndices.indexOf(selectedPromptIndex));
    renderList();
    renderDetail();
  }

  function renderList() {
    listEl.innerHTML = "";
    filteredIndices.forEach((promptIdx, pos) => {
      const p = prompts[promptIdx];
      const { n_correct, n_clusters } = getPromptCounts(p);
      const li = document.createElement("li");
      let cls = "prompt-item";
      if (promptIdx === selectedPromptIndex) cls += " active";
      if (p.has_delta) cls += " has-delta";
      li.className = cls;
      li.dataset.promptIndex = String(promptIdx);
      li.dataset.listPos = String(pos);
      const deltaMark = p.has_delta
        ? '<span class="delta-badge" title="Parsed or correct differs raw vs clean (excl. cluster, run-on)">Δ</span>'
        : "";
      li.innerHTML = `
        <div class="row-top">
          <span class="idx">#${p.index + 1}</span>
          <span class="pid" title="${escapeAttr(p.prompt_id)}">${escapeHtml(truncateId(p.prompt_id))}</span>
          ${deltaMark}
        </div>
        <div class="row-bottom">
          <span class="badge badge-gold">gold: ${escapeHtml(p.gold_answer)}</span>
          <span class="badge badge-correct ${correctBadgeClass(n_correct)}">${n_correct}/8</span>
          <span class="badge badge-clusters">${n_clusters} clusters</span>
        </div>`;
      li.addEventListener("click", () => selectPrompt(promptIdx, pos));
      listEl.appendChild(li);
    });
  }

  function selectPrompt(promptIndex, listPos) {
    selectedPromptIndex = promptIndex;
    if (typeof listPos === "number") activeListPos = listPos;
    else activeListPos = filteredIndices.indexOf(promptIndex);
    renderList();
    renderDetail();
    scrollActiveIntoView();
  }

  function scrollActiveIntoView() {
    const active = listEl.querySelector(".prompt-item.active");
    if (active) active.scrollIntoView({ block: "nearest" });
  }

  function renderDetail() {
    const p = prompts[selectedPromptIndex];
    if (!p) {
      titleEl.textContent = "No prompt selected";
      problemEl.textContent = "";
      goldEl.textContent = "";
      rolloutsEl.innerHTML = "";
      return;
    }

    titleEl.textContent = `#${p.index + 1} · ${p.prompt_id}`;
    setMathText(problemEl, p.problem);
    setMathText(goldEl, p.gold_answer);

    rolloutsEl.innerHTML = "";
    p.rollouts.forEach((r, i) => {
      rolloutsEl.appendChild(buildRolloutCard(r, i + 1));
    });
    syncAllCompletionBodies();

    prevBtn.disabled = activeListPos <= 0;
    nextBtn.disabled = activeListPos >= filteredIndices.length - 1;
  }

  function syncAllCompletionBodies() {
    rolloutsEl.querySelectorAll(".rollout-card").forEach((card) => {
      const body = card.querySelector(".completion-body");
      const toggle = card.querySelector(".completion-toggle");
      if (!body || !toggle) return;
      if (expandAll) {
        body.classList.add("open");
        toggle.setAttribute("aria-expanded", "true");
        toggle.textContent = "Hide full completion";
        typesetCompletion(body);
      } else {
        body.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
        toggle.textContent = "Show full completion";
      }
    });
  }

  function setExpandAll(on) {
    expandAll = on;
    if (expandAllEl) expandAllEl.checked = on;
    syncAllCompletionBodies();
  }

  function toggleDetailsPanel(panel) {
    if (!panel) return;
    panel.open = !panel.open;
  }

  function typesetCompletion(body) {
    const content = body.querySelector(".completion-content");
    if (!content || content.dataset.katexDone === "1") return;
    typesetMath(content);
    content.dataset.katexDone = "1";
  }

  function buildRolloutCard(r, num) {
    const disp = getRolloutDisplay(r);
    const card = document.createElement("article");
    let cardCls = "rollout-card";
    if (rolloutHasDelta(r)) cardCls += " has-delta";
    card.className = cardCls;

    const statusClass = disp.correct ? "status-correct" : "status-wrong";
    const statusLabel = disp.correct ? "Correct" : "Wrong";
    const toggleId = `completion-${selectedPromptIndex}-${num}`;

    const delta = r.delta || {};
    const parsedDelta = delta.parsed ? " field-delta" : "";
    const correctDelta = delta.correct ? " field-delta" : "";

    const isRunon = activeSource === "clean" && Boolean(disp.is_runon_fallback);
    const answerLabel = isRunon ? "run-on:" : "parsed:";

    let extraMeta = "";
    if (activeSource === "clean" && disp.extract_path_clean != null && !isRunon) {
      extraMeta += `<div><dt>extract:</dt><dd>${escapeHtml(String(disp.extract_path_clean))}</dd></div>`;
    }

    card.innerHTML = `
      <div class="rollout-header">
        <span class="rollout-num">Rollout ${num}</span>
        <span class="badge ${statusClass}${correctDelta}">${statusLabel}</span>
      </div>
      <dl class="rollout-meta">
        <div><dt>${answerLabel}</dt><dd class="math-content parsed-dd${parsedDelta}"></dd></div>
        <div><dt>cluster:</dt><dd class="cluster-dd">${escapeHtml(String(disp.cluster_id))}</dd></div>
        <div><dt>chars:</dt><dd>${r.char_count != null ? r.char_count : (r.completion || "").length}</dd></div>
        ${extraMeta}
      </dl>
      <button type="button" class="completion-toggle" aria-expanded="false" aria-controls="${toggleId}">
        Show full completion
      </button>
      <div id="${toggleId}" class="completion-body">
        <div class="completion-content math-content"></div>
      </div>`;

    const parsedDd = card.querySelector(".parsed-dd");
    if (isRunon) {
      parsedDd.classList.add("runon-value");
      const text = String(disp.parsed_answer || "").trim();
      parsedDd.textContent = text || "(rejected)";
    } else {
      setMathText(parsedDd, disp.parsed_answer);
    }

    const content = card.querySelector(".completion-content");
    content.textContent = r.completion;

    const toggle = card.querySelector(".completion-toggle");
    const body = card.querySelector(".completion-body");
    toggle.addEventListener("click", () => {
      const open = body.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      toggle.textContent = open ? "Hide full completion" : "Show full completion";
      if (open) typesetCompletion(body);
      if (!open && expandAll) {
        expandAll = false;
        if (expandAllEl) expandAllEl.checked = false;
      }
    });

    if (expandAll) {
      body.classList.add("open");
      toggle.setAttribute("aria-expanded", "true");
      toggle.textContent = "Hide full completion";
      typesetCompletion(body);
    }

    return card;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function escapeAttr(s) {
    return escapeHtml(s).replace(/'/g, "&#39;");
  }

  function navigate(delta) {
    if (!filteredIndices.length) return;
    const nextPos = activeListPos + delta;
    if (nextPos < 0 || nextPos >= filteredIndices.length) return;
    selectPrompt(filteredIndices[nextPos], nextPos);
  }

  filterEl.addEventListener("change", applyFilters);
  searchEl.addEventListener("input", applyFilters);
  prevBtn.addEventListener("click", () => navigate(-1));
  nextBtn.addEventListener("click", () => navigate(1));
  if (expandAllEl) {
    expandAllEl.addEventListener("change", () => setExpandAll(expandAllEl.checked));
  }

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, select")) return;
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (key === "j" || key === "ArrowDown") {
      e.preventDefault();
      navigate(1);
    } else if (key === "k" || key === "ArrowUp") {
      e.preventDefault();
      navigate(-1);
    } else if (key === "l") {
      e.preventDefault();
      setExpandAll(!expandAll);
    } else if (key === "i") {
      e.preventDefault();
      toggleDetailsPanel(panelAbout);
    } else if (key === "s") {
      e.preventDefault();
      toggleDetailsPanel(panelStats);
    } else if (e.key === ";") {
      e.preventDefault();
      toggleSource();
    }
  });

  function start() {
    updateSourcePill();
    applyFilters();
  }

  if (typeof window.renderMathInElement === "function") {
    start();
  } else {
    window.addEventListener("load", start);
  }
})();

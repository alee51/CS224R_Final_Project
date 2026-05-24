(() => {
  const DATA = window.DASHBOARD;
  let filter = "all";
  let search = "";
  let activeIdx = null;
  const listEl = document.getElementById("list");
  const countEl = document.getElementById("count");
  const detailEl = document.getElementById("detail");
  const emptyEl = document.getElementById("empty");

  function passesFilter(p) {
    if (search && !p.prompt_id.includes(search)) return false;
    switch (filter) {
      case "has-correct": return p.n_correct > 0;
      case "no-correct": return p.n_correct === 0;
      case "partial": return p.n_correct > 0 && p.n_correct < p.n_rollouts;
      case "minority-llm": return p.minority_correct_llm;
      case "minority-cleaned": return p.minority_correct_cleaned;
      default: return true;
    }
  }

  function renderList() {
    const filtered = DATA.prompts.filter(passesFilter);
    countEl.textContent = `${filtered.length} / ${DATA.n_prompts} prompts`;
    listEl.innerHTML = "";
    filtered.forEach((p, i) => {
      const row = document.createElement("div");
      row.className = "prompt-row" + (activeIdx === p.prompt_id ? " active" : "");
      const minoritybadges = [];
      if (p.minority_correct_llm) minoritybadges.push('<span class="badge b-minority">min-LLM</span>');
      if (p.minority_correct_cleaned) minoritybadges.push('<span class="badge b-minority">min-cleaned</span>');
      const correctBadge = p.n_correct > 0
        ? `<span class="badge b-correct">${p.n_correct}/${p.n_rollouts}✓</span>`
        : `<span class="badge b-wrong">0/${p.n_rollouts}</span>`;
      row.innerHTML = `
        <div class="pid">${p.prompt_id.slice(0, 8)}…${p.prompt_id.slice(-4)}</div>
        <div class="stats">
          ${correctBadge}
          <span class="badge b-cluster">ans: ${p.n_clusters_cleaned}c</span>
          <span class="badge b-cluster">LLM: ${p.n_clusters_llm}c</span>
          ${minoritybadges.join("")}
        </div>
      `;
      row.onclick = () => selectPrompt(p.prompt_id);
      listEl.appendChild(row);
    });
  }

  function renderDetail(pid) {
    const p = DATA.prompts.find(x => x.prompt_id === pid);
    if (!p) return;
    emptyEl.hidden = true;
    detailEl.hidden = false;

    const goldStr = p.gold_answer ? `<span class="gold">${escapeHtml(String(p.gold_answer))}</span>` : "<em>(none)</em>";

    const summary = `
      <div class="summary-grid">
        <div class="summary-cell">
          <div class="label">Correct rollouts</div>
          <div class="value">${p.n_correct} / ${p.n_rollouts}</div>
        </div>
        <div class="summary-cell">
          <div class="label">LLM clusters</div>
          <div class="value">${p.n_clusters_llm}</div>
          <div class="sub">largest correct: ${p.largest_correct_llm || "—"}</div>
        </div>
        <div class="summary-cell">
          <div class="label">Cleaned-answer clusters</div>
          <div class="value">${p.n_clusters_cleaned}</div>
          <div class="sub">largest correct: ${p.largest_correct_cleaned || "—"}</div>
        </div>
        <div class="summary-cell">
          <div class="label">Minority-correct?</div>
          <div class="value">${p.minority_correct_llm ? "LLM ✓" : "—"} ${p.minority_correct_cleaned ? "cleaned ✓" : ""}</div>
          <div class="sub">${(!p.minority_correct_llm && !p.minority_correct_cleaned) ? "neither" : ""}</div>
        </div>
      </div>
    `;

    const rolloutsHtml = p.rollouts.map(r => {
      const corrClass = r.cleaned_correct ? "correct" : "wrong";
      const corrMark = r.cleaned_correct ? "✓" : "✗";
      const degen = r.llm_degenerate ? ' <span class="degen">[degenerate]</span>' : "";
      const llmPillClass = "cluster-pill llm" + (r.llm_degenerate ? " degen" : "");
      const ansDisplay = r.cleaned_state === "extracted"
        ? String(r.cleaned_answer)
        : `(${r.cleaned_state})`;
      return `
        <div class="rollout">
          <div class="rollout-head" data-idx="${r.idx}">
            <span class="idx">#${r.idx}</span>
            <span class="answer ${corrClass}">${corrMark} ${escapeHtml(ansDisplay)}</span>
            <span class="${llmPillClass}">LLM ${r.llm_cluster_id} (×${r.llm_cluster_size})${degen}</span>
            <span class="cluster-pill v2">ans ${r.cleaned_cluster_id} (×${r.cleaned_cluster_size})</span>
            <span class="cluster">state: ${escapeHtml(r.cleaned_state)}</span>
          </div>
          <div class="rollout-body" id="rb-${r.idx}"><pre>${escapeHtml(r.completion || "(empty)")}</pre></div>
        </div>
      `;
    }).join("");

    detailEl.innerHTML = `
      <h1>Prompt <span class="pid-mono">${p.prompt_id}</span></h1>
      <div class="problem-block">
        <div><strong>Problem:</strong></div>
        <div id="problem-text">${escapeHtml(p.problem)}</div>
        <div style="margin-top:8px"><strong>Gold:</strong> ${goldStr}</div>
      </div>
      ${summary}
      <div class="toggle-all"><button id="expand-all">Expand all rollouts</button>
        <button id="collapse-all">Collapse all</button></div>
      ${rolloutsHtml}
    `;

    detailEl.querySelectorAll(".rollout-head").forEach(h => {
      h.onclick = () => {
        const idx = h.dataset.idx;
        const body = document.getElementById("rb-" + idx);
        body.classList.toggle("open");
        if (body.classList.contains("open") && window.renderMathInElement) {
          window.renderMathInElement(body, KATEX_OPTS);
        }
      };
    });
    document.getElementById("expand-all").onclick = () => {
      detailEl.querySelectorAll(".rollout-body").forEach(b => {
        b.classList.add("open");
        if (window.renderMathInElement) window.renderMathInElement(b, KATEX_OPTS);
      });
    };
    document.getElementById("collapse-all").onclick = () =>
      detailEl.querySelectorAll(".rollout-body").forEach(b => b.classList.remove("open"));

    if (window.renderMathInElement) {
      window.renderMathInElement(document.getElementById("problem-text"), KATEX_OPTS);
    }
  }

  const KATEX_OPTS = {
    delimiters: [
      {left: "$$", right: "$$", display: true},
      {left: "\\[", right: "\\]", display: true},
      {left: "$", right: "$", display: false},
      {left: "\\(", right: "\\)", display: false},
    ],
    throwOnError: false,
  };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  }

  function selectPrompt(pid) {
    activeIdx = pid;
    renderList();
    renderDetail(pid);
  }

  document.getElementById("search").oninput = e => { search = e.target.value.trim(); renderList(); };
  document.querySelectorAll("#filters button").forEach(b => {
    b.onclick = () => {
      filter = b.dataset.f;
      document.querySelectorAll("#filters button").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      renderList();
    };
  });

  document.addEventListener("keydown", e => {
    if (document.activeElement.tagName === "INPUT") return;
    const filtered = DATA.prompts.filter(passesFilter);
    const i = filtered.findIndex(p => p.prompt_id === activeIdx);
    if (e.key === "j" || e.key === "ArrowDown") {
      const nxt = filtered[Math.min(i + 1, filtered.length - 1)];
      if (nxt) selectPrompt(nxt.prompt_id);
      e.preventDefault();
    } else if (e.key === "k" || e.key === "ArrowUp") {
      const prv = filtered[Math.max(i - 1, 0)];
      if (prv) selectPrompt(prv.prompt_id);
      e.preventDefault();
    } else if (e.key === "l") {
      document.getElementById("expand-all")?.click();
    }
  });

  renderList();
  if (DATA.prompts.length > 0) selectPrompt(DATA.prompts[0].prompt_id);
})();

/* CogniOS manufacturer dashboard controller: DOM and fetch only. */
(function () {
  "use strict";
  const M = window.CogniMfr;
  const $ = (id) => document.getElementById(id);
  const state = { interval: "quarter", seq: 0 };

  async function api(url) {
    let res;
    try { res = await fetch(url); } catch (e) { throw M.describeApiError(0); }
    let body = null;
    try { body = await res.json(); } catch (e) { /* non-JSON error page */ }
    if (!res.ok) throw M.describeApiError(res.status, body);
    return body;
  }

  function readFilters() {
    return { drug: $("f-drug").value, effect: $("f-effect").value, from: $("f-from").value, to: $("f-to").value,
             model: $("f-model").value, interval: state.interval };
  }

  function drawInterval() {
    $("f-interval").innerHTML = [["quarter", "Quarter"], ["month", "Month"]].map(([v, l]) =>
      `<button type="button" data-interval="${v}" aria-pressed="${v === state.interval}">${l}</button>`).join("");
  }

  const options = (sel, items, label) => { sel.innerHTML = `<option value="">${label}</option>` +
    items.map((i) => `<option value="${M.esc(i.value)}">${M.esc(i.label)}</option>`).join(""); };

  async function loadFilters() {
    try {
      const f = await api("/api/manufacturer/filters");
      options($("f-drug"), f.drugs.map((d) => ({ value: d.medication_id, label: d.drug_name })), "All drugs");
      options($("f-effect"), f.effects, "All effects");
      options($("f-model"), f.models, "All models");
      if (f.date_bounds) { $("f-from").min = $("f-to").min = f.date_bounds.from; $("f-from").max = $("f-to").max = f.date_bounds.to; }
    } catch (err) { $("status").innerHTML = M.errorNotice(err); }
  }

  async function load() {
    const filters = readFilters();
    const v = M.validateFilters(filters);
    if (!v.ok) { $("status").innerHTML = M.errorNotice({ title: "Check the dates", message: v.message }); return; }
    const seq = ++state.seq;
    $("status").innerHTML = `<div class="card"><div class="skeleton" style="width:40%"></div><div class="skeleton" style="width:70%;margin-top:12px"></div></div>`;
    try {
      const s = await api("/api/manufacturer/summary?" + M.buildQuery(filters));
      if (seq !== state.seq) return;
      $("status").innerHTML = "";
      $("content").innerHTML = M.dashboard(s);
    } catch (err) {
      if (seq !== state.seq) return;
      $("content").innerHTML = "";
      $("status").innerHTML = M.errorNotice(err);
    }
  }

  $("filter-form").addEventListener("submit", (e) => { e.preventDefault(); load(); });
  $("f-reset").addEventListener("click", () => {
    $("filter-form").reset(); state.interval = "quarter"; drawInterval(); load();
  });
  $("f-interval").addEventListener("click", (e) => {
    const b = e.target.closest("[data-interval]");
    if (b) { state.interval = b.dataset.interval; drawInterval(); load(); }
  });

  drawInterval();
  loadFilters().then(load);
})();

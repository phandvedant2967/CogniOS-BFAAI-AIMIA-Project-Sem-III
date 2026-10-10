/* CogniOS manufacturer dashboard: pure logic and renderers (no DOM, no fetch). Browser + Node tests. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.CogniMfr = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const MIN = 5;
  const CLASS_ORDER = ["medication_associated", "disease_progression", "other_cause", "uncertain"];
  const CLASS_COLOR = {
    medication_associated: "var(--rose-ink)", disease_progression: "var(--blue-ink)",
    other_cause: "var(--lav-ink)", uncertain: "var(--sand-ink)",
  };
  const CLASS_LABEL = {
    medication_associated: "Medication-associated pattern", disease_progression: "Disease-progression pattern",
    other_cause: "Other-cause pattern", uncertain: "Uncertain pattern",
  };
  const DOC_LABEL = {
    medication_associated: "Likely medication-associated", disease_progression: "Likely disease progression",
    other_cause: "Likely other cause", uncertain: "Uncertain",
  };
  const NOT_CONFIRMED =
    "AI-generated signals are pattern flags for clinician review. They are not confirmed adverse effects.";

  function esc(v) {
    return String(v == null ? "" : v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  /** A cell from the API is {value, suppressed}. Suppressed is shown as "<5", never as a number. */
  function fmtCell(c) {
    if (!c) return `<span class="supp">n/a</span>`;
    if (c.suppressed) return `<span class="supp" title="Fewer than ${MIN} patients: withheld to protect privacy">&lt;${MIN}</span>`;
    if (c.value == null) return `<span class="supp">n/a</span>`;
    return `<span class="num">${esc(c.value)}</span>`;
  }
  const cellValue = (c) => (c && !c.suppressed && typeof c.value === "number" ? c.value : null);
  const pct = (x) => (typeof x === "number" ? (x * 100).toFixed(1) + "%" : null);

  function buildQuery(f) {
    const q = new URLSearchParams();
    for (const k of ["drug", "effect", "from", "to", "model"]) if (f && f[k]) q.set(k, f[k]);
    q.set("interval", (f && f.interval) || "quarter");
    return q.toString();
  }

  function validateFilters(f) {
    if (f.from && f.to && f.from > f.to) return { ok: false, message: "The start date must not be later than the end date." };
    return { ok: true };
  }

  function describeApiError(status, body) {
    const b = body || {};
    if (status === 0) return { title: "Can't reach the server", message: "Check that the CogniOS server is running, then try again." };
    if (status === 400) return { title: "Filters not accepted", message: b.message || "The filters were not valid." };
    return { title: "Something went wrong", message: b.message || `The server returned status ${status}.` };
  }

  const chip = (tone, text) => `<span class="chip ${tone}">${esc(text)}</span>`;
  const AI_CHIP = chip("lav", "AI prediction");
  const DOC_CHIP = chip("ok", "Doctor assessment");

  /* ---------- KPIs ---------- */
  function kpis(t) {
    return `<div class="kpis">
      <div class="kpi ai"><div class="muted small">Cases with an AI prediction</div><div class="v">${fmtCell(t.cases)}</div>${AI_CHIP}</div>
      <div class="kpi ai"><div class="muted small">AI-flagged: medication-associated pattern</div><div class="v">${fmtCell(t.ai_flagged)}</div>${AI_CHIP}</div>
      <div class="kpi doctor"><div class="muted small">Doctor-reviewed cases</div><div class="v">${fmtCell(t.doctor_reviewed)}</div>${DOC_CHIP}</div>
      <div class="kpi"><div class="muted small">Drug groups shown / withheld (&lt;${MIN})</div><div class="v"><span class="num">${esc(t.drug_groups_shown)}</span> / <span class="num">${esc(t.drug_groups_withheld)}</span></div><span class="muted small">Counts in the cards above cover shown groups only.</span></div>
    </div>`;
  }

  /* ---------- Drug table ---------- */
  function mixBar(ai, total) {
    const parts = [];
    for (const k of CLASS_ORDER) {
      const c = ai.classes.shown[k];
      if (c && !c.suppressed && total) parts.push({ w: (c.value / total) * 100, color: CLASS_COLOR[k], label: `${CLASS_LABEL[k]}: ${c.value}` });
    }
    const oc = ai.classes.other_combined;
    if (oc && !oc.suppressed && oc.value && total) parts.push({ w: (oc.value / total) * 100, color: "var(--line-strong)", label: `Other classes combined: ${oc.value}` });
    if (!parts.length) return `<span class="supp">not shown</span>`;
    return `<div class="bar" style="display:flex;min-width:140px" role="img" aria-label="${esc(parts.map((p) => p.label).join("; "))}">${
      parts.map((p) => `<i style="width:${p.w.toFixed(1)}%;background:${p.color};border-radius:0" title="${esc(p.label)}"></i>`).join("")}</div>`;
  }

  function drugTable(drugs) {
    const rows = drugs.map((d) => {
      const total = cellValue(d.ai.cases);
      const share = pct(d.ai.flagged_share);
      const rev = d.doctor.reviewed_cases;
      return `<tr>
        <td><b>${esc(d.drug_name)}</b><div class="muted small">${esc(d.medication_id)} · ${esc(d.drug_class)}</div></td>
        <td class="num">${fmtCell(d.ai.cases)}</td>
        <td class="num">${fmtCell(d.ai.flagged)}${share ? ` <span class="muted small">(${esc(share)})</span>` : ""}</td>
        <td>${mixBar(d.ai, total)}</td>
        <td class="num">${fmtCell(rev)}${d.doctor.review_coverage != null ? ` <span class="muted small">(${esc(pct(d.doctor.review_coverage))} of cases)</span>` : ""}</td>
        <td class="num">${fmtCell(d.doctor.likely_medication_associated)}</td>
      </tr>`;
    }).join("");
    const legend = CLASS_ORDER.map((k) => `<span><i style="background:${CLASS_COLOR[k]}"></i>${esc(CLASS_LABEL[k])}</span>`).join("") +
      `<span><i style="background:var(--line-strong)"></i>Other classes combined</span>`;
    return `<div class="legend" aria-label="AI pattern mix legend">${legend}</div>
      <div class="table-wrap"><table class="mfr-table">
        <thead><tr><th rowspan="2">Drug</th><th colspan="3">${AI_CHIP}</th><th colspan="2">${DOC_CHIP}</th></tr>
        <tr><th class="num">Cases</th><th class="num">Flagged medication-associated</th><th>Pattern mix</th>
            <th class="num">Reviewed cases</th><th class="num">Judged likely medication-associated</th></tr></thead>
        <tbody>${rows}</tbody></table></div>`;
  }

  /* ---------- Effects chart ---------- */
  function effectBar(kind, label, c, denom) {
    const v = cellValue(c);
    const w = v != null && denom ? Math.min(100, (v / denom) * 100) : 0;
    return `<div class="bar-line"><div title="${esc(label)}" class="bar" aria-hidden="true"><i class="${kind}" style="width:${w.toFixed(1)}%"></i></div>
      <span class="small">${fmtCell(c)}</span></div>`;
  }

  function effectsChart(effects) {
    if (!effects.length) return "";
    const rows = effects.map((e) => {
      const denom = cellValue(e.cases_assessable);
      return `<div class="bar-row"><div><b>${esc(e.label)}</b>${e.rate != null ? `<div class="muted small">${esc(pct(e.rate))} of assessable cases</div>` : ""}</div>
        <div class="bar-stack">
          ${effectBar("all", "Cases with the effect newly recorded", e.cases_with_effect, denom)}
          ${effectBar("ai", "of which AI-flagged medication-associated", e.ai_flagged_with_effect, denom)}
          ${effectBar("doc", "Doctor-reviewed cases with the effect", e.doctor_reviewed_with_effect, denom)}
        </div></div>`;
    }).join("");
    return `<div class="legend"><span><i class="sw-all"></i>All cases with the effect newly recorded</span>
      <span><i class="sw-ai"></i>of which AI-flagged medication-associated ${AI_CHIP}</span>
      <span><i class="sw-doc"></i>Doctor-reviewed cases with the effect ${DOC_CHIP}</span></div>${rows}
      <p class="muted small" style="margin-top:12px">Bars are scaled to the number of cases that could be assessed for that effect. A case can have more than one effect.</p>`;
  }

  /* ---------- Trend chart ---------- */
  function trendChart(trend) {
    const pts = trend.points || [];
    if (!pts.length) return "";
    const series = [
      { key: "cases", label: "Cases with an AI prediction", color: "var(--ink-3)" },
      { key: "ai_flagged", label: "AI-flagged medication-associated", color: "var(--lav-ink)" },
      { key: "doctor_reviewed", label: "Doctor-reviewed", color: "var(--ok-ink)" },
    ];
    const W = 680, H = 220, PL = 40, PR = 30, PT = 12, PB = 34;
    const max = Math.max(1, ...pts.flatMap((p) => series.map((s) => cellValue(p[s.key]) || 0)));
    const x = (i) => PL + (pts.length === 1 ? (W - PL - PR) / 2 : (i / (pts.length - 1)) * (W - PL - PR));
    const y = (v) => PT + (1 - v / max) * (H - PT - PB);
    const grid = [0, Math.round(max / 2), max].map((g) =>
      `<line x1="${PL}" x2="${W - PR}" y1="${y(g)}" y2="${y(g)}" stroke="var(--line)"/><text x="${PL - 6}" y="${y(g) + 4}" text-anchor="end" font-size="11" fill="var(--ink-3)">${g}</text>`).join("");
    const labels = pts.map((p, i) => (pts.length > 12 && i % 2 ? "" :
      `<text x="${x(i).toFixed(1)}" y="${H - 12}" text-anchor="middle" font-size="11" fill="var(--ink-3)">${esc(p.period)}</text>`)).join("");
    const lines = series.map((s) => {
      let d = "", pen = false;
      const dots = [];
      pts.forEach((p, i) => {
        const v = cellValue(p[s.key]);
        if (v == null) {
          pen = false;
          if (p[s.key] && p[s.key].suppressed) dots.push(`<circle cx="${x(i).toFixed(1)}" cy="${y(0)}" r="3.5" fill="var(--surface)" stroke="${s.color}" stroke-width="1.5"><title>${esc(s.label)}, ${esc(p.period)}: fewer than ${MIN} patients, withheld</title></circle>`);
          return;
        }
        d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)} `; pen = true;
        dots.push(`<circle cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="3" fill="${s.color}"><title>${esc(s.label)}, ${esc(p.period)}: ${v}</title></circle>`);
      });
      return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.75" stroke-linejoin="round"/>${dots.join("")}`;
    }).join("");
    const legend = series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join("");
    const table = `<details style="margin-top:12px"><summary class="small muted" style="cursor:pointer">View as table</summary>
      <div class="table-wrap"><table class="mfr-table"><thead><tr><th>Period</th><th class="num">Cases</th><th class="num">AI-flagged</th><th class="num">Doctor-reviewed</th></tr></thead><tbody>${
        pts.map((p) => `<tr><td>${esc(p.period)}</td><td class="num">${fmtCell(p.cases)}</td><td class="num">${fmtCell(p.ai_flagged)}</td><td class="num">${fmtCell(p.doctor_reviewed)}</td></tr>`).join("")
      }</tbody></table></div></details>`;
    return `<div class="legend">${legend}</div><div class="trend-wrap">
      <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Cases per ${esc(trend.interval)} by date of the assessed medication event, ${pts.length} periods">${grid}${lines}${labels}</svg></div>
      <p class="muted small">Periods are based on the date of the medication start or dose change each assessment is anchored to. Open circles mark periods with fewer than ${MIN} patients, which are withheld; lines break there.</p>${table}`;
  }

  /* ---------- Page ---------- */
  function notices(summary) {
    const L = summary.labels || {};
    return `<div class="notice info"><b>How to read this page.</b>
      <ul><li>${AI_CHIP} ${esc(L.ai || NOT_CONFIRMED)}</li>
      <li>${DOC_CHIP} ${esc(L.doctor || "")}</li>
      <li>${esc(L.privacy || "")}</li></ul></div>`;
  }

  function card(id, title, sub, body) {
    return `<section class="card" aria-labelledby="${id}"><div class="card-head"><div><h2 id="${id}">${esc(title)}</h2><p class="muted small">${sub}</p></div></div>${body}</section>`;
  }

  function emptyState(e) {
    const titles = { no_data: "No signals yet", no_match: "Nothing matches these filters", below_threshold: "Not enough cases to show" };
    return `<div class="card"><h2>${esc(titles[e.state] || "No data")}</h2><p class="muted" style="margin-top:6px">${esc(e.message)}</p></div>`;
  }

  function dashboard(s) {
    if (s.empty) return `<div class="stack">${emptyState(s.empty)}${notices(s)}</div>`;
    const noReviews = cellValue(s.totals.doctor_reviewed) === 0;
    return `<div class="stack">
      ${notices(s)}
      ${kpis(s.totals)}
      ${card("dr-h", "Drug-wise safety signals", `${AI_CHIP} and ${DOC_CHIP} are reported separately. ${esc(NOT_CONFIRMED)}`,
        drugTable(s.drugs) + (noReviews ? `<p class="muted small" style="margin-top:12px">No doctor reviews are linked to these cases yet.</p>` : ""))}
      ${card("ef-h", "Cognitive and behavioral effects", esc((s.labels || {}).effects || ""), effectsChart(s.effects) || `<p class="empty">No effect data.</p>`)}
      ${card("tr-h", "Trend over time", "By date of the assessed medication event", trendChart(s.trend) || `<p class="empty">No trend data.</p>`)}
      <p class="muted small">Reviews that are not linked to a stored AI prediction cannot be attributed to a case and are not included.</p>
    </div>`;
  }

  function errorNotice(err) {
    return `<div class="notice error"><b>${esc(err.title)}.</b> ${esc(err.message)}</div>`;
  }

  return { MIN, esc, fmtCell, cellValue, buildQuery, validateFilters, describeApiError, dashboard, errorNotice, emptyState, trendChart, effectsChart, drugTable, kpis, NOT_CONFIRMED };
});

// Run with: node --test tests/js
const test = require("node:test");
const assert = require("node:assert/strict");
const M = require("../../app/static/js/mfr.js");

const C = (v) => ({ value: v, suppressed: false });
const S = { value: null, suppressed: true };
const labels = { ai: "AI-generated pattern flags. Not confirmed adverse effects.", doctor: "Doctor assessments.", effects: "Recorded observations only.", privacy: "Groups with fewer than 5 patients are withheld." };

const summary = {
  empty: null, min_group_size: 5, labels,
  totals: { cases: C(20), ai_flagged: C(9), doctor_reviewed: C(7), drug_groups_shown: 2, drug_groups_withheld: 1 },
  drugs: [
    { medication_id: "M001", drug_name: "Donepezil <b>", drug_class: "ChEI",
      ai: { cases: C(14), flagged: C(9), flagged_share: 0.643, classes: { shown: { medication_associated: C(9), other_cause: C(5) }, other_combined: null, hidden_classes: [] } },
      doctor: { reviewed_cases: C(7), likely_medication_associated: C(7), review_coverage: 0.5, categories: null } },
    { medication_id: "M002", drug_name: "Sertraline", drug_class: "SSRI",
      ai: { cases: C(6), flagged: S, flagged_share: null, classes: { shown: {}, other_combined: S, hidden_classes: ["uncertain"] } },
      doctor: { reviewed_cases: S, likely_medication_associated: S, review_coverage: null, categories: null } },
  ],
  effects: [{ effect: "confusion", label: "Confusion", cases_with_effect: C(14), cases_assessable: C(20), rate: 0.7, ai_flagged_with_effect: C(9), doctor_reviewed_with_effect: S, doctor_likely_medication_associated: S }],
  trend: { interval: "quarter", points: [
    { period: "2022-Q1", cases: C(14), ai_flagged: C(9), doctor_reviewed: C(7) },
    { period: "2022-Q2", cases: S, ai_flagged: S, doctor_reviewed: S },
    { period: "2022-Q3", cases: C(6), ai_flagged: C(0), doctor_reviewed: C(0) } ] },
};

test("suppressed cells render as <5 and never as a number", () => {
  assert.match(M.fmtCell(S), /&lt;5/);
  assert.doesNotMatch(M.fmtCell(S), />\d+</);
  assert.match(M.fmtCell(C(0)), />0</, "zero is reported as 0");
  assert.match(M.fmtCell(null), /n\/a/);
  assert.equal(M.cellValue(S), null);
});

test("dashboard separates AI predictions from doctor assessments and states they are not confirmed", () => {
  const h = M.dashboard(summary);
  assert.match(h, /AI prediction/);
  assert.match(h, /Doctor assessment/);
  assert.match(h, /not confirmed adverse effects/i);
  assert.match(h, /Drug-wise safety signals/);
  assert.match(h, /Cognitive and behavioral effects/);
  assert.match(h, /Trend over time/);
  assert.doesNotMatch(h, /confirmed (adverse )?(effect|event|reaction)s? (of|caused)/i);
});

test("drug names are escaped and withheld drugs/groups are only counted", () => {
  const h = M.dashboard(summary);
  assert.match(h, /Donepezil &lt;b&gt;/);
  assert.doesNotMatch(h, /Donepezil <b>/);
  assert.match(h, /Drug groups shown \/ withheld/);
});

test("suppressed table cells show <5 in the drug table", () => {
  const t = M.drugTable(summary.drugs);
  assert.ok((t.match(/&lt;5/g) || []).length >= 3);
  assert.match(t, /not shown|Other classes combined/);
});

test("trend chart draws gaps, marks withheld periods and offers a table", () => {
  const h = M.trendChart(summary.trend);
  assert.match(h, /<svg/);
  assert.match(h, /fewer than 5 patients, withheld/);
  assert.match(h, /View as table/);
  assert.match(h, /2022-Q2/);
  // the withheld period contributes no plotted value
  assert.doesNotMatch(h, /2022-Q2: \d/);
});

test("effects chart shows all three bars per effect with labels", () => {
  const h = M.effectsChart(summary.effects);
  assert.match(h, /Confusion/);
  assert.match(h, /70\.0% of assessable cases/);
  assert.match(h, /AI-flagged/);
  assert.match(h, /Doctor-reviewed/);
  assert.equal(M.effectsChart([]), "");
});

test("empty states are explicit and show no charts or numbers", () => {
  for (const state of ["no_data", "no_match", "below_threshold"]) {
    const h = M.dashboard({ empty: { state, message: "Because reasons." }, labels, totals: {}, drugs: [], effects: [], trend: { points: [] } });
    assert.match(h, /Because reasons\./);
    assert.doesNotMatch(h, /<svg|class="kpi/);
  }
  assert.match(M.emptyState({ state: "no_data", message: "m" }), /No signals yet/);
});

test("a dashboard with no reviews says so", () => {
  const s = JSON.parse(JSON.stringify(summary));
  s.totals.doctor_reviewed = C(0);
  assert.match(M.dashboard(s), /No doctor reviews are linked/);
});

test("filter query building and validation", () => {
  assert.equal(M.buildQuery({ drug: "M001", effect: "", from: "2022-01-01", interval: "month" }), "drug=M001&from=2022-01-01&interval=month");
  assert.equal(M.buildQuery({}), "interval=quarter");
  assert.equal(M.validateFilters({ from: "2023-01-01", to: "2022-01-01" }).ok, false);
  assert.equal(M.validateFilters({ from: "2022-01-01", to: "2022-01-01" }).ok, true);
  assert.equal(M.describeApiError(0).title, "Can't reach the server");
  assert.match(M.describeApiError(400, { message: "bad" }).message, /bad/);
});

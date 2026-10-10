// Run with: node --test tests/js
const test = require("node:test");
const assert = require("node:assert/strict");
const L = require("../../app/static/js/logic.js");
const R = require("../../app/static/js/render.js");

const meds = [
  { medication_record_id: "a", medication_id: "M001", drug_name: "Donepezil", drug_class: "ChEI", start_date: "2021-01-01", dose: 5, dose_unit: "mg", dose_change_date: "2021-06-01", new_dose: 10, stop_date: null, medication_status: "Active", adherence_status: "Unknown" },
  { medication_record_id: "b", medication_id: "M002", drug_name: "Zed", drug_class: "X", start_date: null, dose_change_date: null, dose: null },
];
const success = {
  status: "success", predicted_class: "medication_associated", pattern_description: "Possible medication-associated pattern",
  probabilities: { medication_associated: 0.5, disease_progression: 0.2, other_cause: 0.2, uncertain: 0.1 }, model_name: "XGBoost (Tabular Baseline)",
  anchor_details: { medication_id: "M001", drug_name: "Donepezil", drug_class: "ChEI", anchor_date: "2021-06-01", anchor_type: "dose_change", dose_mg: 5, previous_dose: 5, new_dose: 10, dose_unit: "mg", cutoff_date: "2021-11-28", horizon_days: 180 },
  usable_observation_dates: { pre_anchor: ["2021-05-01"], post_anchor: ["2021-07-01", "2021-08-01"], cutoff_date: "2021-11-28" },
  data_quality_warnings: ["Example warning <b>x</b>"], supporting_evidence: ["MMSE changed by -3.0 points."],
};
const insufficient = { status: "insufficient_data", predicted_class: null, pattern_description: "Insufficient longitudinal data for reliable assessment",
  probabilities: { medication_associated: 0, disease_progression: 0, other_cause: 0, uncertain: 0 }, model_name: "X", anchor_details: null,
  usable_observation_dates: { pre_anchor: [], post_anchor: [] }, data_quality_warnings: ["No baseline."], supporting_evidence: [] };

test("patient ID validation normalizes and rejects bad input", () => {
  assert.deepEqual(L.normalizePatientId("  v4_2_p0001 "), { ok: true, id: "V4_2_P0001" });
  assert.equal(L.normalizePatientId("").ok, false);
  assert.equal(L.normalizePatientId("a b").ok, false);
  assert.equal(L.normalizePatientId("x'; DROP").ok, false);
});

test("medication options: one per id, undated medications disabled", () => {
  const o = L.medicationOptions(meds);
  assert.equal(o.length, 2);
  assert.equal(o[0].disabled, false);
  assert.equal(o[1].disabled, true);
  assert.match(o[1].label, /no start or dose-change date/);
});

test("prediction request omits medication_id when auto, defaults to xgboost", () => {
  assert.deepEqual(L.buildPredictionRequest("P", "", undefined), { patient_id: "P", model_type: "xgboost" });
  assert.deepEqual(L.buildPredictionRequest("P", "M001", "cnn_gru"), { patient_id: "P", model_type: "cnn_gru", medication_id: "M001" });
});

test("API errors map to actionable messages", () => {
  assert.match(L.describeApiError(0).title, /reach/);
  assert.match(L.describeApiError(404, { entity: "medication", identifier: "M9" }).message, /M9/);
  assert.match(L.describeApiError(503, { message: "model missing" }).message, /model missing/);
  assert.match(L.describeApiError(404, { message: "No patient 'X'" }).title, /Patient not found/);
});

test("timeline filter keeps chronological order and groups types", () => {
  const t = [
    { date: "2021-01-01", entry_type: "medication_start" }, { date: "2021-02-01", entry_type: "cognitive_assessment" },
    { date: "2021-03-01", entry_type: "clinical_event" }, { date: "2021-04-01", entry_type: "clinical_event_resolution" },
  ];
  assert.equal(L.filterTimeline(t, "all").length, 4);
  assert.equal(L.filterTimeline(t, "event").length, 2);
  assert.deepEqual(L.filterTimeline(t, "cognitive").map((e) => e.date), ["2021-02-01"]);
});

test("missing-data notes only state what is absent", () => {
  const notes = L.missingDataNotes({ cognitive: [], medications: [], events: [], medical_history: { primary_diagnosis: "AD" } });
  assert.equal(notes.length, 3);
  const none = L.missingDataNotes({ cognitive: [{ mmse_score: 20 }], medications: meds.slice(0, 1), events: [{}], medical_history: { primary_diagnosis: "AD" } });
  assert.deepEqual(none, []);
});

test("MMSE series skips null scores and reports how many", () => {
  const s = L.mmseSeries([{ assessment_date: "2021-02-01", mmse_score: 20 }, { assessment_date: "2021-01-01", mmse_score: 22 }, { assessment_date: "2021-03-01", mmse_score: null }]);
  assert.deepEqual(s.points.map((p) => p.date), ["2021-01-01", "2021-02-01"]);
  assert.equal(s.missing, 1);
});

test("success result shows class, all four scores, anchor, evidence, warnings and the labels", () => {
  const h = R.assessmentResult(success);
  assert.match(h, /Possible medication-associated pattern/);
  for (const k of L.CLASS_ORDER) assert.match(h, new RegExp(L.classMeta(k).label));
  assert.equal((h.match(/class="score-row/g) || []).length, 4);
  assert.match(h, /50\.0%/);
  assert.match(h, /not calibrated probabilities/);
  assert.match(h, /does not show that any medication caused/);
  assert.match(h, /Donepezil/);
  assert.match(h, /Previous dose<\/dt><dd>5 mg/);
  assert.match(h, /New dose<\/dt><dd>10 mg/);
  assert.doesNotMatch(h, /starting dose, not the new dose/);
  assert.match(h, /MMSE changed by -3\.0 points/);
  assert.match(h, /Example warning &lt;b&gt;x&lt;\/b&gt;/, "warnings are escaped");
});

test("insufficient data shows no scores and no predicted class", () => {
  const h = R.assessmentResult(insufficient);
  assert.match(h, /Insufficient data/);
  assert.match(h, /No baseline\./);
  assert.doesNotMatch(h, /score-row/);
  assert.doesNotMatch(h, /0\.0%/);
});

test("missing dose is shown as not recorded, never as 0 mg", () => {
  const r = JSON.parse(JSON.stringify(success));
  r.anchor_details.dose_mg = 0; r.anchor_details.previous_dose = null; r.anchor_details.new_dose = null;
  const h = R.assessmentResult(r);
  assert.doesNotMatch(h, /0 mg/);
  assert.match(h, /not recorded/);
});

test("unexpected status renders an error, not a prediction", () => {
  assert.match(R.assessmentResult({ status: "weird" }), /Unexpected response/);
});

test("timeline renderer shows missing values honestly and escapes notes", () => {
  const entries = [
    { date: "2021-01-01", entry_type: "cognitive_assessment", summary: "Cognitive/behavioral assessment", details: { mmse_score: null, memory_score: 50, confusion_present: null, memory_loss_present: null, attention_problem_present: null, behavioral_change_present: null } },
    { date: "2021-02-01", entry_type: "clinical_event", summary: "Clinical event: Fall", details: { severity: "Mild", clinical_note: "<script>alert(1)</script>", duration_days: null } },
  ];
  const h = R.timeline(entries, "all", 2);
  assert.match(h, /not recorded/);
  assert.match(h, /symptom flags not recorded/);
  assert.doesNotMatch(h, /<script>/);
  assert.match(h, /does not imply sequence/);
});

test("empty timeline, medications and chart give plain empty states", () => {
  assert.match(R.timeline([], "all", 0), /No dated records/);
  assert.match(R.medicationTable([], ""), /No medication history/);
  assert.match(R.mmseChart([]), /No MMSE scores/);
});

test("review validation requires one of four options and caps the note", () => {
  assert.equal(L.validateReview({ decision: "" }).ok, false);
  assert.equal(L.validateReview({ decision: "uncertain" }).ok, true);
  assert.equal(L.validateReview({ decision: "uncertain", note: "x".repeat(501) }).ok, false);
  assert.equal(L.REVIEW_OPTIONS.length, 4);
});

test("review request carries category, note and the prediction/medication reference", () => {
  const withPred = Object.assign({}, success, { prediction_id: 7 });
  const req = L.buildReviewRequest({ patientId: "P1", decision: "other_cause", note: " hello ", assessment: withPred });
  assert.deepEqual(req, { patient_id: "P1", review_category: "other_cause", clinical_note: "hello", medication_id: "M001", prediction_id: 7 });
  // no assessment: only the selected medication (if any) is referenced, never a prediction
  const bare = L.buildReviewRequest({ patientId: "P1", decision: "uncertain", assessment: null, selectedMedicationId: "M002" });
  assert.equal(bare.prediction_id, undefined);
  assert.equal(bare.medication_id, "M002");
  assert.equal(L.buildReviewRequest({ patientId: "P1", decision: "uncertain" }).medication_id, undefined);
  // an insufficient-data result never yields a prediction reference
  assert.equal(L.reviewTarget(insufficient, "").prediction_id, null);
});

test("saved review is labelled as a doctor assessment, separate from the AI prediction", () => {
  const saved = { review_id: 3, patient_id: "P1", review_category: "other_cause", clinical_note: "<i>n</i>", reviewed_at: "2026-10-10T00:00:00Z" };
  const msg = R.reviewSaved(saved);
  assert.match(msg, /Doctor assessment saved/);
  assert.doesNotMatch(msg, /not saved|page session/);
  assert.match(msg, /&lt;i&gt;n&lt;\/i&gt;/, "note is escaped");
  const list = R.reviewList([Object.assign({}, saved, {
    medication: { medication_id: "M001", drug_name: "Donepezil" },
    prediction_reference: { prediction_id: 7, predicted_class_at_review: "medication_associated", model_name: "XGBoost", anchor_date: "2021-06-01" } })]);
  assert.match(list, /Doctor assessment/);
  assert.match(list, /AI prediction/);
  assert.match(list, /Likely other cause/);
  assert.match(list, /Medication-associated pattern/);
  assert.match(R.reviewList([]), /No doctor assessments/);
});

test("dose-change anchor shows previous and new dose; missing values are not recorded, never 0", () => {
  const r = JSON.parse(JSON.stringify(success));
  r.anchor_details.new_dose = null;
  const h = R.assessmentResult(r);
  assert.match(h, /Previous dose<\/dt><dd>5 mg/);
  assert.match(h, /New dose<\/dt><dd><span class="missing">not recorded/);
  const start = JSON.parse(JSON.stringify(success));
  Object.assign(start.anchor_details, { anchor_type: "start", new_dose: null });
  const hs = R.assessmentResult(start);
  assert.match(hs, /Dose on record/);
  assert.doesNotMatch(hs, /New dose/);
});

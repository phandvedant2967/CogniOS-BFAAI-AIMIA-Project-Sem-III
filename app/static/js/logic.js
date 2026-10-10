/* CogniOS dashboard: pure logic (no DOM, no fetch). Works in the browser and in Node tests. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.CogniLogic = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const ID_RE = /^[A-Za-z0-9_-]{1,40}$/;
  const NOTE_MAX = 500;

  const MODELS = [
    { value: "xgboost", label: "XGBoost" },
    { value: "cnn_gru", label: "CNN-GRU" },
  ];
  const DEFAULT_MODEL = "xgboost";

  // Order matches the API's TARGET_CLASSES.
  const CLASS_ORDER = ["medication_associated", "disease_progression", "other_cause", "uncertain"];
  const CLASS_META = {
    medication_associated: { label: "Medication-associated pattern", tone: "rose" },
    disease_progression: { label: "Disease-progression pattern", tone: "blue" },
    other_cause: { label: "Other-cause pattern", tone: "lav" },
    uncertain: { label: "Uncertain pattern", tone: "sand" },
  };

  const REVIEW_OPTIONS = [
    { value: "medication_associated", label: "Likely medication-associated" },
    { value: "disease_progression", label: "Likely disease progression" },
    { value: "other_cause", label: "Likely other cause" },
    { value: "uncertain", label: "Uncertain" },
  ];

  function classMeta(key) {
    return CLASS_META[key] || { label: String(key), tone: "plain" };
  }

  function normalizePatientId(input) {
    const id = String(input == null ? "" : input).trim();
    if (!id) return { ok: false, message: "Enter a patient ID." };
    if (!ID_RE.test(id)) {
      return { ok: false, message: "Patient IDs contain only letters, numbers, underscores and hyphens." };
    }
    return { ok: true, id: id.toUpperCase() };
  }

  /** One option per distinct medication_id. Only episodes with a start or dose-change date can anchor an assessment. */
  function medicationOptions(medications) {
    const byId = new Map();
    for (const m of medications || []) {
      if (!byId.has(m.medication_id)) byId.set(m.medication_id, []);
      byId.get(m.medication_id).push(m);
    }
    const out = [];
    for (const [id, rows] of byId) {
      const usable = rows.filter((r) => r.start_date || r.dose_change_date);
      const first = rows[0];
      const dates = rows.map((r) => r.start_date).filter(Boolean).sort();
      let label = `${first.drug_name} (${id})`;
      if (dates.length) label += ` · started ${dates[0]}`;
      if (rows.length > 1) label += ` · ${rows.length} records`;
      if (!usable.length) label += " · no start or dose-change date";
      out.push({ value: id, label, disabled: usable.length === 0 });
    }
    return out;
  }

  const TYPE_GROUP = {
    cognitive_assessment: "cognitive",
    medication_start: "medication",
    medication_dose_change: "medication",
    medication_stop: "medication",
    clinical_event: "event",
    clinical_event_resolution: "event",
    diagnosis: "history",
    enrollment: "history",
  };
  const FILTERS = [
    { value: "all", label: "All" },
    { value: "cognitive", label: "Assessments" },
    { value: "medication", label: "Medications" },
    { value: "event", label: "Events" },
  ];

  function filterTimeline(entries, filter) {
    if (!filter || filter === "all") return entries.slice();
    return entries.filter((e) => TYPE_GROUP[e.entry_type] === filter);
  }

  function mmseSeries(cognitive) {
    const points = [];
    let missing = 0;
    for (const c of cognitive || []) {
      if (c.mmse_score == null || !c.assessment_date) missing++;
      else points.push({ date: c.assessment_date, value: Number(c.mmse_score) });
    }
    points.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
    return { points, missing };
  }

  /** Plain statements about what is absent. Nothing is inferred. */
  function missingDataNotes(bundle) {
    const notes = [];
    const c = bundle.cognitive || [];
    if (!c.length) notes.push("No cognitive assessments are recorded.");
    else {
      const noMmse = c.filter((x) => x.mmse_score == null).length;
      if (noMmse) notes.push(`${noMmse} of ${c.length} assessments have no MMSE score recorded.`);
    }
    if (!(bundle.medications || []).length) notes.push("No medication history is recorded.");
    else {
      const noDates = bundle.medications.filter((m) => !m.start_date && !m.dose_change_date).length;
      if (noDates) notes.push(`${noDates} medication record(s) have no start or dose-change date.`);
      const noStart = bundle.medications.filter((m) => !m.start_date).length;
      if (noStart && noStart !== noDates) notes.push(`${noStart} medication record(s) have no start date.`);
    }
    if (!(bundle.events || []).length) notes.push("No clinical events are recorded.");
    const mh = bundle.medical_history || {};
    if (!mh.primary_diagnosis) notes.push("No primary diagnosis is recorded.");
    return notes;
  }

  function buildPredictionRequest(patientId, medicationId, modelType) {
    const body = { patient_id: patientId, model_type: modelType || DEFAULT_MODEL };
    if (medicationId) body.medication_id = medicationId;
    return body;
  }

  /** Turn an HTTP failure into a message a clinician can act on. */
  function describeApiError(status, body) {
    const b = body || {};
    if (status === 0) {
      return { title: "Can't reach the server", message: "Check that the CogniOS server is running, then try again." };
    }
    if (status === 400) {
      return { title: "Request not accepted", message: b.message || "The request was not valid." };
    }
    if (status === 404) {
      if (b.entity === "prediction") {
        return { title: "Prediction not found", message: "The AI prediction this review refers to is no longer stored. Run the assessment again." };
      }
      if (b.entity === "medication") {
        return { title: "Medication not found", message: `${b.identifier} is not in this patient's medication history.` };
      }
      return { title: "Patient not found", message: b.message || `No patient '${b.identifier || ""}' was found.` };
    }
    if (status === 503) {
      return { title: "Assessment service unavailable", message: b.message || "The model could not be loaded." };
    }
    return { title: "Something went wrong", message: b.message || `The server returned status ${status}.` };
  }

  function fmtProb(p) {
    if (typeof p !== "number" || !isFinite(p)) return "n/a";
    return (p * 100).toFixed(1) + "%";
  }

  function validateReview(review) {
    const r = review || {};
    if (!REVIEW_OPTIONS.some((o) => o.value === r.decision)) {
      return { ok: false, message: "Select one of the four review options." };
    }
    if ((r.note || "").length > NOTE_MAX) {
      return { ok: false, message: `Keep the note to ${NOTE_MAX} characters or fewer.` };
    }
    return { ok: true };
  }

  /** Which medication / stored prediction a review should point at. Only a successful assessment has a prediction_id. */
  function reviewTarget(assessment, selectedMedicationId) {
    const ok = assessment && assessment.status === "success" && assessment.anchor_details;
    return {
      medication_id: ok ? assessment.anchor_details.medication_id : (selectedMedicationId || null),
      prediction_id: ok && Number.isInteger(assessment.prediction_id) ? assessment.prediction_id : null,
    };
  }

  /** Body for POST /api/reviews. The server sets the timestamp. */
  function buildReviewRequest(ctx) {
    const t = reviewTarget(ctx.assessment, ctx.selectedMedicationId);
    const body = { patient_id: ctx.patientId, review_category: ctx.decision, clinical_note: (ctx.note || "").trim() };
    if (t.medication_id) body.medication_id = t.medication_id;
    if (t.prediction_id != null) body.prediction_id = t.prediction_id;
    return body;
  }

  return {
    NOTE_MAX, MODELS, DEFAULT_MODEL, CLASS_ORDER, REVIEW_OPTIONS, FILTERS,
    classMeta, normalizePatientId, medicationOptions, filterTimeline, mmseSeries,
    missingDataNotes, buildPredictionRequest, describeApiError, fmtProb,
    validateReview, reviewTarget, buildReviewRequest,
  };
});

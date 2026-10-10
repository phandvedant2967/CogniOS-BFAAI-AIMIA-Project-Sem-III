/* CogniOS dashboard: pure renderers. Each returns an HTML string; all data is escaped. */
(function (root, factory) {
  const L = typeof module === "object" && module.exports ? require("./logic.js") : root.CogniLogic;
  if (typeof module === "object" && module.exports) module.exports = factory(L);
  else root.CogniRender = factory(L);
})(typeof self !== "undefined" ? self : this, function (L) {
  "use strict";

  const PROB_NOTE = "Model-predicted probabilities. They are not calibrated probabilities.";
  const NON_CAUSAL_NOTE =
    "This is a pattern flag from a statistical model for clinician review. It does not show that any medication caused a change.";

  function esc(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  const missing = (label) => `<span class="missing">${esc(label || "not recorded")}</span>`;
  const val = (v, suffix) => (v == null || v === "" ? missing() : esc(v) + (suffix ? " " + esc(suffix) : ""));

  /* ---------- Patient header ---------- */
  function patientHeader(b) {
    const p = b.patient, mh = b.medical_history;
    const como = (mh.comorbidities || []).length ? esc(mh.comorbidities.join(", ")) : missing("none recorded");
    const notes = L.missingDataNotes(b);
    return `
    <section class="card" aria-labelledby="pt-h">
      <div class="patient-head">
        <div>
          <h2 id="pt-h">${esc(p.patient_id)}</h2>
          <p class="muted">${val(p.age, "years")} · ${val(p.sex)}</p>
        </div>
        <div class="stats">
          <div class="stat"><div class="v num">${b.counts.cognitive}</div><div class="muted small">Assessments</div></div>
          <div class="stat"><div class="v num">${b.counts.medications}</div><div class="muted small">Medication records</div></div>
          <div class="stat"><div class="v num">${b.counts.events}</div><div class="muted small">Clinical events</div></div>
        </div>
      </div>
      <dl class="kv" style="margin-top:16px">
        <dt>Primary diagnosis</dt><dd>${val(mh.primary_diagnosis)}</dd>
        <dt>Disease stage</dt><dd>${val(mh.disease_stage)}</dd>
        <dt>Diagnosis date</dt><dd class="num">${val(mh.diagnosis_date)}</dd>
        <dt>Enrollment date</dt><dd class="num">${val(mh.enrollment_date)}</dd>
        <dt>Comorbidities</dt><dd>${como}</dd>
      </dl>
      ${notes.length ? `<div class="notice warn" style="margin-top:16px"><b>Missing in the record</b><ul>${notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></div>` : ""}
    </section>`;
  }

  /* ---------- Medication table ---------- */
  function medicationTable(meds, selectedId) {
    if (!meds.length) return `<p class="empty">No medication history is recorded for this patient.</p>`;
    const rows = meds.map((m) => {
      const usable = m.start_date || m.dose_change_date;
      const dose = m.dose == null ? missing() : esc(m.dose) + " " + esc(m.dose_unit || "");
      const change = m.dose_change_date
        ? `${esc(m.dose_change_date)}${m.new_dose != null ? ` → ${esc(m.new_dose)} ${esc(m.dose_unit || "")}` : ""}`
        : missing("none recorded");
      const sel = selectedId === m.medication_id;
      return `<tr class="${usable ? "selectable" : ""} ${sel ? "is-selected" : ""}" data-med="${esc(m.medication_id)}" ${usable ? "" : 'aria-disabled="true"'}>
        <td><b>${esc(m.drug_name)}</b><div class="muted small">${esc(m.medication_id)} · ${esc(m.drug_class)}</div></td>
        <td class="num">${dose}</td>
        <td class="num">${val(m.start_date)}</td>
        <td class="num">${change}</td>
        <td class="num">${val(m.stop_date, "")}</td>
        <td>${val(m.medication_status)}<div class="muted small">Adherence: ${val(m.adherence_status)}</div></td>
      </tr>`;
    }).join("");
    return `<div class="table-wrap"><table>
      <thead><tr><th>Medication</th><th>Dose</th><th>Started</th><th>Dose change</th><th>Stopped</th><th>Status</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="muted small" style="margin-top:8px">Select a row to choose that medication for assessment.</p>`;
  }

  /* ---------- MMSE chart ---------- */
  function mmseChart(cognitive) {
    const { points, missing: miss } = L.mmseSeries(cognitive);
    if (points.length < 2) {
      return `<p class="empty">${points.length ? "Only one MMSE score is recorded, so no trend is drawn." : "No MMSE scores are recorded."}</p>`;
    }
    const W = 640, H = 170, PL = 34, PR = 10, PT = 10, PB = 24;
    const t = (d) => new Date(d + "T00:00:00Z").getTime();
    const t0 = t(points[0].date), t1 = t(points[points.length - 1].date) || t0 + 1;
    const vals = points.map((p) => p.value);
    let lo = Math.floor(Math.min(...vals) - 1), hi = Math.ceil(Math.max(...vals) + 1);
    if (hi === lo) hi = lo + 1;
    const x = (d) => PL + ((t(d) - t0) / Math.max(t1 - t0, 1)) * (W - PL - PR);
    const y = (v) => PT + (1 - (v - lo) / (hi - lo)) * (H - PT - PB);
    const path = points.map((p, i) => `${i ? "L" : "M"}${x(p.date).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
    const mid = Math.round((lo + hi) / 2);
    const grid = [lo, mid, hi].map((g) =>
      `<line x1="${PL}" x2="${W - PR}" y1="${y(g)}" y2="${y(g)}" stroke="var(--line)"/><text x="${PL - 6}" y="${y(g) + 4}" text-anchor="end" font-size="11" fill="var(--ink-3)">${g}</text>`).join("");
    const dots = points.map((p) => `<circle cx="${x(p.date).toFixed(1)}" cy="${y(p.value).toFixed(1)}" r="3" fill="var(--accent)"><title>${esc(p.date)}: MMSE ${esc(p.value)}</title></circle>`).join("");
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="MMSE scores over time, ${points.length} assessments from ${esc(points[0].date)} to ${esc(points[points.length - 1].date)}">
      ${grid}<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="1.75" stroke-linejoin="round"/>${dots}
      <text x="${PL}" y="${H - 6}" font-size="11" fill="var(--ink-3)">${esc(points[0].date)}</text>
      <text x="${W - PR}" y="${H - 6}" font-size="11" text-anchor="end" fill="var(--ink-3)">${esc(points[points.length - 1].date)}</text></svg>
      <p class="muted small">MMSE only. The line joins recorded scores and does not imply values between assessments.${miss ? ` ${miss} assessment(s) without an MMSE score are not drawn.` : ""}</p>`;
  }

  /* ---------- Timeline ---------- */
  const TYPE_LABEL = {
    cognitive_assessment: ["Assessment", "blue"],
    medication_start: ["Medication start", "lav"],
    medication_dose_change: ["Dose change", "lav"],
    medication_stop: ["Medication stop", "lav"],
    clinical_event: ["Event", "sand"],
    clinical_event_resolution: ["Event resolved", "plain"],
    diagnosis: ["Diagnosis", "plain"],
    enrollment: ["Enrollment", "plain"],
  };
  const item = (k, v) => `<span><b>${esc(k)}:</b> ${v}</span>`;
  const yesNo = (v) => (v == null ? missing() : v === 1 || v === "Yes" ? "Yes" : "No");

  function timelineDetails(e) {
    const d = e.details || {};
    switch (e.entry_type) {
      case "cognitive_assessment": {
        const flags = [["confusion_present", "confusion"], ["memory_loss_present", "memory loss"],
          ["attention_problem_present", "attention problem"], ["behavioral_change_present", "behavioral change"]];
        const unknown = flags.filter(([k]) => d[k] == null).length;
        const present = flags.filter(([k]) => d[k] === 1).map(([, n]) => n);
        let flagText;
        if (unknown === flags.length) flagText = missing("symptom flags not recorded");
        else flagText = (present.length ? esc(present.join(", ")) : "none marked present") +
          (unknown ? ` ${missing("(" + unknown + " flag(s) not recorded)")}` : "");
        return [item("MMSE", val(d.mmse_score)), item("Memory", val(d.memory_score)), item("Attention", val(d.attention_score)),
          item("Executive", val(d.executive_function_score)), item("Mood", val(d.mood_score)), item("Symptom flags", flagText)].join("");
      }
      case "medication_start":
        return [item("Indication", val(d.indication)), item("Status", val(d.medication_status)), item("Adherence", val(d.adherence_status))].join("");
      case "medication_dose_change":
        return item("Dose", `${val(d.dose, d.dose_unit)} → ${val(d.new_dose, d.dose_unit)}`);
      case "medication_stop":
        return item("Status", val(d.medication_status));
      case "clinical_event":
        return [item("Severity", val(d.severity)), item("Duration", val(d.duration_days, "days")),
          item("Cognitive impact possible (as recorded)", val(d.cognitive_impact_possible)),
          item("Functional impact", val(d.functional_impact)), item("Hospitalization", yesNo(d.hospitalization_required)),
          item("Note", val(d.clinical_note))].join("");
      default:
        return "";
    }
  }

  function timeline(entries, filter, total) {
    const rows = L.filterTimeline(entries, filter);
    const chips = L.FILTERS.map((f) =>
      `<button type="button" class="chip ${f.value === (filter || "all") ? "ok" : "plain"}" data-filter="${f.value}" aria-pressed="${f.value === (filter || "all")}" style="border:0;cursor:pointer">${esc(f.label)}</button>`).join("");
    let body;
    if (!entries.length) body = `<p class="empty">No dated records are available for this patient.</p>`;
    else if (!rows.length) body = `<p class="empty">No ${esc((L.FILTERS.find((f) => f.value === filter) || {}).label || "")} records for this patient.</p>`;
    else body = `<ol class="timeline">${rows.map((e) => {
      const [label, tone] = TYPE_LABEL[e.entry_type] || [e.entry_type, "plain"];
      return `<li class="tl-item"><div class="tl-date">${esc(e.date)}</div><div>
        <div class="tl-title"><span class="chip ${tone}">${esc(label)}</span><span>${esc(e.summary)}</span></div>
        <div class="tl-detail">${timelineDetails(e)}</div></div></li>`;
    }).join("")}</ol>`;
    return `<div class="filters" role="group" aria-label="Filter timeline">${chips}</div>${body}
      <p class="muted small" style="margin-top:12px">Showing ${rows.length} of ${total} records, oldest first. Records on the same date are listed in a fixed order that does not imply sequence. Nothing on this timeline is linked or interpreted.</p>`;
  }

  /* ---------- Assessment result ---------- */
  const ANCHOR_TYPE = { start: "Medication start", dose_change: "Dose change" };

  function anchorBlock(a, usable) {
    if (!a) return `<p class="muted">No medication anchor could be determined.</p>`;
    const pre = (usable && usable.pre_anchor) || [], post = (usable && usable.post_anchor) || [];
    const unit = a.dose_unit ? " " + a.dose_unit : "";
    const dose = (v) => (v == null || v === 0 ? missing() : esc(v) + esc(unit));
    let doseRows;
    if (a.anchor_type === "dose_change" && a.previous_dose !== undefined) {
      // Both values come from the medication record the anchor is based on.
      doseRows = `<dt>Previous dose</dt><dd>${dose(a.previous_dose)}</dd><dt>New dose</dt><dd>${dose(a.new_dose)}</dd>`;
    } else if (a.previous_dose !== undefined) {
      doseRows = `<dt>Dose on record</dt><dd>${dose(a.previous_dose)}</dd>`;
    } else {
      doseRows = `<dt>Dose on record</dt><dd>${a.dose_mg == null || a.dose_mg === 0 ? missing() : esc(a.dose_mg) + " mg"}</dd>`;
    }
    return `<dl class="kv">
      <dt>Medication</dt><dd>${esc(a.drug_name)} <span class="muted">(${esc(a.medication_id)})</span></dd>
      <dt>Drug class</dt><dd>${val(a.drug_class)}</dd>
      <dt>Assessment anchor</dt><dd>${esc(ANCHOR_TYPE[a.anchor_type] || a.anchor_type)} on <span class="num">${esc(a.anchor_date)}</span></dd>
      ${doseRows}
      <dt>Look-ahead window</dt><dd>${esc(a.horizon_days)} days, to <span class="num">${esc(a.cutoff_date)}</span></dd>
      <dt>Assessments before anchor</dt><dd>${pre.length}${pre.length ? ` <span class="muted small num">(${esc(pre.join(", "))})</span>` : ""}</dd>
      <dt>Assessments after anchor</dt><dd>${post.length}${post.length ? ` <span class="muted small num">(${esc(post.join(", "))})</span>` : ""}</dd>
    </dl>`;
  }

  function warningsBlock(list) {
    if (!list || !list.length) return `<p class="muted small">No data-quality warnings were returned.</p>`;
    return `<div class="notice warn"><b>Data-quality warnings</b><ul>${list.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>`;
  }

  function assessmentResult(r) {
    if (r.status === "insufficient_data") {
      return `<div class="stack" style="gap:16px">
        <div class="notice warn"><b>Insufficient data. No assessment was produced.</b><br>${esc(r.pattern_description || "")}</div>
        ${warningsBlock(r.data_quality_warnings)}
        <div><h3 style="margin-bottom:8px">Anchor considered</h3>${anchorBlock(r.anchor_details, r.usable_observation_dates)}</div>
        <p class="muted small">Model: ${esc(r.model_name)}. No scores are shown because none were computed.</p></div>`;
    }
    if (r.status !== "success") {
      return `<div class="notice error"><b>Unexpected response.</b> The server returned status "${esc(r.status)}".</div>`;
    }
    const meta = L.classMeta(r.predicted_class);
    const probs = r.probabilities || {};
    const max = Math.max(...L.CLASS_ORDER.map((k) => probs[k] || 0));
    const rows = L.CLASS_ORDER.map((k) => {
      const p = probs[k];
      return `<div class="score-row ${p === max ? "top" : ""}"><span class="name">${esc(L.classMeta(k).label)}</span>
        <div class="bar" aria-hidden="true"><i style="width:${Math.max(0, Math.min(100, (p || 0) * 100)).toFixed(1)}%"></i></div>
        <span class="num" title="raw value ${esc(p)}">${esc(L.fmtProb(p))}</span></div>`;
    }).join("");
    const ev = (r.supporting_evidence || []).length
      ? `<ul class="evidence">${r.supporting_evidence.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>`
      : `<p class="muted">No supporting evidence was returned.</p>`;
    return `<div class="stack" style="gap:20px">
      <div>
        <span class="chip ${meta.tone}">${esc(r.pattern_description || meta.label)}</span>
        <p class="muted small" style="margin-top:6px">Highest-scoring class · ${esc(r.model_name)}</p>
      </div>
      <div><h3>Model scores</h3><p class="muted small" style="margin-bottom:6px">${PROB_NOTE}</p>${rows}</div>
      <div><h3 style="margin-bottom:8px">Medication and anchor</h3>${anchorBlock(r.anchor_details, r.usable_observation_dates)}</div>
      <div><h3 style="margin-bottom:6px">Supporting evidence</h3><p class="muted small">As returned by the model service.</p>${ev}</div>
      ${warningsBlock(r.data_quality_warnings)}
      <div class="notice info">${NON_CAUSAL_NOTE}</div></div>`;
  }

  function errorNotice(err) {
    return `<div class="notice error"><b>${esc(err.title)}.</b> ${esc(err.message)}</div>`;
  }

  const DOCTOR_CHIP = `<span class="chip ok">Doctor assessment</span>`;
  const AI_CHIP = `<span class="chip lav">AI prediction</span>`;
  const reviewLabel = (v) => (L.REVIEW_OPTIONS.find((o) => o.value === v) || { label: v }).label;

  function reviewSaved(rec) {
    return `<div class="notice info"><b>Doctor assessment saved.</b> It is stored with the time it was recorded and is kept separate from the AI prediction.
      <dl class="kv" style="margin-top:8px"><dt>Selection</dt><dd>${esc(reviewLabel(rec.review_category))}</dd>
      <dt>Note</dt><dd>${rec.clinical_note ? esc(rec.clinical_note) : missing("no note")}</dd>
      <dt>Saved at</dt><dd class="num">${esc(rec.reviewed_at)}</dd></dl></div>`;
  }

  /** Saved doctor assessments for a patient. The AI prediction each refers to is shown on its own, labelled line. */
  function reviewList(reviews) {
    if (!reviews.length) return `<p class="empty">No doctor assessments have been saved for this patient.</p>`;
    return `<ul class="review-list">${reviews.map((r) => {
      const med = r.medication ? `${esc(r.medication.drug_name || "")} <span class="muted">(${esc(r.medication.medication_id)})</span>` : missing("no medication linked");
      const ref = r.prediction_reference;
      const ai = ref
        ? `<div class="small muted">${AI_CHIP} at the time: ${esc(L.classMeta(ref.predicted_class_at_review).label)}${ref.model_name ? ` · ${esc(ref.model_name)}` : ""}${ref.anchor_date ? ` · anchor ${esc(ref.anchor_date)}` : ""}</div>`
        : `<div class="small muted">Not linked to an AI prediction.</div>`;
      return `<li class="review-item"><div>${DOCTOR_CHIP} <b>${esc(reviewLabel(r.review_category))}</b> <span class="muted small num">${esc(r.reviewed_at)}</span></div>
        <div class="small">Medication: ${med}</div>
        <div class="small">${r.clinical_note ? esc(r.clinical_note) : missing("no note")}</div>${ai}</li>`;
    }).join("")}</ul>`;
  }

  return { esc, patientHeader, medicationTable, mmseChart, timeline, assessmentResult, errorNotice, reviewSaved, reviewList, DOCTOR_CHIP, AI_CHIP, PROB_NOTE, NON_CAUSAL_NOTE };
});

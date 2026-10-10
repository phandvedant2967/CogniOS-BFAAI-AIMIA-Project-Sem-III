/* CogniOS dashboard controller: DOM, fetch, state. Logic and markup live in logic.js / render.js. */
(function () {
  "use strict";
  const L = window.CogniLogic, R = window.CogniRender;
  const $ = (id) => document.getElementById(id);

  const state = {
    patientId: null, bundle: null, medicationId: "", model: L.DEFAULT_MODEL,
    filter: "all", assessment: null, assessing: false, lastReview: null, requestSeq: 0,
  };

  async function api(url, options) {
    let res;
    try { res = await fetch(url, options); }
    catch (e) { throw L.describeApiError(0); }
    let body = null;
    try { body = await res.json(); } catch (e) { /* non-JSON error page */ }
    if (!res.ok) throw L.describeApiError(res.status, body);
    return body;
  }

  function setStatus(html) { $("status").innerHTML = html || ""; }

  /* ---------- Patient lookup ---------- */
  async function lookup(rawId) {
    const check = L.normalizePatientId(rawId);
    if (!check.ok) { setStatus(R.errorNotice({ title: "Check the patient ID", message: check.message })); return; }

    const seq = ++state.requestSeq;
    $("lookup-btn").disabled = true;
    setStatus(`<div class="card"><div class="skeleton" style="width:40%"></div><div class="skeleton" style="width:70%;margin-top:12px"></div></div>`);
    try {
      const bundle = await api("/api/dashboard/patients/" + encodeURIComponent(check.id));
      if (seq !== state.requestSeq) return; // a newer lookup superseded this one
      Object.assign(state, {
        patientId: bundle.patient.patient_id, bundle, medicationId: "", model: L.DEFAULT_MODEL,
        filter: "all", assessment: null, lastReview: null,
      });
      setStatus("");
      $("landing").hidden = true;
      drawWorkspace();
    } catch (err) {
      if (seq !== state.requestSeq) return;
      setStatus(R.errorNotice(err));
    } finally {
      if (seq === state.requestSeq) $("lookup-btn").disabled = false;
    }
  }

  /* ---------- Workspace ---------- */
  function drawWorkspace() {
    const b = state.bundle;
    const ws = $("workspace");
    ws.hidden = false;
    ws.innerHTML = `
      <div class="stack">
        ${R.patientHeader(b)}
        <div class="grid-2">
          <div class="stack">
            <section class="card" aria-labelledby="tl-h">
              <div class="card-head"><div><h2 id="tl-h">Timeline</h2><p class="muted small">Recorded events in date order</p></div></div>
              <div id="mmse-chart" style="margin-bottom:16px">${R.mmseChart(b.cognitive)}</div>
              <div id="timeline">${R.timeline(b.timeline, state.filter, b.counts.timeline)}</div>
            </section>
          </div>
          <div class="stack">
            <section class="card" aria-labelledby="as-h">
              <div class="card-head"><div><h2 id="as-h">AI assessment</h2><p class="muted small">Pattern flag for clinician review</p></div>${R.AI_CHIP}</div>
              <div class="stack" style="gap:16px">
                <div class="field"><span class="label" id="model-label">Model</span>
                  <div class="segmented" role="group" aria-labelledby="model-label" id="model-seg"></div></div>
                <div class="field"><label for="med-select">Medication to assess</label>
                  <select class="select" id="med-select"></select></div>
                <button class="btn" type="button" id="assess-btn">Run assessment</button>
                <div id="assess-out" aria-live="polite"></div>
              </div>
            </section>
            <section class="card" aria-labelledby="rv-h">
              <div class="card-head"><div><h2 id="rv-h">Doctor review</h2><p class="muted small">Your clinical judgement on this patient</p></div>${R.DOCTOR_CHIP}</div>
              <form id="review-form" class="stack" style="gap:16px">
                <fieldset class="radio-list"><legend class="sr-only">Review selection</legend>
                  ${L.REVIEW_OPTIONS.map((o) => `<label class="radio"><input type="radio" name="decision" value="${o.value}"> ${R.esc(o.label)}</label>`).join("")}
                </fieldset>
                <div class="field"><label for="note">Clinical note (optional)</label>
                  <textarea class="textarea" id="note" maxlength="${L.NOTE_MAX}" placeholder="Short note"></textarea>
                  <span class="muted small" id="note-count">0 / ${L.NOTE_MAX}</span></div>
                <div id="review-msg" role="status"></div>
                <div style="display:flex;gap:8px;flex-wrap:wrap">
                  <button class="btn" type="submit">Record review</button>
                  <button class="btn secondary" type="button" id="download-btn" disabled>Download as JSON</button>
                </div>
                <p class="muted small" id="review-link-note"></p>
              </form>
              <h3 style="margin:20px 0 8px">Saved doctor assessments</h3>
              <div id="review-list" aria-live="polite"><div class="skeleton" style="width:50%"></div></div>
            </section>
          </div>
        </div>
        <section class="card" aria-labelledby="md-h">
          <div class="card-head"><div><h2 id="md-h">Medication history</h2><p class="muted small">As recorded</p></div></div>
          <div id="med-table">${R.medicationTable(b.medications, state.medicationId)}</div>
        </section>
      </div>`;
    drawModelSeg();
    drawMedSelect();
    drawAssessment();
    drawReviewNote();
    loadReviews();
  }

  function drawReviewNote() {
    const el = $("review-link-note"); if (!el) return;
    const t = L.reviewTarget(state.assessment, state.medicationId);
    el.textContent = t.prediction_id != null
      ? "This review will be saved and linked to the AI prediction shown above. It is stored as a separate doctor assessment."
      : "No AI assessment is linked. The review is saved for this patient only and will not be linked to a prediction.";
  }

  async function loadReviews() {
    const pid = state.patientId;
    try {
      const out = await api("/api/reviews?patient_id=" + encodeURIComponent(pid));
      if (pid !== state.patientId) return;
      state.reviews = out.reviews;
      $("review-list").innerHTML = R.reviewList(out.reviews);
    } catch (err) {
      if (pid === state.patientId && $("review-list")) $("review-list").innerHTML = R.errorNotice(err);
    }
  }

  function drawModelSeg() {
    const seg = $("model-seg");
    seg.innerHTML = L.MODELS.map((m) =>
      `<button type="button" data-model="${m.value}" aria-pressed="${m.value === state.model}">${R.esc(m.label)}${m.value === L.DEFAULT_MODEL ? " (default)" : ""}</button>`).join("");
  }

  function drawMedSelect() {
    const opts = L.medicationOptions(state.bundle.medications);
    $("med-select").innerHTML =
      `<option value="">Let the service choose (default)</option>` +
      opts.map((o) => `<option value="${R.esc(o.value)}" ${o.disabled ? "disabled" : ""} ${o.value === state.medicationId ? "selected" : ""}>${R.esc(o.label)}</option>`).join("");
    $("med-table").innerHTML = R.medicationTable(state.bundle.medications, state.medicationId);
  }

  function drawAssessment(html) {
    $("assess-out").innerHTML = html != null ? html
      : state.assessment ? R.assessmentResult(state.assessment)
      : `<p class="muted small">No assessment has been run for this patient yet.</p>`;
  }

  /* ---------- AI assessment ---------- */
  async function runAssessment() {
    if (state.assessing || !state.patientId) return;
    state.assessing = true;
    const btn = $("assess-btn");
    btn.disabled = true; btn.textContent = "Running…";
    drawAssessment(`<div class="skeleton" style="width:60%"></div><div class="skeleton" style="width:85%;margin-top:10px"></div>`);
    const patientAtStart = state.patientId;
    try {
      const result = await api("/api/predictions", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(L.buildPredictionRequest(patientAtStart, state.medicationId, state.model)),
      });
      if (patientAtStart !== state.patientId) return;
      state.assessment = result;
      drawAssessment();
      drawReviewNote();
    } catch (err) {
      if (patientAtStart !== state.patientId) return;
      state.assessment = null;
      drawAssessment(R.errorNotice(err));
      drawReviewNote();
    } finally {
      state.assessing = false;
      const b = $("assess-btn");
      if (b) { b.disabled = false; b.textContent = "Run assessment"; }
    }
  }

  /* ---------- Review (saved to the server) ---------- */
  async function submitReview(form) {
    const decision = (form.querySelector('input[name="decision"]:checked') || {}).value;
    const note = $("note").value;
    const v = L.validateReview({ decision, note });
    if (!v.ok) { $("review-msg").innerHTML = R.errorNotice({ title: "Review not recorded", message: v.message }); return; }
    const btn = form.querySelector('button[type="submit"]');
    btn.disabled = true;
    const patientAtStart = state.patientId;
    try {
      const saved = await api("/api/reviews", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(L.buildReviewRequest({
          patientId: patientAtStart, decision, note, assessment: state.assessment, selectedMedicationId: state.medicationId })),
      });
      if (patientAtStart !== state.patientId) return;
      state.lastReview = saved;
      $("review-msg").innerHTML = R.reviewSaved(saved);
      $("download-btn").disabled = false;
      loadReviews();
    } catch (err) {
      if (patientAtStart !== state.patientId) return;
      $("review-msg").innerHTML = R.errorNotice({ title: "Review not saved", message: err.message });
    } finally {
      btn.disabled = false;
    }
  }

  function downloadReview() {
    if (!state.lastReview) return;
    const blob = new Blob([JSON.stringify(state.lastReview, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `cognios-review-${state.lastReview.patient_id}-${state.lastReview.review_id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  /* ---------- Events ---------- */
  $("lookup-form").addEventListener("submit", (e) => { e.preventDefault(); lookup($("patient-id").value); });

  const ws = $("workspace");
  ws.addEventListener("click", (e) => {
    const modelBtn = e.target.closest("[data-model]");
    if (modelBtn) { state.model = modelBtn.dataset.model; drawModelSeg(); return; }
    const filterBtn = e.target.closest("[data-filter]");
    if (filterBtn) { state.filter = filterBtn.dataset.filter; $("timeline").innerHTML = R.timeline(state.bundle.timeline, state.filter, state.bundle.counts.timeline); return; }
    const row = e.target.closest("tr.selectable[data-med]");
    if (row) { state.medicationId = row.dataset.med; drawMedSelect(); drawReviewNote(); return; }
    if (e.target.closest("#assess-btn")) runAssessment();
    if (e.target.closest("#download-btn")) downloadReview();
  });
  ws.addEventListener("change", (e) => {
    if (e.target.id === "med-select") { state.medicationId = e.target.value; $("med-table").innerHTML = R.medicationTable(state.bundle.medications, state.medicationId); drawReviewNote(); }
  });
  ws.addEventListener("input", (e) => {
    if (e.target.id === "note") $("note-count").textContent = `${e.target.value.length} / ${L.NOTE_MAX}`;
  });
  ws.addEventListener("submit", (e) => { if (e.target.id === "review-form") { e.preventDefault(); submitReview(e.target); } });
})();

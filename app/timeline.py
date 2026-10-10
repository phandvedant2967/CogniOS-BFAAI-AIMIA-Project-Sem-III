"""Chronological patient timeline.

Merges every dated record for one patient into a single list sorted by date.
This is a plain merge of what is stored: no interpretation, no inferred
relationships between entries. Entries that fall on the same date are ordered
by a fixed display order (see _ORDER) and then by record id; that order is
only a tie-break and does NOT imply that one entry came before another.
"""
from . import queries

_ORDER = {
    "diagnosis": 0,
    "enrollment": 1,
    "medication_start": 2,
    "medication_dose_change": 3,
    "medication_stop": 4,
    "cognitive_assessment": 5,
    "clinical_event": 6,
    "clinical_event_resolution": 7,
}


def _fmt(v):
    """12.0 -> '12', 2.5 -> '2.5'."""
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _entry(date, entry_type, source_table, source_id, summary, details):
    return {
        "date": date,
        "entry_type": entry_type,
        "source_table": source_table,
        "source_id": source_id,
        "summary": summary,
        "details": details,
    }


def build_patient_timeline(conn, patient_id):
    """Return a patient's records as one date-ordered list, or None if the patient is unknown."""
    patient = queries.get_patient(conn, patient_id)
    if patient is None:
        return None

    entries = []

    if patient["diagnosis_date"]:
        entries.append(_entry(
            patient["diagnosis_date"], "diagnosis", "patients", patient_id,
            f"Diagnosis recorded: {patient['primary_diagnosis']}",
            {"primary_diagnosis": patient["primary_diagnosis"],
             "disease_stage": patient["disease_stage"]},
        ))
    if patient["enrollment_date"]:
        entries.append(_entry(
            patient["enrollment_date"], "enrollment", "patients", patient_id,
            "Enrolled in study", {},
        ))

    for m in queries.get_medication_history(conn, patient_id):
        rid, name = m["medication_record_id"], m["drug_name"]
        unit = m["dose_unit"] or ""
        if m["start_date"]:
            dose = f" {_fmt(m['dose'])} {unit}".rstrip() if m["dose"] is not None else ""
            entries.append(_entry(
                m["start_date"], "medication_start", "medication_history", rid,
                f"Medication start: {name}{dose}"
                + (f", {m['frequency']}" if m["frequency"] else ""),
                m,
            ))
        if m["dose_change_date"]:
            new = f" to {_fmt(m['new_dose'])} {unit}".rstrip() if m["new_dose"] is not None else ""
            entries.append(_entry(
                m["dose_change_date"], "medication_dose_change", "medication_history", rid,
                f"Dose change: {name}{new}", m,
            ))
        if m["stop_date"]:
            entries.append(_entry(
                m["stop_date"], "medication_stop", "medication_history", rid,
                f"Medication stop: {name}", m,
            ))

    for c in queries.get_cognitive_history(conn, patient_id):
        if c["assessment_date"]:
            mmse = f" (MMSE {_fmt(c['mmse_score'])})" if c["mmse_score"] is not None else ""
            entries.append(_entry(
                c["assessment_date"], "cognitive_assessment", "cognitive_behavioral",
                c["cognitive_record_id"], f"Cognitive/behavioral assessment{mmse}", c,
            ))

    for e in queries.get_clinical_events(conn, patient_id):
        eid = e["clinical_event_id"]
        if e["event_date"]:
            sev = f" ({e['severity']})" if e["severity"] else ""
            entries.append(_entry(
                e["event_date"], "clinical_event", "clinical_events", eid,
                f"Clinical event: {e['event_type']}{sev}", e,
            ))
        if e["resolution_date"]:
            entries.append(_entry(
                e["resolution_date"], "clinical_event_resolution", "clinical_events", eid,
                f"Clinical event resolution date: {e['event_type']}", e,
            ))

    entries.sort(key=lambda x: (x["date"], _ORDER[x["entry_type"]], x["source_id"]))
    return entries

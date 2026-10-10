"""Plain data-access functions. Each takes an open sqlite3 connection.

Used by both the API routes and the timeline builder, so the ML step
can import them directly without going through HTTP.
"""
from .db import rows_to_dicts


def get_patient(conn, patient_id):
    row = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
    return dict(row) if row else None


def get_comorbidities(conn, patient_id):
    rows = conn.execute(
        "SELECT comorbidity FROM patient_comorbidities WHERE patient_id = ? ORDER BY comorbidity",
        (patient_id,),
    ).fetchall()
    return [r["comorbidity"] for r in rows]


def get_medication_history(conn, patient_id):
    """Medication episodes joined to drug name/class/area, ordered by start date."""
    rows = conn.execute(
        """
        SELECT m.medication_record_id, m.patient_id, m.medication_id,
               d.drug_name, d.drug_class, d.therapeutic_area,
               m.indication, m.dose, m.dose_unit, m.frequency,
               m.start_date, m.dose_change_date, m.new_dose, m.stop_date,
               m.medication_status, m.adherence_status
        FROM medication_history m
        JOIN drugs d ON d.medication_id = m.medication_id
        WHERE m.patient_id = ?
        ORDER BY m.start_date, m.medication_record_id
        """,
        (patient_id,),
    ).fetchall()
    return rows_to_dicts(rows)


def get_cognitive_history(conn, patient_id):
    rows = conn.execute(
        "SELECT * FROM cognitive_behavioral WHERE patient_id = ? "
        "ORDER BY assessment_date, cognitive_record_id",
        (patient_id,),
    ).fetchall()
    return rows_to_dicts(rows)


def get_clinical_events(conn, patient_id):
    rows = conn.execute(
        "SELECT * FROM clinical_events WHERE patient_id = ? ORDER BY event_date, clinical_event_id",
        (patient_id,),
    ).fetchall()
    return rows_to_dicts(rows)


def get_drug(conn, identifier):
    """Look up by medication_id (e.g. 'M005') or drug name (case-insensitive)."""
    row = conn.execute(
        "SELECT * FROM drugs WHERE medication_id = ? COLLATE NOCASE OR drug_name = ? COLLATE NOCASE",
        (identifier, identifier),
    ).fetchone()
    return dict(row) if row else None


def list_drugs(conn):
    return rows_to_dicts(conn.execute("SELECT * FROM drugs ORDER BY medication_id").fetchall())

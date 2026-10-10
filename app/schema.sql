-- SQLite schema. Dates are stored as ISO-8601 TEXT (YYYY-MM-DD).
-- Missing values are stored as NULL. Source IDs are used as primary keys.
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS patient_comorbidities;
DROP TABLE IF EXISTS clinical_events;
DROP TABLE IF EXISTS cognitive_behavioral;
DROP TABLE IF EXISTS medication_history;
DROP TABLE IF EXISTS drugs;
DROP TABLE IF EXISTS patients;

-- Dataset 1
CREATE TABLE patients (
    patient_id        TEXT PRIMARY KEY,
    age               INTEGER,
    sex               TEXT,
    primary_diagnosis TEXT,
    diagnosis_date    TEXT,
    disease_stage     TEXT,
    comorbidities     TEXT,          -- original "A; B; C" string, kept verbatim
    enrollment_date   TEXT
);

-- Dataset 1 (comorbidities split into rows; same values, one per row)
CREATE TABLE patient_comorbidities (
    patient_id  TEXT NOT NULL REFERENCES patients(patient_id),
    comorbidity TEXT NOT NULL,
    PRIMARY KEY (patient_id, comorbidity)
);

-- Dataset 5 (+ Dataset 2 "medication_reference" sheet, which is identical
-- to the id/name/class/area columns here)
CREATE TABLE drugs (
    medication_id              TEXT PRIMARY KEY,
    drug_name                  TEXT NOT NULL,
    drug_class                 TEXT,
    therapeutic_area           TEXT,
    mechanism_summary          TEXT,
    cognitive_relevance        TEXT,
    memory_effect_potential    TEXT,
    confusion_effect_potential TEXT,
    attention_effect_potential TEXT,
    mood_effect_potential      TEXT,
    sedation_potential         TEXT,
    anticholinergic_burden     TEXT,   -- 'None' is a real category in the source
    interaction_risk           TEXT,
    cognitive_risk_index       INTEGER,
    sedation_index             INTEGER,
    anticholinergic_index      INTEGER,
    interaction_risk_index     INTEGER,
    evidence_category          TEXT
);

-- Dataset 2 (drug_name / drug_class / therapeutic_area live in `drugs`)
CREATE TABLE medication_history (
    medication_record_id TEXT PRIMARY KEY,
    patient_id           TEXT NOT NULL REFERENCES patients(patient_id),
    medication_id        TEXT NOT NULL REFERENCES drugs(medication_id),
    indication           TEXT,
    dose                 REAL,
    dose_unit            TEXT,
    frequency            TEXT,
    start_date           TEXT,
    dose_change_date     TEXT,
    new_dose             REAL,
    stop_date            TEXT,
    medication_status    TEXT,
    adherence_status     TEXT
);

-- Dataset 3
CREATE TABLE cognitive_behavioral (
    cognitive_record_id          TEXT PRIMARY KEY,
    patient_id                   TEXT NOT NULL REFERENCES patients(patient_id),
    assessment_date              TEXT,
    mmse_score                   REAL,
    memory_score                 REAL,
    attention_score              REAL,
    executive_function_score     REAL,
    mood_score                   REAL,
    confusion_present            INTEGER CHECK (confusion_present IN (0,1)),
    memory_loss_present          INTEGER CHECK (memory_loss_present IN (0,1)),
    attention_problem_present    INTEGER CHECK (attention_problem_present IN (0,1)),
    behavioral_change_present    INTEGER CHECK (behavioral_change_present IN (0,1)),
    trajectory_context           TEXT   -- source label, stored verbatim ('None' is a source value)
);

-- Dataset 4
CREATE TABLE clinical_events (
    clinical_event_id         TEXT PRIMARY KEY,
    patient_id                TEXT NOT NULL REFERENCES patients(patient_id),
    event_date                TEXT,
    event_type                TEXT,
    event_category            TEXT,
    severity                  TEXT,
    duration_days             INTEGER,
    resolution_date           TEXT,
    cognitive_impact_possible TEXT,
    functional_impact         TEXT,
    hospitalization_required  TEXT,
    clinical_note             TEXT
);

CREATE INDEX idx_comorb_patient ON patient_comorbidities(patient_id);
CREATE INDEX idx_med_patient    ON medication_history(patient_id, start_date);
CREATE INDEX idx_med_drug       ON medication_history(medication_id);
CREATE INDEX idx_cog_patient    ON cognitive_behavioral(patient_id, assessment_date);
CREATE INDEX idx_event_patient  ON clinical_events(patient_id, event_date);

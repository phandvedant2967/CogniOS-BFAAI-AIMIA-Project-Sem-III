"""Load the 5 source datasets into SQLite.

Rules:
  * Nothing is invented or inferred. Empty cells become NULL.
  * No rows are dropped (row counts are verified after loading).
  * Dates are standardized to ISO YYYY-MM-DD text.
  * Source category values are kept verbatim (including the literal
    string 'None' in drug anticholinergic_burden / trajectory_context).
"""
import os
import sqlite3

import pandas as pd

import config

# Read every cell as text and treat ONLY empty cells as missing, so that
# real values such as the category 'None' are not silently turned into NaN.
_READ_KW = dict(dtype=str, keep_default_na=False, na_values=[""])


def _read_csv(path):
    return pd.read_csv(path, **_READ_KW)


def _read_sheet(path, sheet):
    return pd.read_excel(path, sheet_name=sheet, **_READ_KW)


# ---- column cleaners -------------------------------------------------------
def _strip(df):
    for col in df.columns:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        df[col] = df[col].replace("", pd.NA)
    return df


def _dates(df, cols, warnings, table):
    """Standardize to YYYY-MM-DD. Accepts date or 'YYYY-MM-DD HH:MM:SS' input."""
    for col in cols:
        before = df[col].notna().sum()
        parsed = pd.to_datetime(df[col], errors="coerce")
        lost = before - parsed.notna().sum()
        if lost:
            warnings.append(f"{table}.{col}: {lost} unparseable date(s) stored as NULL")
        df[col] = parsed.dt.strftime("%Y-%m-%d").astype(object)
        df.loc[parsed.isna(), col] = None
    return df


def _numbers(df, cols, warnings, table, integer=False):
    for col in cols:
        before = df[col].notna().sum()
        num = pd.to_numeric(df[col], errors="coerce")
        lost = before - num.notna().sum()
        if lost:
            warnings.append(f"{table}.{col}: {lost} non-numeric value(s) stored as NULL")
        if integer:
            num = num.astype("Int64")
        df[col] = num.astype(object)
        df.loc[num.isna(), col] = None
    return df


def _insert(conn, table, df):
    df = df.astype(object).where(df.notna(), None)
    cols = list(df.columns)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    conn.executemany(sql, df.itertuples(index=False, name=None))
    return len(df)


# ---- main ------------------------------------------------------------------
def import_all(db_path=None, raw_dir=None, reset=True):
    db_path = db_path or config.DB_PATH
    raw_dir = raw_dir or config.RAW_DIR
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if reset and os.path.exists(db_path):
        os.remove(db_path)

    source_files = {
        "patients": os.path.join(raw_dir, "dataset1_Patients.csv"),
        "medication_history": os.path.join(raw_dir, "dataset2_Medication_History.xlsx"),
        "cognitive_behavioral": os.path.join(raw_dir, "dataset3_Cognitive_Behavioral.xlsx"),
        "clinical_events": os.path.join(raw_dir, "dataset4_Clinical_Events.xlsx"),
        "drug_knowledge": os.path.join(raw_dir, "dataset5_Drug_Knowledge_Base.xlsx"),
    }

    warnings, report = [], {}
    conn = sqlite3.connect(db_path)
    try:
        with open(config.SCHEMA_PATH, encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.execute("PRAGMA foreign_keys = ON")

        # Dataset 5 (+ medication_reference sheet from Dataset 2) -> drugs
        drugs = _strip(_read_sheet(source_files["drug_knowledge"], "drug_knowledge"))
        drugs = _numbers(drugs, ["cognitive_risk_index", "sedation_index",
                                 "anticholinergic_index", "interaction_risk_index"],
                         warnings, "drugs", integer=True)
        report["drugs"] = (len(drugs), _insert(conn, "drugs", drugs))

        ref = _strip(_read_sheet(source_files["medication_history"], "medication_reference"))
        chk = ref.merge(drugs[["medication_id", "drug_name", "drug_class", "therapeutic_area"]],
                        on="medication_id", how="outer", suffixes=("", "_kb"), indicator=True)
        bad = chk[(chk["_merge"] != "both") |
                  (chk.drug_name != chk.drug_name_kb) |
                  (chk.drug_class != chk.drug_class_kb) |
                  (chk.therapeutic_area != chk.therapeutic_area_kb)]
        if len(bad):
            warnings.append(f"medication_reference differs from drug_knowledge for {len(bad)} drug(s)")

        # Dataset 1 -> patients (+ comorbidities split)
        pts = _strip(_read_csv(source_files["patients"]))
        pts = _numbers(pts, ["age"], warnings, "patients", integer=True)
        pts = _dates(pts, ["diagnosis_date", "enrollment_date"], warnings, "patients")
        report["patients"] = (len(pts), _insert(conn, "patients", pts))

        como = pts[["patient_id", "comorbidities"]].dropna().copy()
        como["comorbidity"] = como["comorbidities"].str.split(";")
        como = como.explode("comorbidity")
        como["comorbidity"] = como["comorbidity"].str.strip()
        # The literal source value 'None' means "no comorbidities listed", not a condition.
        # It stays verbatim in patients.comorbidities but is not turned into a comorbidity row.
        como = como[como["comorbidity"].notna() & ~como["comorbidity"].isin(["", "None"])]
        como = como[["patient_id", "comorbidity"]].drop_duplicates()
        report["patient_comorbidities"] = (len(como), _insert(conn, "patient_comorbidities", como))

        # Dataset 2 -> medication_history
        med = _strip(_read_sheet(source_files["medication_history"], "medication_history"))
        med = med.drop(columns=["drug_name", "drug_class", "therapeutic_area"])  # live in `drugs`
        med = _numbers(med, ["dose", "new_dose"], warnings, "medication_history")
        med = _dates(med, ["start_date", "dose_change_date", "stop_date"], warnings, "medication_history")
        report["medication_history"] = (len(med), _insert(conn, "medication_history", med))

        # Dataset 3 -> cognitive_behavioral
        cog = _strip(_read_sheet(source_files["cognitive_behavioral"], "cognitive_behavioral"))
        cog = _numbers(cog, ["mmse_score", "memory_score", "attention_score",
                             "executive_function_score", "mood_score"],
                       warnings, "cognitive_behavioral")
        cog = _numbers(cog, ["confusion_present", "memory_loss_present",
                             "attention_problem_present", "behavioral_change_present"],
                       warnings, "cognitive_behavioral", integer=True)
        cog = _dates(cog, ["assessment_date"], warnings, "cognitive_behavioral")
        report["cognitive_behavioral"] = (len(cog), _insert(conn, "cognitive_behavioral", cog))

        # Dataset 4 -> clinical_events
        ev = _strip(_read_sheet(source_files["clinical_events"], "clinical_events"))
        ev = _numbers(ev, ["duration_days"], warnings, "clinical_events", integer=True)
        ev = _dates(ev, ["event_date", "resolution_date"], warnings, "clinical_events")
        report["clinical_events"] = (len(ev), _insert(conn, "clinical_events", ev))

        conn.commit()

        # Integrity checks
        fk = conn.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            warnings.append(f"{len(fk)} foreign-key violation(s) found")
        for table, (src, loaded) in report.items():
            if src != loaded:
                warnings.append(f"{table}: source rows {src} != loaded rows {loaded}")
    finally:
        conn.close()

    return report, warnings

"""Central configuration. Override with environment variables if needed."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# SQLite database file (primary REST API – original dataset)
DB_PATH = os.environ.get("EHR_DB_PATH", os.path.join(BASE_DIR, "data", "ehr.db"))

# SQLite database used exclusively by the ML inference pipeline (V4.2 dataset)
ML_DB_PATH = os.environ.get("EHR_ML_DB_PATH", os.path.join(BASE_DIR, "data", "v4_2", "ehr.db"))

# Source datasets (original files, untouched)
RAW_DIR = os.environ.get("EHR_RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
SCHEMA_PATH = os.path.join(BASE_DIR, "app", "schema.sql")

SOURCE_FILES = {
    "patients": os.path.join(RAW_DIR, "dataset1_Patients.csv"),
    "medication_history": os.path.join(RAW_DIR, "dataset2_Medication_History.xlsx"),
    "cognitive_behavioral": os.path.join(RAW_DIR, "dataset3_Cognitive_Behavioral.xlsx"),
    "clinical_events": os.path.join(RAW_DIR, "dataset4_Clinical_Events.xlsx"),
    "drug_knowledge": os.path.join(RAW_DIR, "dataset5_Drug_Knowledge_Base.xlsx"),
}

# Application database (doctor reviews + logged AI predictions). Kept separate from the
# read-only EHR databases so no dataset is ever modified. Created on first use.
APP_DB_PATH = os.environ.get("COGNIOS_APP_DB_PATH", os.path.join(BASE_DIR, "data", "cognios_app.db"))

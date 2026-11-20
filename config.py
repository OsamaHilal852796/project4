"""Central settings. Change values here, not in the ML or UI code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage"          # trained models + subject data live here
SEED_FILE = ROOT / "data" / "seed_data.xlsx"
STORAGE.mkdir(exist_ok=True)

# --- Column conventions of the seed file (used only as defaults in the mapping UI)
DEFAULT_ID = "student_ID"
DEFAULT_NAME = "name"
DEFAULT_SUBJECT = "Subject"
DEFAULT_FINAL = "final"              # numeric target 1 : final exam score
DEFAULT_TOTAL = "final_total"        # numeric target 2 : total grade
IGNORE_HINTS = ("final_result", "result", "status", "pass", "grade_letter")
CONTACT_COLS = ["student_email", "guardian_email", "guardian_phone"]

# --- Risk classes (multi-class target). Labels are built from the total grade
#     as a PERCENTAGE of its max; a model then learns to predict them.
RISK_LABELS = ["Low", "Moderate", "High", "Critical"]      # order = ascending severity
RISK_CUTS_PCT = [50, 60, 75]   # <50 Critical | 50-59.99 High | 60-74.99 Moderate | >=75 Low

# --- ML
PCA_VARIANCE = 0.95            # keep enough components to explain 95% of variance
TEST_SIZE = 0.2
CV_FOLDS = 5
SEED = 42
MIN_ROWS_TO_TRAIN = 30

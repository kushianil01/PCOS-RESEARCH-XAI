"""Single source of truth for paths, seeds, feature definitions and plausibility ranges."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_XLSX = ROOT / "data/raw/kerala_kaggle/extracted/PCOS_data_without_infertility.xlsx"
RAW_ZIP = ROOT / "data/raw/kerala_kaggle/pcos.zip"
RAW_SHEET = "Full_new"
PROCESSED = ROOT / "data/processed/kerala_pcos_clean.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
PREDS = RESULTS / "predictions"
FIGS = RESULTS / "figures"
LOGS = RESULTS / "logs"
SPLIT_FILE = RESULTS / "metrics/split_definition.json"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 5          # model selection repeated stratified CV on the training split
NESTED_REPEATS = 3      # secondary nested CV on all data
N_BOOT = 2000
TARGET = "pcos"
TARGET_SENSITIVITY = 0.85   # screening-oriented operating point (chosen on train OOF predictions only)

# raw (stripped) column name -> canonical name
RAW_TO_CANON = {
    "Sl. No": "row_id", "PCOS (Y/N)": "pcos", "Age (yrs)": "age_years", "Weight (Kg)": "weight_kg",
    "Height(Cm)": "height_cm", "BMI": "bmi_reported", "Blood Group": "blood_group", "Pulse rate(bpm)": "pulse_bpm",
    "RR (breaths/min)": "resp_rate", "Hb(g/dl)": "hb_g_dl", "Cycle(R/I)": "cycle_code_raw",
    "Cycle length(days)": "cycle_length_days", "Marraige Status (Yrs)": "marriage_years", "Pregnant(Y/N)": "pregnant",
    "No. of aborptions": "abortions", "I   beta-HCG(mIU/mL)": "bhcg_1", "II    beta-HCG(mIU/mL)": "bhcg_2",
    "FSH(mIU/mL)": "fsh", "LH(mIU/mL)": "lh", "FSH/LH": "fsh_lh", "Hip(inch)": "hip_inch", "Waist(inch)": "waist_inch",
    "Waist:Hip Ratio": "whr", "TSH (mIU/L)": "tsh", "AMH(ng/mL)": "amh", "PRL(ng/mL)": "prl", "Vit D3 (ng/mL)": "vit_d3",
    "PRG(ng/mL)": "prg", "RBS(mg/dl)": "rbs", "Weight gain(Y/N)": "weight_gain", "hair growth(Y/N)": "hair_growth",
    "Skin darkening (Y/N)": "skin_darkening", "Hair loss(Y/N)": "hair_loss", "Pimples(Y/N)": "pimples",
    "Fast food (Y/N)": "fast_food", "Reg.Exercise(Y/N)": "regular_exercise", "BP _Systolic (mmHg)": "bp_sys",
    "BP _Diastolic (mmHg)": "bp_dia", "Follicle No. (L)": "follicle_l", "Follicle No. (R)": "follicle_r",
    "Avg. F size (L) (mm)": "fsize_l_mm", "Avg. F size (R) (mm)": "fsize_r_mm", "Endometrium (mm)": "endometrium_mm",
}
DROP_RAW = ["Patient File No.", "Unnamed: 44"]

# ---- feature groups -------------------------------------------------------------------------------------------
SYMPTOMS = ["weight_gain", "hair_growth", "skin_darkening", "hair_loss", "pimples"]
LIFESTYLE = ["fast_food", "regular_exercise"]
BINARY = ["cycle_irregular"] + SYMPTOMS + LIFESTYLE
CONTINUOUS_BASE = ["age_years", "height_cm", "weight_kg"]
# Primary (non-invasive) MODEL features. Chosen by experiments/02_feature_redundancy.py (results/metrics/feature_set_decision.json;
# tests/test_consistency.py asserts this list equals that decision). BMI is derived from height and weight.
PRIMARY_FEATURES = ["age_years", "bmi_derived", "cycle_irregular"] + SYMPTOMS + LIFESTYLE
# User-facing INPUT fields (what the application asks for)
INPUT_FIELDS = ["age_years", "height_cm", "weight_kg", "cycle_irregular"] + SYMPTOMS + LIFESTYLE
LAB = ["fsh", "lh", "fsh_lh", "tsh", "amh", "prl", "prg", "vit_d3", "rbs", "hb_g_dl"]
ULTRASOUND = ["follicle_l", "follicle_r", "fsize_l_mm", "fsize_r_mm", "endometrium_mm"]
EXCLUDED_BY_DESIGN = ["pregnant", "marriage_years", "abortions", "blood_group", "hb_g_dl", "pulse_bpm", "resp_rate",
                      "bp_sys", "bp_dia", "cycle_length_days", "bhcg_1", "bhcg_2"]

# Physiological plausibility ranges used for INPUT validation in the app (hard limits; not training-range limits).
INPUT_RANGES = {"age_years": (12, 65), "height_cm": (120, 210), "weight_kg": (25, 250)}

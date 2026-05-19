"""Paths and tariff defaults for the Industrial Energy Optimizer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AI4I_CSV = ROOT / "ai4i+2020+predictive+maintenance+dataset" / "ai4i2020.csv"
CCPP_XLSX = ROOT / "combined+cycle+power+plant" / "CCPP" / "Folds5x2_pp.xlsx"

MODELS_DIR = ROOT / "models"
REG_MODEL_PATH = MODELS_DIR / "demand_forecast_rf.pkl"
CLF_MODEL_PATH = MODELS_DIR / "anomaly_classifier_rf.pkl"

# Industrial tariff (₹/kWh) — editable in dashboard sidebar
DEFAULT_TARIFF_PER_KWH = 8.0
PEAK_HOUR_MULTIPLIER = 1.25
PEAK_HOURS = list(range(18, 22))  # 6 PM – 10 PM illustrative peak window

# Map CCPP electrical output (MW scale in dataset) to billing kWh proxy
MWH_TO_KWH_SCALE = 1000.0

# Wastage heuristics (software-only, no sensors)
IDLE_TORQUE_RATIO = 0.35
ANOMALY_RISK_THRESHOLD = 0.55
WASTAGE_PENALTY_FACTOR = 0.18

RANDOM_STATE = 42

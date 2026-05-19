"""Load, clean, and normalize CCPP + AI4I datasets."""
from __future__ import annotations

import pandas as pd

from energy_optimizer.config import AI4I_CSV, CCPP_XLSX

CCPP_RENAME = {
    "AT": "Temp",
    "V": "Vacuum",
    "AP": "Pressure",
    "RH": "Humidity",
    "PE": "EnergyOutput",
}

AI4I_RENAME = {
    "Air temperature [K]": "AirTemp",
    "Process temperature [K]": "ProcTemp",
    "Rotational speed [rpm]": "RPM",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "ToolWear",
}

CLF_FEATURE_COLS = [
    "Type",
    "AirTemp",
    "ProcTemp",
    "RPM",
    "Torque",
    "ToolWear",
]

REG_FEATURE_COLS = ["Temp", "Vacuum", "Pressure", "Humidity"]


def load_ai4i(path=AI4I_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=AI4I_RENAME)
    df["Type"] = df["Type"].astype(str)
    return df


def load_ccpp(path=CCPP_XLSX) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df.rename(columns=CCPP_RENAME)


def prepare_ccpp(df: pd.DataFrame | None = None) -> pd.DataFrame:
    data = load_ccpp() if df is None else df.copy()
    data = data.dropna(subset=REG_FEATURE_COLS + ["EnergyOutput"])
    for col in REG_FEATURE_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["EnergyOutput"] = pd.to_numeric(data["EnergyOutput"], errors="coerce")
    return data.dropna()


def prepare_ai4i(df: pd.DataFrame | None = None) -> pd.DataFrame:
    data = load_ai4i() if df is None else df.copy()
    data = data.dropna(subset=CLF_FEATURE_COLS)
    data["Machine failure"] = data["Machine failure"].astype(int)
    return data


def clf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features aligned with trained classifier (original column names)."""
    out = df.copy()
    out = out.rename(
        columns={
            "AirTemp": "Air temperature [K]",
            "ProcTemp": "Process temperature [K]",
            "RPM": "Rotational speed [rpm]",
            "Torque": "Torque [Nm]",
            "ToolWear": "Tool wear [min]",
        }
    )
    return out[
        [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
    ]

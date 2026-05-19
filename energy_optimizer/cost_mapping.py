"""Convert energy proxies to ₹ and estimate machine-wise wastage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from energy_optimizer.config import (
    DEFAULT_TARIFF_PER_KWH,
    MWH_TO_KWH_SCALE,
    PEAK_HOUR_MULTIPLIER,
    PEAK_HOURS,
)
from energy_optimizer.wastage_rules import WastageRules, PRESETS


def kwh_from_ccpp_output(energy_output_mw: pd.Series | np.ndarray) -> pd.Series:
    """Dataset PE is plant net output (MW); scale to kWh billing proxy per hour."""
    return pd.Series(energy_output_mw) * MWH_TO_KWH_SCALE


def rupees_from_kwh(
    kwh: pd.Series | np.ndarray,
    tariff: float = DEFAULT_TARIFF_PER_KWH,
    hour: int | None = None,
) -> pd.Series:
    cost = pd.Series(kwh, copy=True) * tariff
    if hour is not None and hour in PEAK_HOURS:
        cost *= PEAK_HOUR_MULTIPLIER
    return cost


# Rated hourly draw by machine class (kWh) — maps AI4I types to industrial load tiers
RATED_KWH_BY_TYPE = {"L": 45.0, "M": 75.0, "H": 120.0}


def estimate_machine_power_kwh(df: pd.DataFrame) -> pd.Series:
    """Proxy hourly kWh from rated load × utilization (torque/RPM vs type baseline)."""
    rated = df["Type"].map(RATED_KWH_BY_TYPE).fillna(60.0)
    torque_util = (df["Torque"] / df.groupby("Type")["Torque"].transform("median").clip(lower=1)).clip(
        0.1, 1.5
    )
    rpm_util = (df["RPM"] / df.groupby("Type")["RPM"].transform("median").clip(lower=1)).clip(0.1, 1.5)
    utilization = (torque_util * rpm_util) / 2.0
    return (rated * utilization).clip(lower=1.0)


def compute_machine_wastage(
    df: pd.DataFrame,
    failure_proba: pd.Series,
    tariff: float = DEFAULT_TARIFF_PER_KWH,
    rules: WastageRules | None = None,
) -> pd.DataFrame:
    rules = rules or PRESETS["Balanced (default)"]
    """
    Machine-wise wastage in ₹:
    - Idle: high RPM, torque below type baseline × ratio
    - Abnormal: elevated failure probability or recorded failure
    """
    work = df.copy()
    work["kwh_proxy"] = estimate_machine_power_kwh(work)
    work["failure_risk"] = failure_proba.values

    type_median_torque = work.groupby("Type")["Torque"].transform("median")
    idle_mask = (work["RPM"] > work.groupby("Type")["RPM"].transform("median") * 0.5) & (
        work["Torque"] < type_median_torque * rules.idle_torque_ratio
    )

    anomaly_mask = (work["failure_risk"] >= rules.anomaly_risk_threshold) | (
        work["Machine failure"] == 1
    )

    baseline_kwh = work.groupby("Type")["kwh_proxy"].transform("median")
    idle_excess = np.where(
        idle_mask,
        (work["kwh_proxy"] - baseline_kwh * rules.idle_baseline_fraction).clip(lower=0),
        0,
    )
    anomaly_excess = np.where(
        anomaly_mask,
        work["kwh_proxy"] * rules.wastage_penalty_factor,
        0,
    )

    scale = rules.demo_wastage_scale
    work["idle_wastage_kwh"] = idle_excess
    work["anomaly_wastage_kwh"] = anomaly_excess
    work["total_wastage_kwh"] = work["idle_wastage_kwh"] + work["anomaly_wastage_kwh"]
    work["idle_wastage_inr"] = work["idle_wastage_kwh"] * tariff * scale
    work["anomaly_wastage_inr"] = work["anomaly_wastage_kwh"] * tariff * scale
    work["total_wastage_inr"] = work["idle_wastage_inr"] + work["anomaly_wastage_inr"]
    work["is_idle"] = idle_mask
    work["is_anomaly"] = anomaly_mask
    if rules.min_wastage_inr_per_machine > 0:
        flagged = idle_mask | anomaly_mask
        low = flagged & (work["total_wastage_inr"] < rules.min_wastage_inr_per_machine)
        work.loc[low, "total_wastage_inr"] = rules.min_wastage_inr_per_machine
    return work


def aggregate_wastage_by_machine(wastage_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        wastage_df.groupby(["Product ID", "Type"], as_index=False)
        .agg(
            records=("UDI", "count"),
            total_wastage_inr=("total_wastage_inr", "sum"),
            idle_wastage_inr=("idle_wastage_inr", "sum"),
            anomaly_wastage_inr=("anomaly_wastage_inr", "sum"),
            avg_failure_risk=("failure_risk", "mean"),
            anomaly_events=("is_anomaly", "sum"),
        )
        .sort_values("total_wastage_inr", ascending=False)
    )
    return agg

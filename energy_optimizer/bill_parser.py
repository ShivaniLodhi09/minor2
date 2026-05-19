"""Parse uploaded electricity bills and reconcile with model estimates."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_optimizer.config import ROOT

SAMPLE_BILL_PATH = ROOT / "data" / "sample_electricity_bill.csv"

COLUMN_ALIASES = {
    "month": ["month", "billing_month", "period"],
    "billing_start": ["billing_start", "start_date", "from_date", "period_start"],
    "billing_end": ["billing_end", "end_date", "to_date", "period_end"],
    "kwh_consumed": [
        "kwh_consumed",
        "kwh",
        "consumption_kwh",
        "units",
        "energy_kwh",
        "total_kwh",
    ],
    "amount_inr": [
        "amount_inr",
        "amount",
        "bill_amount",
        "total_amount",
        "total_inr",
        "bill_inr",
    ],
    "peak_kwh": ["peak_kwh", "peak_units", "peak_consumption"],
    "off_peak_kwh": ["off_peak_kwh", "offpeak_kwh", "off_peak_units"],
    "tariff_applied": ["tariff_applied", "tariff", "rate_per_kwh", "unit_rate"],
    "demand_charge_inr": ["demand_charge_inr", "demand_charge", "fixed_charge"],
    "notes": ["notes", "remark", "comments"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    work = df.rename(columns=lower)
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in work.columns:
                mapping[alias] = canonical
                break
    work = work.rename(columns=mapping)
    if "kwh_consumed" not in work.columns or "amount_inr" not in work.columns:
        raise ValueError(
            "Bill CSV must include consumption (kwh_consumed / kwh / units) and "
            "amount (amount_inr / bill_amount / total_amount)."
        )
    return work


def load_bill_csv(source: str | Path | bytes) -> pd.DataFrame:
    if isinstance(source, bytes):
        import io

        df = pd.read_csv(io.BytesIO(source))
    else:
        df = pd.read_csv(source)
    return normalize_bill_dataframe(df)


def normalize_bill_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    work = _normalize_columns(df.copy())
    work["kwh_consumed"] = pd.to_numeric(work["kwh_consumed"], errors="coerce")
    work["amount_inr"] = pd.to_numeric(work["amount_inr"], errors="coerce")
    work = work.dropna(subset=["kwh_consumed", "amount_inr"])
    for col in ("peak_kwh", "off_peak_kwh", "tariff_applied", "demand_charge_inr"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "tariff_applied" not in work.columns or work["tariff_applied"].isna().all():
        work["tariff_applied"] = work["amount_inr"] / work["kwh_consumed"].replace(0, pd.NA)
    if "month" not in work.columns:
        work["month"] = [f"Row-{i+1}" for i in range(len(work))]
    return work.reset_index(drop=True)


def load_sample_bill() -> pd.DataFrame:
    return load_bill_csv(SAMPLE_BILL_PATH)


def summarize_bills(bills: pd.DataFrame) -> dict:
    total_kwh = float(bills["kwh_consumed"].sum())
    total_inr = float(bills["amount_inr"].sum())
    avg_tariff = total_inr / total_kwh if total_kwh > 0 else 0.0
    peak_share = 0.0
    if "peak_kwh" in bills.columns and bills["peak_kwh"].notna().any():
        peak_share = float(bills["peak_kwh"].sum() / total_kwh) if total_kwh else 0.0
    return {
        "months": len(bills),
        "total_kwh": total_kwh,
        "total_inr": total_inr,
        "avg_monthly_inr": total_inr / len(bills) if len(bills) else 0.0,
        "avg_tariff": avg_tariff,
        "peak_share": peak_share,
        "implied_wastage_budget_inr": total_inr * 0.10,
    }


def reconcile_with_model(
    bills: pd.DataFrame,
    model_monthly_cost_inr: float,
    model_monthly_wastage_inr: float,
) -> pd.DataFrame:
    """Per-month bill vs model proxy (scaled to billing periods)."""
    rows = []
    n_months = max(len(bills), 1)
    model_cost_per_month = model_monthly_cost_inr
    model_waste_per_month = model_monthly_wastage_inr / n_months

    for _, bill in bills.iterrows():
        actual = float(bill["amount_inr"])
        kwh = float(bill["kwh_consumed"])
        implied_model = model_cost_per_month * (kwh / bills["kwh_consumed"].mean())
        wastage_est = min(model_waste_per_month * (kwh / bills["kwh_consumed"].mean()), actual * 0.25)
        rows.append(
            {
                "month": bill.get("month", ""),
                "kwh_consumed": kwh,
                "bill_amount_inr": actual,
                "model_estimate_inr": implied_model,
                "variance_inr": actual - implied_model,
                "variance_pct": ((actual - implied_model) / actual * 100) if actual else 0,
                "allocatable_wastage_inr": wastage_est,
            }
        )
    return pd.DataFrame(rows)

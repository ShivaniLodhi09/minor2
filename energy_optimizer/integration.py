"""Combine macro demand forecast with micro machine wastage and recommendations."""
from __future__ import annotations

import pandas as pd

from energy_optimizer.bill_parser import reconcile_with_model, summarize_bills
from energy_optimizer.config import DEFAULT_TARIFF_PER_KWH, PEAK_HOURS
from energy_optimizer.wastage_rules import WastageRules, PRESETS
from energy_optimizer.cost_mapping import (
    aggregate_wastage_by_machine,
    compute_machine_wastage,
    kwh_from_ccpp_output,
    rupees_from_kwh,
)
from energy_optimizer.modeling import predict_demand, predict_failure_risk
from energy_optimizer.preprocessing import prepare_ai4i, prepare_ccpp


def build_demand_forecast_table(
    ccpp: pd.DataFrame | None = None,
    tariff: float = DEFAULT_TARIFF_PER_KWH,
) -> pd.DataFrame:
    data = prepare_ccpp(ccpp).copy()
    data["predicted_output"] = predict_demand(data)
    data["kwh_proxy"] = kwh_from_ccpp_output(data["predicted_output"])
    data["cost_inr"] = rupees_from_kwh(data["kwh_proxy"], tariff=tariff)
    data["forecast_error"] = data["EnergyOutput"] - data["predicted_output"]
    return data


def build_wastage_report(
    ai4i: pd.DataFrame | None = None,
    tariff: float = DEFAULT_TARIFF_PER_KWH,
    rules: WastageRules | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    machines = prepare_ai4i(ai4i)
    risk = predict_failure_risk(machines)
    detail = compute_machine_wastage(machines, risk, tariff=tariff, rules=rules)
    summary = aggregate_wastage_by_machine(detail)
    return detail, summary


def generate_recommendations(
    demand_df: pd.DataFrame,
    wastage_summary: pd.DataFrame,
    tariff: float,
) -> list[dict]:
    tips: list[dict] = []

    total_waste = float(wastage_summary["total_wastage_inr"].sum())
    if total_waste > 0:
        top = wastage_summary.iloc[0]
        tips.append(
            {
                "priority": "High",
                "category": "Machine wastage",
                "message": (
                    f"Machine {top['Product ID']} ({top['Type']}) shows the highest "
                    f"wastage at ₹{top['total_wastage_inr']:,.0f}. Inspect for idle running "
                    "or schedule maintenance."
                ),
                "savings_inr": float(top["total_wastage_inr"] * 0.6),
            }
        )

    peak_demand = demand_df.nlargest(int(len(demand_df) * 0.1), "predicted_output")
    if len(peak_demand) > 0:
        tips.append(
            {
                "priority": "Medium",
                "category": "Peak scheduling",
                "message": (
                    f"Shift {len(peak_demand)} high-demand intervals away from "
                    f"peak hours ({PEAK_HOURS[0]}:00–{PEAK_HOURS[-1]}:00) to cut "
                    f"tariff surcharge (~{(tariff * 0.25):.1f} ₹/kWh effective)."
                ),
                "savings_inr": float(peak_demand["cost_inr"].mean() * 24 * 30 * 0.12),
            }
        )

    high_risk = wastage_summary[wastage_summary["avg_failure_risk"] > 0.5]
    if len(high_risk) > 0:
        tips.append(
            {
                "priority": "High",
                "category": "Predictive maintenance",
                "message": (
                    f"{len(high_risk)} machines show elevated failure risk. "
                    "Plan maintenance before abnormal power draw escalates."
                ),
                "savings_inr": float(high_risk["total_wastage_inr"].sum() * 0.4),
            }
        )

    idle_heavy = wastage_summary.nlargest(3, "idle_wastage_inr")
    if idle_heavy["idle_wastage_inr"].sum() > 0:
        tips.append(
            {
                "priority": "Medium",
                "category": "Idle reduction",
                "message": (
                    "Enable auto-shutdown on low-torque/high-RPM states for top idle "
                    "machines to eliminate phantom load."
                ),
                "savings_inr": float(idle_heavy["idle_wastage_inr"].sum() * 0.5),
            }
        )

    if not tips:
        tips.append(
            {
                "priority": "Low",
                "category": "Status",
                "message": "No major wastage hotspots detected. Continue monitoring.",
                "savings_inr": 0.0,
            }
        )

    return tips


def run_full_analysis(
    tariff: float = DEFAULT_TARIFF_PER_KWH,
    rules: WastageRules | None = None,
    bills: pd.DataFrame | None = None,
) -> dict:
    rules = rules or PRESETS["College demo (recommended)"]
    if bills is not None and len(bills) > 0:
        bs_early = summarize_bills(bills)
        if bs_early["avg_tariff"] > 0:
            tariff = bs_early["avg_tariff"]
    demand = build_demand_forecast_table(tariff=tariff)
    wastage_detail, wastage_summary = build_wastage_report(tariff=tariff, rules=rules)
    recommendations = generate_recommendations(demand, wastage_summary, tariff)

    hourly_cost = demand["cost_inr"]
    est_monthly = float(hourly_cost.mean() * 24 * 30)
    total_wastage = float(wastage_summary["total_wastage_inr"].sum())

    bill_summary = None
    reconciliation = None
    if bills is not None and len(bills) > 0:
        bill_summary = summarize_bills(bills)
        reconciliation = reconcile_with_model(
            bills,
            model_monthly_cost_inr=est_monthly,
            model_monthly_wastage_inr=total_wastage / max(len(bills), 1) * 12,
        )

    kpis = {
        "total_wastage_inr": total_wastage,
        "machines_flagged": int((wastage_summary["anomaly_events"] > 0).sum()),
        "avg_forecast_mw": float(demand["predicted_output"].mean()),
        "avg_hourly_cost_inr": float(hourly_cost.mean()),
        "est_monthly_cost_inr": est_monthly,
        "potential_savings_inr": float(sum(r["savings_inr"] for r in recommendations)),
    }
    if bill_summary:
        kpis["bill_total_inr"] = bill_summary["total_inr"]
        kpis["bill_avg_monthly_inr"] = bill_summary["avg_monthly_inr"]
        kpis["bill_wastage_budget_inr"] = bill_summary["implied_wastage_budget_inr"]
        kpis["wastage_vs_bill_pct"] = (
            (total_wastage / bill_summary["total_inr"] * 100) if bill_summary["total_inr"] else 0
        )

    return {
        "demand": demand,
        "wastage_detail": wastage_detail,
        "wastage_summary": wastage_summary,
        "recommendations": recommendations,
        "kpis": kpis,
        "bill_summary": bill_summary,
        "reconciliation": reconciliation,
        "rules": rules,
    }

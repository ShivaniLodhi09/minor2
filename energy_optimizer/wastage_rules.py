"""Configurable wastage heuristics — presets tuned for college demo vs production."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class WastageRules:
    """Tunable thresholds for idle / anomaly wastage (software-only proxies)."""

    idle_torque_ratio: float = 0.40
    anomaly_risk_threshold: float = 0.48
    wastage_penalty_factor: float = 0.22
    idle_baseline_fraction: float = 0.35
    demo_wastage_scale: float = 1.0
    min_wastage_inr_per_machine: float = 0.0

    def scaled(self) -> WastageRules:
        if self.demo_wastage_scale == 1.0:
            return self
        return replace(self, demo_wastage_scale=1.0)


PRESETS: dict[str, WastageRules] = {
    "College demo (recommended)": WastageRules(
        idle_torque_ratio=0.42,
        anomaly_risk_threshold=0.45,
        wastage_penalty_factor=0.25,
        idle_baseline_fraction=0.32,
        demo_wastage_scale=12.0,
        min_wastage_inr_per_machine=15.0,
    ),
    "Balanced (default)": WastageRules(
        idle_torque_ratio=0.40,
        anomaly_risk_threshold=0.48,
        wastage_penalty_factor=0.22,
        idle_baseline_fraction=0.35,
        demo_wastage_scale=1.0,
    ),
    "Conservative": WastageRules(
        idle_torque_ratio=0.30,
        anomaly_risk_threshold=0.60,
        wastage_penalty_factor=0.15,
        idle_baseline_fraction=0.40,
        demo_wastage_scale=1.0,
    ),
}


def rules_from_sidebar(
    preset_name: str,
    idle_torque_ratio: float,
    anomaly_risk_threshold: float,
    wastage_penalty_factor: float,
    demo_scale: float,
) -> WastageRules:
    if preset_name == "Custom":
        return WastageRules(
            idle_torque_ratio=idle_torque_ratio,
            anomaly_risk_threshold=anomaly_risk_threshold,
            wastage_penalty_factor=wastage_penalty_factor,
            demo_wastage_scale=demo_scale,
        )
    base = PRESETS.get(preset_name, PRESETS["Balanced (default)"])
    return replace(
        base,
        idle_torque_ratio=idle_torque_ratio,
        anomaly_risk_threshold=anomaly_risk_threshold,
        wastage_penalty_factor=wastage_penalty_factor,
        demo_wastage_scale=demo_scale,
    )


def rules_to_dict(rules: WastageRules) -> dict:
    return asdict(rules)

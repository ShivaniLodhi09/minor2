"""Generate PDF energy optimization reports."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCustom",
            parent=styles["Heading1"],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.HexColor("#1a3a5c"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontSize=13,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#2563eb"),
        )
    )
    return styles


def _table(data: list[list], col_widths=None) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def build_pdf_report(
    kpis: dict[str, Any],
    recommendations: list[dict],
    wastage_summary: pd.DataFrame,
    bill_summary: dict | None = None,
    reconciliation: pd.DataFrame | None = None,
    rules_label: str = "College demo",
    tariff: float = 8.0,
) -> bytes:
    if not HAS_REPORTLAB:
        raise ImportError("reportlab is required for PDF export. Run: pip install reportlab")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = _styles()
    story = []

    story.append(Paragraph("Industrial Energy Optimizer", styles["TitleCustom"]))
    story.append(
        Paragraph(
            f"Energy audit report · {datetime.now().strftime('%d %b %Y, %H:%M')} · "
            f"Tariff ₹{tariff:.2f}/kWh · Rules: {rules_label}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Executive summary", styles["Section"]))
    kpi_rows = [
        ["Metric", "Value"],
        ["Total machine wastage (₹)", f"₹{kpis.get('total_wastage_inr', 0):,.0f}"],
        ["Machines flagged", str(kpis.get("machines_flagged", 0))],
        ["Avg demand forecast (MW)", f"{kpis.get('avg_forecast_mw', 0):.1f}"],
        ["Est. monthly plant cost (₹)", f"₹{kpis.get('est_monthly_cost_inr', 0):,.0f}"],
        ["Potential savings (₹)", f"₹{kpis.get('potential_savings_inr', 0):,.0f}"],
    ]
    if bill_summary:
        kpi_rows.extend(
            [
                ["Bill records (months)", str(bill_summary.get("months", 0))],
                ["Total billed (₹)", f"₹{bill_summary.get('total_inr', 0):,.0f}"],
                ["Avg effective tariff (₹/kWh)", f"{bill_summary.get('avg_tariff', 0):.2f}"],
                ["10% wastage budget (₹)", f"₹{bill_summary.get('implied_wastage_budget_inr', 0):,.0f}"],
            ]
        )
    story.append(_table(kpi_rows, [8 * cm, 9 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    if reconciliation is not None and len(reconciliation) > 0:
        story.append(Paragraph("Electricity bill reconciliation", styles["Section"]))
        rec = reconciliation.copy()
        rec_display = [
            ["Month", "Bill (₹)", "Model est. (₹)", "Variance %", "Wastage alloc. (₹)"]
        ]
        for _, row in rec.head(12).iterrows():
            rec_display.append(
                [
                    str(row["month"])[:12],
                    f"₹{row['bill_amount_inr']:,.0f}",
                    f"₹{row['model_estimate_inr']:,.0f}",
                    f"{row['variance_pct']:.1f}%",
                    f"₹{row['allocatable_wastage_inr']:,.0f}",
                ]
            )
        story.append(_table(rec_display))
        story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Top wastage hotspots (machines)", styles["Section"]))
    top = wastage_summary.head(10)
    waste_rows = [["Machine ID", "Type", "Wastage (₹)", "Idle (₹)", "Anomaly (₹)", "Risk"]]
    for _, row in top.iterrows():
        waste_rows.append(
            [
                str(row["Product ID"]),
                str(row["Type"]),
                f"₹{row['total_wastage_inr']:,.0f}",
                f"₹{row['idle_wastage_inr']:,.0f}",
                f"₹{row['anomaly_wastage_inr']:,.0f}",
                f"{row['avg_failure_risk']:.0%}",
            ]
        )
    story.append(_table(waste_rows))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Recommended actions", styles["Section"]))
    for rec in recommendations[:6]:
        story.append(
            Paragraph(
                f"<b>[{rec['priority']}] {rec['category']}</b> — {rec['message']} "
                f"<i>(Est. savings ₹{rec['savings_inr']:,.0f})</i>",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.15 * cm))

    story.append(Spacer(1, 0.5 * cm))
    story.append(
        Paragraph(
            "Software-only analysis · CCPP demand forecast · AI4I anomaly detection · "
            "No IoT hardware required.",
            styles["Italic"],
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

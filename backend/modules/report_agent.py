# backend/modules/report_agent.py

import os
import re
import uuid
import datetime
from typing import Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
    KeepTogether,
)
from reportlab.pdfgen import canvas

from modules.state import AnalystState
from services.store import REPORTS_DIR


C_PRIMARY = colors.HexColor("#1f2937")
C_ACCENT = colors.HexColor("#2563eb")
C_BG = colors.HexColor("#f8fafc")
C_BORDER = colors.HexColor("#dbe3ef")
C_TEXT = colors.HexColor("#111827")
C_MUTED = colors.HexColor("#6b7280")
C_WHITE = colors.white


def _styles():
    return {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=24,
            textColor=C_PRIMARY,
            leading=30,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=9.5,
            textColor=C_MUTED,
            leading=13,
            spaceAfter=3,
            alignment=TA_LEFT,
        ),
        "meta": ParagraphStyle(
            "meta",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_MUTED,
            leading=13,
            spaceAfter=2,
            alignment=TA_LEFT,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=C_ACCENT,
            leading=18,
            spaceBefore=14,
            spaceAfter=8,
            alignment=TA_LEFT,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=C_PRIMARY,
            leading=15,
            spaceBefore=8,
            spaceAfter=6,
            alignment=TA_LEFT,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9.5,
            textColor=C_TEXT,
            leading=14,
            spaceAfter=5,
            alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=9.5,
            textColor=C_TEXT,
            leading=14,
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Helvetica",
            fontSize=8.5,
            textColor=C_MUTED,
            leading=12,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_MUTED,
            leading=12,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=12,
        ),
        "table_head": ParagraphStyle(
            "table_head",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=C_WHITE,
            alignment=TA_CENTER,
            leading=11,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            fontName="Helvetica",
            fontSize=8.5,
            textColor=C_TEXT,
            alignment=TA_LEFT,
            leading=11,
        ),
        "center": ParagraphStyle(
            "center",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_MUTED,
            alignment=TA_CENTER,
            leading=11,
        ),
    }


def _safe(text: str) -> str:
    return escape(str(text or ""))


def _clean_text(text: str) -> str:
    if not text:
        return ""

    text = str(text)

    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Remove markdown symbols
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("###", "")
    text = text.replace("##", "")
    text = text.replace("#", "")
    text = text.replace("`", "")
    text = text.replace("•", "-")

    # Remove markdown italic/simple stars
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)

    bad_endings = [
        "If you need further analysis or details, let me know!",
        "Let me know if you need further analysis.",
        "Let me know if you need more details.",
        "If you need further assistance, let me know!",
    ]

    for ending in bad_endings:
        text = text.replace(ending, "")

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _parse_sections(text: str) -> list[tuple[str, str]]:
    text = _clean_text(text)
    sections = []

    known_headings = {
        "executive summary",
        "dataset health",
        "target column",
        "key findings",
        "important columns",
        "weak or less useful columns",
        "weak columns",
        "recommendations",
        "analysis",
        "summary",
        "data quality",
        "visual analysis",
    }

    for line in text.splitlines():
        s = line.strip()

        if not s:
            continue

        clean = s.strip(":").strip()
        lower = clean.lower()

        if lower in known_headings:
            sections.append(("heading", clean.title()))
        elif re.match(r"^[A-Z][A-Z\s]{3,}$", clean):
            sections.append(("heading", clean.title()))
        elif re.match(r"^\d+\.\s", s):
            sections.append(("item", re.sub(r"^\d+\.\s", "", s)))
        elif s.startswith("- "):
            sections.append(("item", s[2:]))
        else:
            sections.append(("body", s))

    return sections


def _on_page(c: canvas.Canvas, doc):
    width, height = A4

    c.setFillColor(C_ACCENT)
    c.rect(0, height - 18, width, 18, fill=1, stroke=0)

    c.setFillColor(C_WHITE)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(28, height - 12, "AI DATA ANALYST SYSTEM")

    c.setFillColor(C_MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(28, 18, "Confidential - Validate insights before business use.")
    c.drawRightString(width - 28, 18, f"Page {doc.page}")


def _kpi_table(summary: dict, styles: dict, usable_w: float):
    shape = summary.get("shape", {}) or {}
    counts = summary.get("column_count_by_type", {}) or {}
    missing_cols = len(summary.get("missing_values", {}) or {})
    duplicate_rows = summary.get("duplicate_rows", 0)

    labels = [
        "Rows",
        "Columns",
        "Numeric",
        "Categorical",
        "Missing Cols",
        "Duplicates",
    ]

    values = [
        f"{shape.get('rows', 0):,}",
        f"{shape.get('columns', 0):,}",
        f"{counts.get('numeric', 0):,}",
        f"{counts.get('categorical', 0) + counts.get('categorical_numeric', 0):,}",
        f"{missing_cols:,}",
        f"{duplicate_rows:,}",
    ]

    data = [
        [Paragraph(f"<b>{_safe(v)}</b>", styles["center"]) for v in values],
        [Paragraph(_safe(l), styles["center"]) for l in labels],
    ]

    tbl = Table(data, colWidths=[usable_w / 6] * 6, rowHeights=[24, 18])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), C_BG),
                ("BOX", (0, 0), (-1, -1), 0.6, C_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, C_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    return tbl


def _dataset_overview_table(summary: dict, styles: dict, usable_w: float):
    roles = summary.get("column_roles", {}) or {}
    dtypes = summary.get("dtypes", {}) or {}
    missing = summary.get("missing_values", {}) or {}

    rows = [
        [
            Paragraph("Column", styles["table_head"]),
            Paragraph("Role", styles["table_head"]),
            Paragraph("Dtype", styles["table_head"]),
            Paragraph("Missing %", styles["table_head"]),
        ]
    ]

    for col, role in roles.items():
        miss = missing.get(col, {}) or {}
        miss_pct = f"{miss.get('pct', 0)}%" if miss else "0%"

        rows.append(
            [
                Paragraph(_safe(col), styles["table_cell"]),
                Paragraph(_safe(role), styles["table_cell"]),
                Paragraph(_safe(dtypes.get(col, "")), styles["table_cell"]),
                Paragraph(_safe(miss_pct), styles["table_cell"]),
            ]
        )

    tbl = Table(
        rows,
        colWidths=[
            usable_w * 0.34,
            usable_w * 0.18,
            usable_w * 0.26,
            usable_w * 0.22,
        ],
        repeatRows=1,
    )

    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
                ("GRID", (0, 0), (-1, -1), 0.4, C_BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_BG]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return tbl


def _numeric_stats_story(summary: dict, styles: dict, usable_w: float):
    story = []
    stats = summary.get("numeric_stats", {}) or {}

    if not stats:
        return story

    for col, col_stats in list(stats.items())[:12]:
        story.append(Paragraph(_safe(col), styles["h2"]))

        rows = [
            [
                Paragraph("Metric", styles["table_head"]),
                Paragraph("Value", styles["table_head"]),
            ]
        ]

        for key in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            val = col_stats.get(key, "-")

            if isinstance(val, float):
                val_text = f"{val:,.2f}"
            else:
                val_text = str(val)

            rows.append(
                [
                    Paragraph(_safe(key), styles["table_cell"]),
                    Paragraph(_safe(val_text), styles["table_cell"]),
                ]
            )

        tbl = Table(rows, colWidths=[usable_w * 0.35, usable_w * 0.65])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
                    ("GRID", (0, 0), (-1, -1), 0.4, C_BORDER),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_BG]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

        story.append(tbl)
        story.append(Spacer(1, 8))

    return story


def _add_analysis_story(story: list, text_response: str, styles: dict):
    cleaned = _clean_text(text_response)

    if not cleaned:
        story.append(Paragraph("No text analysis was generated.", styles["body"]))
        return

    for kind, content in _parse_sections(cleaned):
        if kind == "heading":
            story.append(Paragraph(_safe(content), styles["h2"]))
        elif kind == "item":
            story.append(Paragraph(_safe(f"- {content}"), styles["bullet"]))
        else:
            story.append(Paragraph(_safe(content), styles["body"]))


def _centered_image_table(img: Image, usable_w: float):
    table = Table([[img]], colWidths=[usable_w])
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def _add_chart_story(
    story: list,
    chart_paths: list[str],
    chart_captions: list[str],
    styles: dict,
    usable_w: float,
):
    if not chart_paths:
        return

    story.append(PageBreak())
    story.append(Paragraph("Visual Analysis", styles["h1"]))

    for idx, path in enumerate(chart_paths):
        if not os.path.exists(path):
            continue

        caption = chart_captions[idx] if idx < len(chart_captions) else f"Chart {idx + 1}"

        block = []

        # Chart title above image
        block.append(Paragraph(_safe(caption), styles["h2"]))

        img = Image(path)

        iw, ih = img.drawWidth, img.drawHeight
        ratio = iw / ih if ih else 1.0

        # Keep chart inside page width
        img.drawWidth = min(usable_w, iw)
        img.drawHeight = img.drawWidth / ratio if ratio else 220

        # Limit very tall charts
        if img.drawHeight > 15 * cm:
            img.drawHeight = 15 * cm
            img.drawWidth = img.drawHeight * ratio

        # Center image properly
        img.hAlign = "CENTER"
        block.append(_centered_image_table(img, usable_w))

        # Center figure caption below image
        block.append(
            Paragraph(
                _safe(f"Figure {idx + 1}: {caption}"),
                styles["caption"],
            )
        )

        block.append(Spacer(1, 8))

        story.append(KeepTogether(block))


def generate_pdf_report(
    text_response: Optional[str],
    chart_paths: list[str],
    chart_captions: list[str],
    dataset_summary: Optional[dict],
    user_question: str,
    session_id: str = "session",
) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    output_path = os.path.join(REPORTS_DIR, f"report_{uuid.uuid4().hex[:8]}.pdf")
    styles = _styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.9 * cm,
        bottomMargin=1.3 * cm,
    )

    usable_w = A4[0] - doc.leftMargin - doc.rightMargin
    story = []

    now_text = datetime.datetime.now().strftime("%d %B %Y %H:%M")

    # Header only. No target summary box. No query box.
    story.append(Paragraph("Data Analysis Report", styles["title"]))
    story.append(Paragraph(_safe(f"Session ID: {session_id}"), styles["subtitle"]))
    story.append(Paragraph(_safe(f"Generated: {now_text}"), styles["meta"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_BORDER, spaceAfter=10))

    if dataset_summary:
        story.append(_kpi_table(dataset_summary, styles, usable_w))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Analysis", styles["h1"]))
    _add_analysis_story(story, text_response or "", styles)

    _add_chart_story(story, chart_paths or [], chart_captions or [], styles, usable_w)

    if dataset_summary:
        story.append(PageBreak())
        story.append(Paragraph("Dataset Overview", styles["h1"]))
        story.append(_dataset_overview_table(dataset_summary, styles, usable_w))
        story.append(Spacer(1, 12))

        if dataset_summary.get("numeric_stats"):
            story.append(Paragraph("Numeric Statistics", styles["h1"]))
            story.extend(_numeric_stats_story(dataset_summary, styles, usable_w))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.6, color=C_BORDER, spaceAfter=6))
    story.append(
        Paragraph(
            "This report was generated automatically by the AI Data Analyst System. "
            "Please validate important insights before making business decisions.",
            styles["small"],
        )
    )

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)

    return output_path


def report_generator_node(state: AnalystState) -> AnalystState:
    try:
        report_path = generate_pdf_report(
            text_response=state.get("text_response"),
            chart_paths=state.get("chart_paths", []),
            chart_captions=state.get("chart_captions", []),
            dataset_summary=state.get("dataset_summary"),
            user_question=state.get("user_question", ""),
            session_id=state.get("session_id", "session"),
        )

        return {
            **state,
            "report_path": report_path,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"Report Agent error: {exc}",
        }
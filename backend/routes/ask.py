# backend/routes/ask.py

import os
from fastapi import APIRouter, HTTPException

from models.schemas import AskRequest, AskResponse
from services.store import (
    get_dataset,
    get_session_dataset_id,
    get_chat_history,
    set_chat_history,
)
from modules.workflow import run_analysis

router = APIRouter()


def _clean_history_for_llm(history: list[dict]) -> list[dict]:
    cleaned = []

    for turn in history:
        role = turn.get("role", "")
        content = str(turn.get("content", "")).strip()

        if not content:
            continue

        
        if role == "assistant" and content.lower() in {
            "chart generated successfully.",
            "analysis completed successfully.",
        }:
            continue

        cleaned.append(
            {
                "role": role,
                "content": content,
            }
        )

    return cleaned[-6:]



def _is_missing_values_question(question: str) -> bool:
    q = (question or "").strip().lower()

    return any(
        term in q
        for term in [
            "missing values",
            "find missing values",
            "show missing values",
            "null values",
            "nulls",
            "missing data",
            "missing chart",
            "missing graph",
        ]
    )


def _is_heatmap_question(question: str) -> bool:
    q = (question or "").strip().lower()

    return any(
        term in q
        for term in [
            "heatmap",
            "correlation heatmap",
            "correlation",
            "corr",
        ]
    )


def _is_scatter_question(question: str) -> bool:
    q = (question or "").strip().lower()

    return any(
        term in q
        for term in [
            "scatter",
            "scatter plot",
        ]
    )


def _is_chart_question(question: str) -> bool:
    q = (question or "").strip().lower()

    return any(
        term in q
        for term in [
            "chart",
            "charts",
            "graph",
            "plot",
            "visualize",
            "visual",
            "distribution",
            "histogram",
            "bar",
            "pie",
            "heatmap",
            "scatter",
            "show",
            "draw",
            "display",
        ]
    )



def _build_missing_values_text(dataset_summary: dict) -> str:
    missing = dataset_summary.get("missing_values", {}) or {}

    if not missing:
        return "There are no missing values in this dataset."

    lines = ["The columns with missing values are:"]

    for col, info in missing.items():
        count = info.get("count", 0)
        pct = info.get("pct", 0)
        lines.append(f"- {str(col).strip()}: {count} missing values ({pct}%)")

    return "\n".join(lines)


def _get_available_columns_text(dataset_summary: dict, limit: int = 15) -> str:
    columns = dataset_summary.get("columns", []) or []

    if not columns:
        return "No column list is available."

    clean_columns = [str(c).strip() for c in columns if str(c).strip()]
    shown = clean_columns[:limit]

    columns_text = ", ".join(shown)

    if len(clean_columns) > limit:
        columns_text += f", ... and {len(clean_columns) - limit} more"

    return columns_text


def _build_chart_fallback_text(question: str, dataset_summary: dict) -> str:
    """
    Better fallback when chart generation returns empty chart_paths.
    This avoids confusing message:
    'I could not map this chart request clearly...'
    """

    target_col = dataset_summary.get("target_column")
    target_confidence = dataset_summary.get("target_confidence")
    target_reason = dataset_summary.get("target_reason")
    available_columns = _get_available_columns_text(dataset_summary)

    q = (question or "").strip().lower()

    # User asked target chart, but chart was not generated
    if "target" in q and target_col:
        lines = [
            "I could not generate the target chart automatically.",
            f'The likely target column is "{target_col}".',
        ]

        if target_confidence:
            lines.append(f"Confidence: {target_confidence}.")

        if target_reason:
            lines.append(f"Reason: {target_reason}")

        lines.append("")
        lines.append("Try one of these:")
        lines.append(f'- show chart of {target_col}')
        lines.append(f'- show distribution of {target_col}')
        lines.append(f'- show bar chart of {target_col}')
        lines.append(f'- show pie chart of {target_col}')

        return "\n".join(lines)

    # Dataset has target column, give target-based suggestion
    if target_col:
        return (
            "I could not generate the requested chart automatically.\n"
            f'The likely target column is "{target_col}".\n\n'
            "Try one of these:\n"
            f"- show chart of {target_col}\n"
            f"- show distribution of {target_col}\n"
            f"- show bar chart of {target_col}\n\n"
            f"Available columns: {available_columns}"
        )

    # No target detected, show available columns
    return (
        "I could not generate the requested chart automatically.\n"
        "Please ask using a real column name from your dataset.\n\n"
        f"Available columns: {available_columns}\n\n"
        "Examples:\n"
        "- show chart of column_name\n"
        "- show distribution of column_name\n"
        "- show histogram of column_name\n"
        "- show bar chart of column_name"
    )


def _build_heatmap_fallback_text(dataset_summary: dict) -> str:
    counts = dataset_summary.get("column_count_by_type", {}) or {}

    numeric_count = int(counts.get("numeric", 0) or 0)
    categorical_numeric_count = int(counts.get("categorical_numeric", 0) or 0)
    usable_numeric = numeric_count + categorical_numeric_count

    return (
        "A correlation heatmap cannot be generated because this dataset "
        "does not have at least 2 usable numeric columns.\n\n"
        f"Usable numeric columns found: {usable_numeric}."
    )


def _build_scatter_fallback_text(dataset_summary: dict) -> str:
    counts = dataset_summary.get("column_count_by_type", {}) or {}

    numeric_count = int(counts.get("numeric", 0) or 0)
    categorical_numeric_count = int(counts.get("categorical_numeric", 0) or 0)
    usable_numeric = numeric_count + categorical_numeric_count

    return (
        "A scatter plot cannot be generated because this dataset "
        "does not have at least 2 usable numeric columns.\n\n"
        f"Usable numeric columns found: {usable_numeric}."
    )


def _file_name_from_path(path: str) -> str:
    """
    Works with Linux paths and Windows paths.
    """
    safe_path = str(path or "").replace("\\", "/")
    return os.path.basename(safe_path)


def _chart_urls_from_paths(chart_paths: list[str]) -> list[str]:
    urls = []

    for path in chart_paths:
        filename = _file_name_from_path(path)

        if filename:
            urls.append(f"/api/files/chart/{filename}")

    return urls


def _report_url_from_path(report_path: str | None) -> str | None:
    if not report_path:
        return None

    filename = _file_name_from_path(report_path)

    if not filename:
        return None

    return f"/api/files/report/{filename}"



@router.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest):
    session_dataset_id = get_session_dataset_id(payload.session_id)

    if session_dataset_id != payload.dataset_id:
        raise HTTPException(
            status_code=400,
            detail="This dataset is not attached to the current session.",
        )

    df = get_dataset(payload.dataset_id)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    history = get_chat_history(payload.session_id)
    history_for_llm = _clean_history_for_llm(history)

    final_state = run_analysis(
        dataframe=df,
        user_question=payload.question,
        chat_history=history_for_llm,
        session_id=payload.session_id,
    )

    error = str(final_state.get("error", "") or "").strip()
    intent = str(final_state.get("intent", "both") or "both").strip().lower()

    text_resp = str(final_state.get("text_response", "") or "").strip()
    chart_paths = final_state.get("chart_paths", []) or []
    chart_captions = final_state.get("chart_captions", []) or []
    report_path = final_state.get("report_path")
    dataset_summary = final_state.get("dataset_summary", {}) or {}

    chart_urls = _chart_urls_from_paths(chart_paths)
    report_url = _report_url_from_path(report_path)


    if error:
        assistant_text = f"Error: {error}"
        intent = "text"

    elif intent == "chart":
        if chart_urls:
            assistant_text = (
                f"Generated: {chart_captions[0]}"
                if chart_captions
                else "Chart generated successfully."
            )

        else:
            if _is_missing_values_question(payload.question):
                assistant_text = _build_missing_values_text(dataset_summary)
                intent = "text"

            elif _is_heatmap_question(payload.question):
                assistant_text = _build_heatmap_fallback_text(dataset_summary)
                intent = "text"

            elif _is_scatter_question(payload.question):
                assistant_text = _build_scatter_fallback_text(dataset_summary)
                intent = "text"

            else:
                assistant_text = _build_chart_fallback_text(
                    question=payload.question,
                    dataset_summary=dataset_summary,
                )
                intent = "text"

    elif intent == "both":
        
        if text_resp:
            assistant_text = text_resp
        elif chart_urls:
            assistant_text = (
                f"Generated: {chart_captions[0]}"
                if chart_captions
                else "Chart generated successfully."
            )
        else:
            if _is_chart_question(payload.question):
                assistant_text = _build_chart_fallback_text(
                    question=payload.question,
                    dataset_summary=dataset_summary,
                )
            else:
                assistant_text = "Analysis completed successfully."

    else:
        assistant_text = text_resp or "Analysis completed successfully."
        intent = "text"

    new_history = history + [
        {
            "role": "user",
            "content": payload.question,
        },
        {
            "role": "assistant",
            "content": assistant_text,
        },
    ]

    set_chat_history(payload.session_id, new_history)

    return AskResponse(
        session_id=payload.session_id,
        dataset_id=payload.dataset_id,
        intent=intent,
        text_response=assistant_text,
        chart_urls=chart_urls,
        chart_captions=chart_captions,
        report_url=report_url,
    )
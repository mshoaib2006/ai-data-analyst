# backend/modules/text_agent.py

import json
from typing import Any

from modules.state import AnalystState
from modules.llm_client import get_openai_client
from modules.settings import get_openai_model


REPORT_LIKE_TERMS = [
    "full report",
    "create a full report",
    "generate report",
    "pdf report",
    "complete report",
    "download report",
    "export report",
]


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _is_report_request(question: str) -> bool:
    q = _normalize(question)
    return any(term in q for term in REPORT_LIKE_TERMS)


def _safe_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


def _limit_dict_items(data: dict, limit: int = 20) -> dict:
    if not isinstance(data, dict):
        return {}

    limited = {}

    for index, (key, value) in enumerate(data.items()):
        if index >= limit:
            break
        limited[key] = value

    return limited


def _build_compact_dataset_context(dataset_summary: dict) -> dict:
    """
    This context is sent to the LLM.
    Do not send the full CSV to LLM because large datasets can exceed token limits.
    Send useful profile only.
    """

    if not dataset_summary:
        return {
            "error": "No dataset summary available."
        }

    numeric_stats = dataset_summary.get("numeric_stats", {}) or {}
    categorical_summary = dataset_summary.get("categorical_summary", {}) or {}
    sample_rows = dataset_summary.get("sample_rows", []) or []

    compact_sample_rows = sample_rows[:5]

    return {
        "shape": dataset_summary.get("shape", {}),
        "columns": dataset_summary.get("columns", []),
        "dtypes": dataset_summary.get("dtypes", {}),
        "column_roles": dataset_summary.get("column_roles", {}),

        # Target prediction from data_agent.py
        "predicted_target_column": dataset_summary.get("target_column"),
        "target_confidence": dataset_summary.get("target_confidence"),
        "target_reason": dataset_summary.get("target_reason"),
        "top_target_candidates": dataset_summary.get("target_candidates", [])[:5],

        # Data quality
        "missing_values": dataset_summary.get("missing_values", {}),
        "duplicate_rows": dataset_summary.get("duplicate_rows", 0),
        "memory_usage_kb": dataset_summary.get("memory_usage_kb", 0),
        "column_count_by_type": dataset_summary.get("column_count_by_type", {}),

        # Stats
        "numeric_stats": _limit_dict_items(numeric_stats, limit=20),
        "categorical_summary": _limit_dict_items(categorical_summary, limit=20),

        # Small preview only
        "sample_rows": compact_sample_rows,
    }


def _build_system_prompt(question: str) -> str:
    if _is_report_request(question):
        return """
You are a senior data analyst.

You will receive:
1. USER QUESTION
2. DATASET PROFILE

Rules:
- Read the user question carefully.
- Read the dataset profile carefully.
- Use only the provided dataset profile.
- Do not invent values.
- Do not guess columns that are not present.
- Use predicted_target_column as the predicted target column.
- If target_confidence is low, say the target should be confirmed.
- If the user asks important columns, separate:
  1. target column
  2. useful feature columns
  3. weak/removable columns
- If the user asks unimportant columns, mention ID columns, text-like identifiers, high-missing columns, and columns needing feature engineering.
- Do not simply list all columns unless the user asks for all column names.
- Keep English simple and clear.

For report requests, use this structure:

EXECUTIVE SUMMARY
[2 short paragraphs]

DATASET HEALTH
- [point]
- [point]
- [point]

TARGET COLUMN
- Predicted target:
- Confidence:
- Reason:
- Other possible target candidates:

KEY FINDINGS
- [point]
- [point]
- [point]

IMPORTANT COLUMNS
- [point]
- [point]

WEAK OR LESS USEFUL COLUMNS
- [point]
- [point]

RECOMMENDATIONS
1. [recommendation]
2. [recommendation]
3. [recommendation]
"""

    return """
You are a senior data analyst.

You will receive:
1. USER QUESTION
2. DATASET PROFILE

Your job:
- Understand the user question.
- Read the dataset profile.
- Give the best correct answer using the dataset profile.

Very important rules:
- Use only the provided dataset profile.
- Never invent values.
- Never say a column exists if it is not in the columns list.
- Do not blindly list all columns unless the user asks for all column names.
- Use predicted_target_column as the predicted target column.
- If target_confidence is low, say it should be confirmed by the user.
- If user asks “target columns”, answer with predicted target column and top target candidates.
- If user asks “important columns”, explain which columns are useful features and why.
- If user asks “unimportant columns”, explain which columns are weak/removable and why.
- For important/unimportant columns, do not use only column names. Use role, missing values, ID/text type, and target relation.
- Keep answer short, practical, and simple.
- Do not write code unless user asks for code.
"""


def analyze_text(
    user_question: str,
    dataset_summary: dict,
    chat_history: list | None = None,
) -> str:
    """
    Main change:
    Almost every text question goes to LLM with dataset profile.
    This allows LLM to understand the query and dataset together.
    """

    dataset_context = _build_compact_dataset_context(dataset_summary or {})

    client = get_openai_client()
    model = get_openai_model()

    messages = [
        {
            "role": "system",
            "content": _build_system_prompt(user_question),
        },
        {
            "role": "system",
            "content": "DATASET PROFILE:\n" + _safe_json(dataset_context),
        },
    ]

    if chat_history:
        for turn in chat_history[-4:]:
            role = turn.get("role", "user")
            content = str(turn.get("content", "")).strip()

            if content:
                messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

    messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000,
        temperature=0.1,
    )

    return (response.choices[0].message.content or "").strip()


def text_analysis_node(state: AnalystState) -> AnalystState:
    if state.get("error"):
        return state

    try:
        response = analyze_text(
            user_question=state.get("user_question", ""),
            dataset_summary=state.get("dataset_summary", {}),
            chat_history=state.get("chat_history", []),
        )

        return {
            **state,
            "text_response": response,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"Text analysis failed: {str(exc)}",
        }
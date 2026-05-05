import re
import pandas as pd
from modules.state import AnalystState


def _norm_key(text: str) -> str:
    return str(text or "").strip().lower().replace(" ", "_")


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _is_id_column(col: str) -> bool:
    c = _norm_key(col)
    return (
        c == "id"
        or c.endswith("_id")
        or c.endswith("id")
        or c in {"uuid", "guid", "key", "index"}
    )


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _infer_column_roles(df: pd.DataFrame) -> dict:
    roles = {}

    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)
        total = max(len(df), 1)
        unique_ratio = nunique / total

        if pd.api.types.is_datetime64_any_dtype(s):
            roles[col] = "datetime"

        elif _is_id_column(col):
            roles[col] = "id"

        elif pd.api.types.is_numeric_dtype(s):
            if nunique <= 20:
                roles[col] = "categorical_numeric"
            else:
                roles[col] = "numeric"

        elif unique_ratio <= 0.35:
            roles[col] = "categorical"

        else:
            roles[col] = "text"

    return roles


def _find_columns_mentioned(text: str, columns: list[str]) -> list[tuple[str, int]]:
    text_raw = str(text or "")
    text_lower = text_raw.lower()
    text_compact = _compact(text_raw)

    found = []

    for col in columns:
        clean_col = str(col).strip()

        variants = {
            clean_col.lower(),
            clean_col.lower().replace("_", " "),
            clean_col.lower().replace("-", " "),
        }

        pos_found = None

        for variant in variants:
            pos = text_lower.find(variant)
            if pos >= 0:
                pos_found = pos
                break

        if pos_found is None:
            compact_col = _compact(clean_col)
            pos = text_compact.find(compact_col)
            if pos >= 0:
                pos_found = pos

        if pos_found is not None:
            found.append((clean_col, pos_found))

    found.sort(key=lambda x: x[1])
    return found


def _extract_user_confirmed_target(
    df: pd.DataFrame,
    chat_history: list[dict] | None,
    current_question: str,
) -> str | None:
    """
    User correction should override auto detection.

    Examples:
    - Loan_Status is target column
    - target column is Loan_Status
    - use Loan_Status as target
    - LoanAmount is wrong, Loan_Status is target
    """

    columns = [str(c).strip() for c in df.columns]
    messages = []

    if chat_history:
        for turn in chat_history:
            if turn.get("role") == "user":
                messages.append(str(turn.get("content", "") or ""))

    if current_question:
        messages.append(str(current_question))

    target_words = [
        "target",
        "target column",
        "target columns",
        "target variable",
        "label",
        "label column",
        "outcome",
        "dependent variable",
        "prediction column",
    ]

    for message in reversed(messages):
        q = message.lower()

        if not any(word in q for word in target_words):
            continue

        mentioned = _find_columns_mentioned(message, columns)

        if not mentioned:
            continue

        scored = []

        for col, pos in mentioned:
            before = q[max(0, pos - 90):pos]
            after = q[pos:pos + 90]
            nearby = before + " " + after

            score = 0

            if any(word in nearby for word in target_words):
                score += 5

            if re.search(r"(target|label|outcome).{0,40}(is|are|as|to|should be)", before):
                score += 8

            if re.search(r"(is|are|as).{0,40}(target|label|outcome)", after):
                score += 8

            if any(word in nearby for word in ["correct", "actually", "use", "set", "treat", "should be"]):
                score += 3

            if "wrong" in after[:60] or "not target" in after[:80]:
                score -= 8

            col_key = _norm_key(col)

            if any(token in col_key for token in ["status", "target", "label", "outcome", "result", "class"]):
                score += 4

            scored.append((score, pos, col))

        if scored:
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

            if scored[0][0] > 0:
                return scored[0][2]

    return None


def _target_name_score(col: str) -> tuple[float, list[str]]:
    key = _norm_key(col)
    compact = _compact(col)

    score = 0.0
    reasons = []

    very_strong_exact = {
        "target",
        "label",
        "y",
        "class",
        "outcome",
        "result",
        "status",
        "loan_status",
        "loanstatus",
        "approval_status",
        "application_status",
        "approved",
        "default",
        "churn",
        "survived",
        "diagnosis",
        "species",
        "fraud",
        "is_fraud",
        "converted",
        "conversion",
        "attrition",
        "stroke",
        "heart_disease",
        "disease",
        "risk",
        "sentiment",
    }

    if key in very_strong_exact or compact in very_strong_exact:
        score += 140
        reasons.append("Column name exactly matches a common target/label name.")

    strong_tokens = [
        "target",
        "label",
        "outcome",
        "result",
        "status",
        "approval",
        "approved",
        "default",
        "churn",
        "surviv",
        "diagnosis",
        "class",
        "species",
        "fraud",
        "converted",
        "conversion",
        "attrition",
        "disease",
        "risk",
        "sentiment",
        "category",
    ]

    if any(token in key for token in strong_tokens) or any(token in compact for token in strong_tokens):
        score += 90
        reasons.append("Column name contains a strong target-like word.")

    regression_exact = {
        "price",
        "sale_price",
        "saleprice",
        "selling_price",
        "sellingprice",
        "house_price",
        "houseprice",
        "loan_amount",
        "loanamount",
        "amount",
        "sales",
        "revenue",
        "score",
        "quality",
        "grade",
        "rating",
        "profit",
        "cost",
        "income",
        "salary",
        "charges",
        "fare",
    }

    if key in regression_exact or compact in regression_exact:
        score += 45
        reasons.append("Column name matches a possible regression target.")

    regression_tokens = [
        "price",
        "amount",
        "sales",
        "revenue",
        "score",
        "quality",
        "grade",
        "rating",
        "profit",
        "cost",
        "salary",
        "charges",
        "fare",
    ]

    if any(token in key for token in regression_tokens) or any(token in compact for token in regression_tokens):
        score += 20
        reasons.append("Column name contains a possible regression target word.")

    feature_like_tokens = [
        "id",
        "name",
        "email",
        "phone",
        "address",
        "date",
        "time",
        "created",
        "updated",
        "description",
        "comment",
        "notes",
        "url",
        "link",
        "image",
        "photo",
        "text",
    ]

    if any(token in key for token in feature_like_tokens):
        score -= 70
        reasons.append("Column name looks like ID/text/metadata, not target.")

    return score, reasons


def _value_pattern_score(df: pd.DataFrame, col: str, role: str) -> tuple[float, list[str]]:
    s = df[col]
    total = max(len(df), 1)
    nunique = s.nunique(dropna=True)
    unique_ratio = nunique / total
    missing_ratio = s.isna().mean()

    score = 0.0
    reasons = []

    if role == "id":
        return -9999.0, ["ID column is not suitable as target."]

    if missing_ratio > 0.60:
        score -= 60
        reasons.append("Column has too many missing values.")

    if role == "categorical":
        score += 35
        reasons.append("Column is categorical.")

    elif role == "categorical_numeric":
        score += 25
        reasons.append("Column is low-cardinality numeric.")

    elif role == "numeric":
        score += 5
        reasons.append("Column is numeric.")

    elif role == "text":
        score -= 25
        reasons.append("Column is text-like.")

    elif role == "datetime":
        score -= 60
        reasons.append("Datetime column is usually not target.")

    if nunique == 2:
        score += 50
        reasons.append("Column is binary, often a classification target.")

    elif 3 <= nunique <= 10:
        score += 35
        reasons.append("Column has low cardinality.")

    elif 11 <= nunique <= 30:
        score += 15
        reasons.append("Column has medium-low cardinality.")

    elif unique_ratio > 0.80:
        score -= 45
        reasons.append("Column has very high uniqueness, less likely target.")

    elif unique_ratio > 0.60:
        score -= 25
        reasons.append("Column has high uniqueness.")

    if role == "numeric" and nunique > 30:
        score -= 10
        reasons.append("Continuous numeric target is possible but less certain.")

    return score, reasons


def _position_score(index: int, total_cols: int) -> tuple[float, list[str]]:
    if total_cols <= 1:
        return 0.0, []

    score = (index / (total_cols - 1)) * 20
    return score, ["Later columns are often targets."]


def _score_target_candidate(
    df: pd.DataFrame,
    col: str,
    roles: dict,
    index: int,
    total_cols: int,
) -> dict:
    role = roles.get(col)

    name_score, name_reasons = _target_name_score(col)
    value_score, value_reasons = _value_pattern_score(df, col, role)
    pos_score, pos_reasons = _position_score(index, total_cols)

    final_score = name_score + value_score + pos_score

    return {
        "column": col,
        "score": round(final_score, 2),
        "role": role,
        "unique_values": int(df[col].nunique(dropna=True)),
        "missing_pct": round(float(df[col].isna().mean() * 100), 2),
        "reasons": (name_reasons + value_reasons + pos_reasons)[:5],
    }


def _infer_target_column(df: pd.DataFrame, roles: dict) -> dict:
    candidates = []
    total_cols = len(df.columns)

    for index, col in enumerate(df.columns):
        candidate = _score_target_candidate(
            df=df,
            col=col,
            roles=roles,
            index=index,
            total_cols=total_cols,
        )

        if candidate["score"] <= -999:
            continue

        candidates.append(candidate)

    if not candidates:
        return {
            "target_column": None,
            "target_confidence": "none",
            "target_reason": "No suitable target column found.",
            "target_candidates": [],
        }

    candidates.sort(key=lambda x: x["score"], reverse=True)

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    margin = best["score"] - second["score"] if second else best["score"]

    if best["score"] >= 170 and margin >= 25:
        confidence = "high"
    elif best["score"] >= 110 and margin >= 10:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "target_column": best["column"],
        "target_confidence": confidence,
        "target_reason": " ".join(best["reasons"][:3]),
        "target_candidates": candidates[:5],
    }


def _missing_value_report(df: pd.DataFrame) -> dict:
    total = len(df)
    report = {}

    for col in df.columns:
        n_missing = int(df[col].isnull().sum())

        if n_missing > 0:
            report[col] = {
                "count": n_missing,
                "pct": round(n_missing / max(total, 1) * 100, 2),
            }

    return report


def _numeric_stats(df: pd.DataFrame) -> dict:
    num_df = df.select_dtypes(include="number")

    if num_df.empty:
        return {}

    return num_df.describe().round(4).to_dict()


def _categorical_summary(df: pd.DataFrame, roles: dict) -> dict:
    summary = {}

    for col, role in roles.items():
        if role in {"categorical", "categorical_numeric"}:
            vc = df[col].astype(str).value_counts(dropna=False).head(10).to_dict()
            summary[col] = {str(k): int(v) for k, v in vc.items()}

    return summary


def build_dataset_summary(
    df: pd.DataFrame,
    chat_history: list[dict] | None = None,
    current_question: str = "",
) -> dict:
    if df is None or df.empty:
        return {"error": "DataFrame is empty or None."}

    df = _clean_dataframe(df)
    roles = _infer_column_roles(df)

    user_confirmed_target = _extract_user_confirmed_target(
        df=df,
        chat_history=chat_history,
        current_question=current_question,
    )

    if user_confirmed_target:
        target_info = {
            "target_column": user_confirmed_target,
            "target_confidence": "user_confirmed",
            "target_reason": "User explicitly confirmed this column as the target column.",
            "target_candidates": [
                {
                    "column": user_confirmed_target,
                    "score": 999,
                    "role": roles.get(user_confirmed_target),
                    "unique_values": int(df[user_confirmed_target].nunique(dropna=True)),
                    "missing_pct": round(float(df[user_confirmed_target].isna().mean() * 100), 2),
                    "reasons": ["User confirmed target column."],
                }
            ],
        }
    else:
        target_info = _infer_target_column(df, roles)

    return {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "column_roles": roles,

        "target_column": target_info.get("target_column"),
        "target_confidence": target_info.get("target_confidence"),
        "target_reason": target_info.get("target_reason"),
        "target_candidates": target_info.get("target_candidates", []),

        "missing_values": _missing_value_report(df),
        "numeric_stats": _numeric_stats(df),
        "categorical_summary": _categorical_summary(df, roles),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_count_by_type": {
            role: sum(1 for r in roles.values() if r == role)
            for role in [
                "numeric",
                "categorical",
                "categorical_numeric",
                "datetime",
                "text",
                "id",
            ]
        },
    }


def load_dataset_node(state: AnalystState) -> AnalystState:
    try:
        df = state.get("dataframe")

        if df is None:
            return {
                **state,
                "error": "No dataset loaded.",
            }

        summary = build_dataset_summary(
            df=df,
            chat_history=state.get("chat_history", []),
            current_question=state.get("user_question", ""),
        )

        return {
            **state,
            "dataset_summary": summary,
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"Data Agent error: {exc}",
        }

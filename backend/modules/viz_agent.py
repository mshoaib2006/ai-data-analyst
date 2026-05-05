# backend/modules/viz_agent.py
import os
import re
import uuid
import warnings

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from modules.state import AnalystState

try:
    from services.store import CHARTS_DIR
except Exception:
    CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "charts")

warnings.filterwarnings("ignore")
os.makedirs(CHARTS_DIR, exist_ok=True)


def _save_figure(fig, prefix: str) -> str:
    safe_prefix = _normalize(prefix) or "chart"
    path = os.path.join(CHARTS_DIR, f"{safe_prefix}_{uuid.uuid4().hex[:8]}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _normal_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _is_id_column(col: str) -> bool:
    c = str(col).strip().lower()
    return c == "id" or c.endswith("_id") or c.endswith("id")


def _safe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _get_roles(df: pd.DataFrame) -> dict:
    roles = {}

    for col in df.columns:
        clean_col = str(col).strip()
        s = df[clean_col]
        nunique = s.nunique(dropna=True)

        if pd.api.types.is_datetime64_any_dtype(s):
            roles[clean_col] = "datetime"

        elif _is_id_column(clean_col):
            roles[clean_col] = "id"

        elif pd.api.types.is_numeric_dtype(s):
            if nunique <= 20:
                roles[clean_col] = "categorical_numeric"
            else:
                roles[clean_col] = "numeric"

        elif nunique / max(len(df), 1) <= 0.30:
            roles[clean_col] = "categorical"

        else:
            roles[clean_col] = "text"

    return roles



def _column_lookup(df: pd.DataFrame) -> dict:
    lookup = {}

    for col in df.columns:
        clean_col = str(col).strip()

        lookup[_normalize(clean_col)] = clean_col
        lookup[_normalize(clean_col.replace("_", " "))] = clean_col
        lookup[_normalize(clean_col.replace("-", " "))] = clean_col

    aliases = {
        "gender": ["sex", "gender"],
        "sex": ["sex", "gender"],
        "survival": ["survived", "survival"],
        "survivals": ["survived", "survival"],
        "fare": ["fare", "ticketfare"],
        "class": ["class", "pclass", "passengerclass"],
        "target": ["target", "label", "outcome", "result", "status"],
        "label": ["label", "target", "outcome", "class"],
        "outcome": ["outcome", "target", "label", "result"],
        "price": ["price", "saleprice", "sale_price", "sellingprice", "selling_price"],
        "sales": ["sales", "sale", "revenue"],
        "revenue": ["revenue", "sales", "amount"],
    }

    for alias, possible_cols in aliases.items():
        if alias in lookup:
            continue

        for mapped in possible_cols:
            mapped_key = _normalize(mapped)
            if mapped_key in lookup:
                lookup[alias] = lookup[mapped_key]
                break

    return lookup


def _find_columns_in_question(question: str, df: pd.DataFrame) -> list[str]:
    qn = _normalize(question)
    q_words = set(_normal_words(question))
    lookup = _column_lookup(df)

    found = []

    for key, col in lookup.items():
        if not key:
            continue

        # Exact normalized substring match
        if key in qn and col not in found:
            found.append(col)
            continue

        # Word match for simple column names
        if key in q_words and col not in found:
            found.append(col)

    return found


def _get_summary_target(df: pd.DataFrame, dataset_summary: dict | None) -> str | None:
    if not dataset_summary:
        return None

    target = dataset_summary.get("target_column")

    if not target:
        return None

    for col in df.columns:
        if _normalize(col) == _normalize(target):
            return col

    return None


def _infer_target_without_summary(df: pd.DataFrame, roles: dict) -> str | None:
    preferred = [
        "target",
        "label",
        "class",
        "outcome",
        "result",
        "status",
        "survived",
        "churn",
        "diagnosis",
        "species",
        "quality",
        "grade",
        "score",
        "price",
        "saleprice",
        "sale_price",
        "sellingprice",
        "selling_price",
        "houseprice",
        "house_price",
        "amount",
        "revenue",
        "sales",
    ]

    normalized_map = {_normalize(col): col for col in df.columns}

    for name in preferred:
        key = _normalize(name)
        if key in normalized_map:
            return normalized_map[key]

    target_tokens = [
        "target",
        "label",
        "outcome",
        "result",
        "status",
        "surviv",
        "churn",
        "diagnosis",
        "class",
        "species",
        "quality",
        "grade",
        "score",
        "price",
        "sale",
        "amount",
        "revenue",
    ]

    for col in df.columns:
        if roles.get(col) == "id":
            continue

        col_key = _normalize(col)

        if any(token in col_key for token in target_tokens):
            return col

    non_id_cols = [col for col in df.columns if roles.get(col) != "id"]

    if non_id_cols:
        return non_id_cols[-1]

    return None


def _find_target_column(
    question: str,
    df: pd.DataFrame,
    roles: dict,
    dataset_summary: dict | None = None,
) -> str | None:
    q = question.lower()
    explicit = _find_columns_in_question(question, df)
    summary_target = _get_summary_target(df, dataset_summary)

    preferred = {
        "target",
        "label",
        "outcome",
        "result",
        "status",
        "survived",
        "loanstatus",
        "loan_status",
        "default",
        "churn",
        "spam",
        "approved",
        "class",
        "diagnosis",
        "species",
        "quality",
        "grade",
        "score",
        "price",
        "saleprice",
        "sale_price",
        "sellingprice",
        "selling_price",
        "houseprice",
        "house_price",
        "amount",
        "revenue",
        "sales",
    }

    for col in explicit:
        if _normalize(col) in {_normalize(x) for x in preferred}:
            return col

    target_words = [
        "target",
        "label",
        "outcome",
        "result",
        "prediction column",
        "dependent variable",
    ]

    if any(word in q for word in target_words):
        if summary_target:
            return summary_target

        inferred = _infer_target_without_summary(df, roles)

        if inferred:
            return inferred

    if any(word in q for word in ["survival", "survived"]):
        for col in df.columns:
            if "surviv" in _normalize(col):
                return col

    return None


def _pick_primary_column(
    question: str,
    df: pd.DataFrame,
    roles: dict,
    dataset_summary: dict | None = None,
) -> str | None:
    explicit = _find_columns_in_question(question, df)

    for col in explicit:
        if roles.get(col) != "id":
            return col

    target = _find_target_column(question, df, roles, dataset_summary)

    if target:
        return target

    q = question.lower()
    summary_target = _get_summary_target(df, dataset_summary)

    chart_words = [
        "chart",
        "charts",
        "plot",
        "graph",
        "visualize",
        "distribution",
        "show",
        "display",
        "draw",
        "make",
        "create",
    ]

    if summary_target and any(word in q for word in chart_words):
        return summary_target

    return None


def _pick_secondary_column(
    question: str,
    df: pd.DataFrame,
    roles: dict,
    exclude: str | None = None,
) -> str | None:
    explicit = _find_columns_in_question(question, df)

    for col in explicit:
        if col != exclude and roles.get(col) != "id":
            return col

    q = question.lower()

    if "sex" in q or "gender" in q:
        for col in df.columns:
            clean_col = str(col).strip()
            if clean_col != exclude and _normalize(clean_col) in {"sex", "gender"}:
                return clean_col

    if "class" in q:
        for col in df.columns:
            clean_col = str(col).strip()
            if clean_col != exclude and "class" in _normalize(clean_col):
                return clean_col

    return None


def _wants_heatmap(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["heatmap", "correlation", "corr"])


def _wants_missing_values(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in [
        "missing values",
        "null values",
        "nulls",
        "missing data",
        "visualize missing",
        "missing chart",
        "missing graph",
    ])


def _wants_bar(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in [
        "bar",
        "bar graph",
        "bar chart",
        "count plot",
        "countplot",
    ])

def _wants_pie(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in [
        "pie",
        "pie chart",
        "percentage chart",
        "percent chart",
    ])


def _wants_distribution(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in [
        "distribution",
        "distributions",
        "histogram",
        "hist",
        "frequency",
    ])


def _wants_numeric_overview(question: str) -> bool:
    q = question.lower()

    phrases = [
        "numeric columns",
        "numerical columns",
        "all numeric columns",
        "all numerical columns",
        "histogram of numerical columns",
        "histogram of numeric columns",
        "distribution of numeric columns",
        "distribution of numerical columns",
        "show histogram of numerical columns",
        "show histogram of numeric columns",
    ]

    return any(p in q for p in phrases)


def _wants_relation(question: str) -> bool:
    q = question.lower()

    relation_words = [
        " by ",
        "regarding",
        "across",
        "grouped",
        "vs",
        "versus",
        "against",
        "compare",
    ]

    return any(w in q for w in relation_words)


def _wants_compare_features(question: str) -> bool:
    q = question.lower()

    patterns = [
        "compare features",
        "compare columns",
        "compare variables",
        "compare features visually",
        "visual comparison",
        "compare all features",
    ]

    return any(p in q for p in patterns)


def _wants_scatter(question: str) -> bool:
    q = question.lower()

    return any(p in q for p in [
        "scatter",
        "scatter plot",
        "show scatter plot",
        "visualize scatter",
    ])


def _is_report_request(question: str) -> bool:
    q = question.lower()

    return any(k in q for k in [
        "full report",
        "create a full report",
        "generate report",
        "pdf report",
        "complete report",
        "dashboard",
        "eda",
        "profile",
    ])



def _is_low_cardinality(df: pd.DataFrame, col: str) -> bool:
    return df[col].nunique(dropna=True) <= 20


def _sorted_counts(series: pd.Series) -> pd.Series:
    vc = series.fillna("Missing").astype(str).value_counts(dropna=False)

    try:
        numeric_idx = pd.to_numeric(vc.index)
        vc = vc.iloc[np.argsort(numeric_idx)]
    except Exception:
        pass

    return vc


def _best_scatter_columns(df: pd.DataFrame, roles: dict) -> tuple[str | None, str | None]:
    numeric_cols = [
        c for c, r in roles.items()
        if r == "numeric" and not _is_id_column(c)
    ]

    if len(numeric_cols) < 2:
        fallback = [
            c for c, r in roles.items()
            if r in {"numeric", "categorical_numeric"} and not _is_id_column(c)
        ]

        if len(fallback) >= 2:
            return fallback[0], fallback[1]

        return None, None

    scored = []

    for col in numeric_cols:
        try:
            scored.append((col, df[col].nunique(dropna=True), float(df[col].var())))
        except Exception:
            scored.append((col, 0, 0.0))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    return scored[0][0], scored[1][0]


def _chart_bar_distribution(df: pd.DataFrame, col: str) -> str:
    vc = _sorted_counts(df[col]).head(30)

    fig, ax = plt.subplots(figsize=(9, 5))
    vc.plot(kind="bar", ax=ax)

    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()

    return _save_figure(fig, f"bar_{col}")


def _chart_pie_distribution(df: pd.DataFrame, col: str) -> str | None:
    vc = _sorted_counts(df[col]).head(10)

    if vc.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 7))
    vc.plot(kind="pie", autopct="%1.1f%%", ax=ax)

    ax.set_title(f"Percentage Distribution of {col}")
    ax.set_ylabel("")

    plt.tight_layout()

    return _save_figure(fig, f"pie_{col}")


def _chart_hist_distribution(df: pd.DataFrame, col: str) -> str:
    data = df[col].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data, bins=30)

    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

    if not data.empty:
        mean_value = data.mean()
        median_value = data.median()

        ax.axvline(
            mean_value,
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_value:,.2f}",
        )

        ax.axvline(
            median_value,
            linestyle=":",
            linewidth=1.5,
            label=f"Median: {median_value:,.2f}",
        )

        ax.legend()

    plt.tight_layout()

    return _save_figure(fig, f"hist_{col}")


def _chart_grouped_distribution(df: pd.DataFrame, group_col: str, target_col: str) -> str:
    temp = df[[group_col, target_col]].copy()

    temp[group_col] = temp[group_col].fillna("Missing").astype(str)
    temp[target_col] = temp[target_col].fillna("Missing").astype(str)

    cross = pd.crosstab(temp[group_col], temp[target_col])

    if len(cross) > 25:
        top_groups = temp[group_col].value_counts().head(25).index
        temp = temp[temp[group_col].isin(top_groups)]
        cross = pd.crosstab(temp[group_col], temp[target_col])

    fig, ax = plt.subplots(figsize=(10, 5))
    cross.plot(kind="bar", ax=ax)

    ax.set_title(f"Distribution of {target_col} by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()

    return _save_figure(fig, f"grouped_{group_col}_{target_col}")


def _chart_heatmap(df: pd.DataFrame, roles: dict) -> str | None:
    cols = [
        c for c, r in roles.items()
        if r in {"numeric", "categorical_numeric"} and not _is_id_column(c)
    ]

    if len(cols) < 2:
        return None

    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax,
    )

    ax.set_title("Correlation Heatmap")

    plt.tight_layout()

    return _save_figure(fig, "heatmap")


def _chart_missing_values(df: pd.DataFrame) -> str | None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    missing.plot(kind="bar", ax=ax)

    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Count")
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()

    return _save_figure(fig, "missing_values")


def _chart_outliers(df: pd.DataFrame, roles: dict) -> str | None:
    cols = [
        c for c, r in roles.items()
        if r == "numeric" and not _is_id_column(c)
    ]

    if not cols:
        return None

    cols = cols[:8]

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.boxplot(data=df[cols], ax=ax)

    ax.set_title("Outlier Overview")
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()

    return _save_figure(fig, "outliers")


def _chart_numeric_overview(df: pd.DataFrame, roles: dict) -> str | None:
    cols = [
        c for c, r in roles.items()
        if r in {"numeric", "categorical_numeric"} and not _is_id_column(c)
    ]

    if not cols:
        return None

    cols = cols[:6]
    n = len(cols)
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        ax = axes[i]

        if _is_low_cardinality(df, col):
            _sorted_counts(df[col]).plot(kind="bar", ax=ax)
        else:
            ax.hist(df[col].dropna(), bins=25)

        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=25)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    return _save_figure(fig, "numeric_overview")


def _chart_categorical_overview(df: pd.DataFrame, roles: dict) -> str | None:
    cols = [
        c for c, r in roles.items()
        if r == "categorical"
    ]

    if not cols:
        return None

    cols = cols[:4]

    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 4 * len(cols)))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        _sorted_counts(df[col]).head(10).plot(kind="bar", ax=axes[i])

        axes[i].set_title(f"Top Categories: {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis="x", rotation=25)

    plt.tight_layout()

    return _save_figure(fig, "categorical_overview")


def _chart_scatter(df: pd.DataFrame, roles: dict, question: str) -> str | None:
    explicit = _find_columns_in_question(question, df)

    numeric_cols = [
        c for c, r in roles.items()
        if r in {"numeric", "categorical_numeric"} and not _is_id_column(c)
    ]

    x_col = None
    y_col = None

    explicit_numeric = [c for c in explicit if c in numeric_cols]

    if len(explicit_numeric) >= 2:
        x_col, y_col = explicit_numeric[0], explicit_numeric[1]
    else:
        x_col, y_col = _best_scatter_columns(df, roles)

    if not x_col or not y_col:
        return None

    temp = df[[x_col, y_col]].dropna()

    if temp.empty:
        return None

    if len(temp) > 2000:
        temp = temp.sample(2000, random_state=42)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(temp[x_col], temp[y_col], alpha=0.6)

    ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    plt.tight_layout()

    return _save_figure(fig, f"scatter_{x_col}_{y_col}")


def generate_chart_pack(
    df: pd.DataFrame,
    question: str,
    intent: str | None = None,
    dataset_summary: dict | None = None,
) -> tuple[list[str], list[str]]:
    df = _safe_columns(df)
    question = question or ""
    q = question.lower()

    roles = _get_roles(df)
    summary_target = _get_summary_target(df, dataset_summary)

    # Full report 
    if _is_report_request(question):
        items = [
            ("Numeric Overview", _chart_numeric_overview(df, roles)),
            ("Categorical Overview", _chart_categorical_overview(df, roles)),
            ("Correlation Heatmap", _chart_heatmap(df, roles)),
            ("Missing Values", _chart_missing_values(df)),
            ("Outlier Overview", _chart_outliers(df, roles)),
        ]

        paths, captions = [], []

        for caption, path in items:
            if path:
                paths.append(path)
                captions.append(caption)

        return paths, captions

    # Compare features
    if _wants_compare_features(question):
        path = _chart_heatmap(df, roles)

        if path:
            return [path], ["Feature Comparison Heatmap"]

        path = _chart_numeric_overview(df, roles)

        if path:
            return [path], ["Numeric Overview"]

        path = _chart_categorical_overview(df, roles)

        if path:
            return [path], ["Categorical Overview"]

        return [], []

    # Missing values chart
    if _wants_missing_values(question):
        path = _chart_missing_values(df)

        return ([path], ["Missing Values"]) if path else ([], [])

    # Heatmap
    if _wants_heatmap(question):
        path = _chart_heatmap(df, roles)

        return ([path], ["Correlation Heatmap"]) if path else ([], [])

    # Scatter
    if _wants_scatter(question):
        path = _chart_scatter(df, roles, question)

        return ([path], ["Scatter Plot"]) if path else ([], [])

    # Numeric overview
    if _wants_numeric_overview(question):
        path = _chart_numeric_overview(df, roles)

        return ([path], ["Numeric Overview"]) if path else ([], [])

    # Target and column selection
    target_col = _find_target_column(
        question=question,
        df=df,
        roles=roles,
        dataset_summary=dataset_summary,
    )

    if not target_col and "target" in q and summary_target:
        target_col = summary_target

    primary_col = _pick_primary_column(
        question=question,
        df=df,
        roles=roles,
        dataset_summary=dataset_summary,
    )

    if not primary_col and summary_target and any(
        k in q for k in ["chart", "plot", "graph", "distribution", "show", "display"]
    ):
        primary_col = summary_target

    secondary_col = _pick_secondary_column(
        question=question,
        df=df,
        roles=roles,
        exclude=target_col or primary_col,
    )

    # Target by another column
    if target_col and secondary_col and target_col != secondary_col and _wants_relation(question):
        path = _chart_grouped_distribution(df, secondary_col, target_col)

        return [path], [f"Distribution of {target_col} by {secondary_col}"]

    # Target chart
    if target_col and (
        "target" in q
        or "label" in q
        or "outcome" in q
        or "survival" in q
        or "survived" in q
    ):
        if _wants_pie(question) and _is_low_cardinality(df, target_col):
            path = _chart_pie_distribution(df, target_col)

            if path:
                return [path], [f"Percentage Distribution of {target_col}"]

        if _wants_bar(question) or _is_low_cardinality(df, target_col):
            path = _chart_bar_distribution(df, target_col)
        else:
            path = _chart_hist_distribution(df, target_col)

        return [path], [f"Distribution of {target_col}"]

    # Any selected column chart
    if primary_col:
        if _wants_pie(question) and _is_low_cardinality(df, primary_col):
            path = _chart_pie_distribution(df, primary_col)

            if path:
                return [path], [f"Percentage Distribution of {primary_col}"]

        if _wants_bar(question) or _is_low_cardinality(df, primary_col):
            path = _chart_bar_distribution(df, primary_col)
        else:
            path = _chart_hist_distribution(df, primary_col)

        return [path], [f"Distribution of {primary_col}"]

    return [], []


# LangGraph Node


def visualization_node(state: AnalystState) -> AnalystState:
    if state.get("error"):
        return state

    df = state.get("dataframe")

    if df is None:
        return {
            **state,
            "error": "No DataFrame for visualization.",
        }

    try:
        chart_paths, chart_captions = generate_chart_pack(
            df=df,
            question=state.get("user_question", ""),
            intent=state.get("intent"),
            dataset_summary=state.get("dataset_summary", {}),
        )

        return {
            **state,
            "chart_paths": chart_paths,
            "chart_captions": chart_captions,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"Visualization Agent error: {str(exc)}",
        }
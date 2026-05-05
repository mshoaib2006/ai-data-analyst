from modules.state import AnalystState

CHART_PREFIXES = [
    "show", "plot", "draw", "visualize", "display", "make", "create"
]

CHART_TERMS = [
    "chart", "charts", "graph", "graphs", "plot", "plots",
    "bar", "bar chart", "bar graph",
    "histogram", "hist",
    "heatmap", "correlation",
    "scatter", "scatter plot",
    "distribution", "distributions",
    "pie", "pie chart",
    "visual", "visualize",
]

TEXT_TERMS = [
    "explain", "summary", "summarize", "recommend", "recommendations",
    "insights", "interpret", "analysis", "analyze", "why", "how",
    "duplicate", "duplicate rows",
    "missing values", "null values",
    "target column", "target variable", "label column", "outcome column",
    "is target", "is the target",
]

BOTH_TERMS = [
    "explain and show",
    "analyze and show",
    "summary with chart",
    "insights with chart",
    "report with charts",
    "full report",
    "complete report",
    "eda",
    "dashboard",
]


def _has_any(q: str, terms: list[str]) -> bool:
    return any(term in q for term in terms)


def _starts_like_chart(q: str) -> bool:
    return any(q.startswith(prefix + " ") for prefix in CHART_PREFIXES)


def classify_intent(user_question: str, chat_history: list | None = None) -> str:
    q = (user_question or "").strip().lower()

    if not q:
        return "both"

    if _has_any(q, BOTH_TERMS):
        return "both"

    has_chart = _starts_like_chart(q) or _has_any(q, CHART_TERMS)
    has_text = _has_any(q, TEXT_TERMS)

    # Important:
    # "show target column chart" must be chart, not text.
    if has_chart and has_text:
        if any(word in q for word in ["explain", "analysis", "analyze", "summary", "insight"]):
            return "both"
        return "chart"

    if has_chart:
        return "chart"

    if has_text:
        return "text"

    return "text"


def intent_classifier_node(state: AnalystState) -> AnalystState:
    if state.get("error"):
        return state

    intent = classify_intent(
        user_question=state.get("user_question", ""),
        chat_history=state.get("chat_history", []),
    )

    return {
        **state,
        "intent": intent,
    }


def route_by_intent(state: AnalystState) -> str:
    if state.get("error"):
        return "report_generator_node"

    intent = state.get("intent", "text")

    if intent == "chart":
        return "visualization_node"

    if intent == "both":
        return "text_analysis_node"

    return "text_analysis_node"
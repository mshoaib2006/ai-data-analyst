from langgraph.graph import StateGraph, END
from modules.state import AnalystState
from modules.data_agent import load_dataset_node
from modules.intent_agent import intent_classifier_node, route_by_intent
from modules.text_agent import text_analysis_node
from modules.viz_agent import visualization_node
from modules.report_agent import report_generator_node

def _report_requested(question: str) -> bool:
    q = (question or "").strip().lower()
    return any(term in q for term in [
        "full report", "create a full report", "generate report",
        "pdf report", "download report", "export report", "complete report"
    ])

def _route_after_text(state: AnalystState) -> str:
    if state.get("error"):
        return "end"
    question = state.get("user_question", "")
    intent = state.get("intent", "text")
    if _report_requested(question):
        return "visualization_node"
    if intent == "both":
        return "visualization_node"
    return "end"

def _route_after_visualization(state: AnalystState) -> str:
    if state.get("error"):
        return "end"
    if _report_requested(state.get("user_question", "")):
        return "report_generator_node"
    return "end"

def build_workflow():
    graph = StateGraph(AnalystState)
    graph.add_node("load_dataset_node", load_dataset_node)
    graph.add_node("intent_classifier_node", intent_classifier_node)
    graph.add_node("text_analysis_node", text_analysis_node)
    graph.add_node("visualization_node", visualization_node)
    graph.add_node("report_generator_node", report_generator_node)
    graph.set_entry_point("load_dataset_node")
    graph.add_edge("load_dataset_node", "intent_classifier_node")
    graph.add_conditional_edges("intent_classifier_node", route_by_intent, {
        "text_analysis_node": "text_analysis_node",
        "visualization_node": "visualization_node",
        "report_generator_node": "report_generator_node",
    })
    graph.add_conditional_edges("text_analysis_node", _route_after_text, {
        "visualization_node": "visualization_node",
        "end": END,
    })
    graph.add_conditional_edges("visualization_node", _route_after_visualization, {
        "report_generator_node": "report_generator_node",
        "end": END,
    })
    graph.add_edge("report_generator_node", END)
    return graph.compile()

_app = None

def get_workflow():
    global _app
    if _app is None:
        _app = build_workflow()
    return _app

def run_analysis(dataframe, user_question: str, chat_history: list | None = None, session_id: str = "default") -> AnalystState:
    app = get_workflow()
    initial_state: AnalystState = {
        "dataframe": dataframe,
        "user_question": user_question,
        "session_id": session_id,
        "dataset_summary": None,
        "intent": None,
        "text_response": None,
        "chart_paths": [],
        "chart_captions": [],
        "report_path": None,
        "chat_history": chat_history or [],
        "error": None,
    }
    return app.invoke(initial_state)

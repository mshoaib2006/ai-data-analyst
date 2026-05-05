from typing import TypedDict, Optional, List, Dict, Any

class AnalystState(TypedDict):
    dataframe: Optional[object]
    user_question: str
    session_id: str
    dataset_summary: Optional[Dict[str, Any]]
    intent: Optional[str]
    text_response: Optional[str]
    chart_paths: List[str]
    chart_captions: List[str]
    report_path: Optional[str]
    chat_history: List[dict]
    error: Optional[str]

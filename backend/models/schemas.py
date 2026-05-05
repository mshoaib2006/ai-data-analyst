from typing import List, Optional
from pydantic import BaseModel

class UploadResponse(BaseModel):
    session_id: str
    dataset_id: str
    filename: str
    rows: int
    columns: int

class AskRequest(BaseModel):
    session_id: str
    dataset_id: str
    question: str

class AskResponse(BaseModel):
    session_id: str
    dataset_id: str
    intent: str
    text_response: str
    chart_urls: List[str]
    chart_captions: List[str]
    report_url: Optional[str] = None

import os
import uuid
from typing import Any, Dict

DATASET_STORE: Dict[str, Dict[str, Any]] = {}
SESSION_STORE: Dict[str, Dict[str, Any]] = {}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHARTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "charts"))
REPORTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "reports"))

os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"

def create_session() -> str:
    session_id = new_id("sess")
    SESSION_STORE[session_id] = {"dataset_id": None, "chat_history": []}
    return session_id

def ensure_session(session_id: str | None) -> str:
    if session_id and session_id in SESSION_STORE:
        return session_id
    return create_session()

def attach_dataset_to_session(session_id: str, dataset_id: str) -> None:
    SESSION_STORE.setdefault(session_id, {"dataset_id": None, "chat_history": []})
    SESSION_STORE[session_id]["dataset_id"] = dataset_id
    SESSION_STORE[session_id]["chat_history"] = []

def add_dataset(dataset_id: str, dataframe, filename: str) -> None:
    DATASET_STORE[dataset_id] = {"dataframe": dataframe, "filename": filename}

def get_dataset(dataset_id: str):
    item = DATASET_STORE.get(dataset_id)
    return item["dataframe"] if item else None

def get_session_dataset_id(session_id: str) -> str | None:
    item = SESSION_STORE.get(session_id)
    return item.get("dataset_id") if item else None

def get_chat_history(session_id: str) -> list[dict]:
    item = SESSION_STORE.get(session_id)
    return item.get("chat_history", []) if item else []

def set_chat_history(session_id: str, history: list[dict]) -> None:
    SESSION_STORE.setdefault(session_id, {"dataset_id": None, "chat_history": []})
    SESSION_STORE[session_id]["chat_history"] = history

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

def get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

def get_optional_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()

def get_openai_model() -> str:
    return get_optional_env("OPENAI_MODEL", "gpt-4o-mini")

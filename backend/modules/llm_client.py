from openai import OpenAI
from modules.settings import get_required_env

_client = None

def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_required_env("OPENAI_API_KEY"))
    return _client

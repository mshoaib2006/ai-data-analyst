import html
import os
import re
import uuid
from typing import Any

import requests
import streamlit as st


API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8001/api")
API_HOST = API_BASE[:-4] if API_BASE.endswith("/api") else API_BASE


st.set_page_config(
    page_title="AI Data Analyst",
    
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
html, body, [class*="css"] {
  background-color: #f7f9fc !important;
}

.block-container {
  max-width: 980px;
  padding-top: 2rem;
  padding-bottom: 3rem;
}

.msg-user {
  background: #2563eb;
  color: #fff;
  padding: 12px 16px;
  border-radius: 16px 16px 4px 16px;
  margin: 10px 0 10px 18%;
  line-height: 1.45;
  white-space: normal;
}

.msg-ai {
  background: #fff;
  color: #111827;
  padding: 14px 16px;
  border-radius: 4px 16px 16px 16px;
  margin: 10px 18% 10px 0;
  border: 1px solid #dbe3ef;
  line-height: 1.45;
  white-space: normal;
}

.ai-content {
  margin-top: 10px;
  font-size: 15px;
  line-height: 1.45;
}

.ai-heading {
  font-weight: 700;
  font-size: 15px;
  color: #111827;
  margin-top: 14px;
  margin-bottom: 6px;
  text-transform: uppercase;
}

.ai-heading:first-child {
  margin-top: 0;
}

.ai-paragraph {
  margin: 0 0 7px 0;
  line-height: 1.45;
}

.ai-bullet {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  margin: 0 0 6px 0;
  line-height: 1.45;
}

.ai-bullet-dot {
  min-width: 8px;
  color: #111827;
  font-weight: 700;
}

.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  margin-left: 8px;
}

.badge-text {
  background:#ecfdf5;
  color:#065f46;
}

.badge-chart {
  background:#eff6ff;
  color:#1d4ed8;
}

.badge-both {
  background:#f5f3ff;
  color:#6d28d9;
}

.chart-caption {
  color: #4b5563;
  font-size: 12px;
  margin-top: -6px;
  margin-bottom: 12px;
}

.dataset-card {
  background: #ffffff;
  border: 1px solid #dbe3ef;
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 12px;
}

.dataset-meta {
  color: #4b5563;
  font-size: 13px;
}
</style>
""",
    unsafe_allow_html=True,
)


def _init() -> None:
    defaults = {
        "session_id": None,
        "dataset_id": None,
        "chat_history": [],
        "df_name": None,
        "rows": None,
        "columns": None,
        "uploaded_signature": None,
        "_pending": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_text(value: Any) -> str:
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    return value


def _clean_markdown_text(text: Any) -> str:
    value = _safe_text(text).strip()

    if not value:
        return ""

    # Remove markdown bold/italic
    value = value.replace("**", "")
    value = value.replace("__", "")

    # Remove markdown headings
    value = re.sub(r"^\s{0,3}#{1,6}\s*", "", value, flags=re.MULTILINE)

    # Remove backticks
    value = value.replace("`", "")

    # Convert bullet symbols
    value = value.replace("•", "-")

    # Remove useless ending lines
    bad_endings = [
        "If you need further analysis or details, let me know!",
        "Let me know if you need further analysis.",
        "Let me know if you need more details.",
        "If you need further assistance, let me know!",
    ]

    for ending in bad_endings:
        value = value.replace(ending, "").strip()

    # Remove extra spaces around lines
    lines = []
    for line in value.splitlines():
        clean_line = line.strip()
        lines.append(clean_line)

    # Collapse too many blank lines
    output_lines = []
    previous_blank = False

    for line in lines:
        is_blank = line == ""

        if is_blank and previous_blank:
            continue

        output_lines.append(line)
        previous_blank = is_blank

    value = "\n".join(output_lines)

    # Remove blank line before bullet
    value = re.sub(r"\n\s*\n\s*(-\s+)", r"\n\1", value)

    # Remove blank line after headings if any
    value = re.sub(r"([A-Z][A-Z\s]{3,})\n\s*\n", r"\1\n", value)

    return value.strip()


def _is_heading(line: str) -> bool:
    clean = line.strip().strip(":")

    if not clean:
        return False

    known_headings = {
        "EXECUTIVE SUMMARY",
        "DATASET HEALTH",
        "TARGET COLUMN",
        "KEY FINDINGS",
        "IMPORTANT COLUMNS",
        "WEAK OR LESS USEFUL COLUMNS",
        "RECOMMENDATIONS",
        "ANALYSIS",
        "SUMMARY",
        "DATA QUALITY",
        "VISUAL ANALYSIS",
    }

    if clean.upper() in known_headings:
        return True

    if clean.isupper() and 3 <= len(clean) <= 45:
        return True

    return False


def _message_to_html(text: Any) -> str:
    clean = _clean_markdown_text(text)

    if not clean:
        return ""

    html_parts = []

    for raw_line in clean.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        escaped = html.escape(line)

        if _is_heading(line):
            html_parts.append(f'<div class="ai-heading">{escaped}</div>')
            continue

        if line.startswith("- "):
            item = html.escape(line[2:].strip())
            html_parts.append(
                f'<div class="ai-bullet"><span class="ai-bullet-dot">•</span><span>{item}</span></div>'
            )
            continue

        if re.match(r"^\d+\.\s+", line):
            html_parts.append(f'<div class="ai-paragraph">{escaped}</div>')
            continue

        html_parts.append(f'<div class="ai-paragraph">{escaped}</div>')

    return "\n".join(html_parts)


def _badge(intent: str) -> str:
    return {
        "text": '<span class="badge badge-text">text</span>',
        "chart": '<span class="badge badge-chart">chart</span>',
        "both": '<span class="badge badge-both">both</span>',
    }.get(intent, "")


def _file_signature(uploaded_file) -> str:
    return f"{uploaded_file.name}_{uploaded_file.size}"


def _safe_get(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def _upload_dataset(uploaded_file) -> None:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "text/csv",
        )
    }

    params = {}

    if st.session_state.session_id:
        params["session_id"] = st.session_state.session_id

    response = requests.post(
        f"{API_BASE}/upload",
        files=files,
        params=params,
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    st.session_state.session_id = data["session_id"]
    st.session_state.dataset_id = data["dataset_id"]
    st.session_state.df_name = data["filename"]
    st.session_state.rows = data["rows"]
    st.session_state.columns = data["columns"]
    st.session_state.chat_history = []
    st.session_state.uploaded_signature = _file_signature(uploaded_file)


def _render_assistant(turn: dict) -> None:
    content_html = _message_to_html(turn.get("content", ""))

    st.markdown(
        f"""
<div class="msg-ai">
  <strong>AI Analyst</strong> {_badge(turn.get("intent", ""))}
  <div class="ai-content">
    {content_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    chart_urls = turn.get("chart_urls", []) or []
    chart_captions = turn.get("chart_captions", []) or []

    for idx, chart_url in enumerate(chart_urls):
        full_chart_url = f"{API_HOST}{chart_url}"
        st.image(full_chart_url, use_container_width=True)

        if idx < len(chart_captions):
            st.markdown(
                f'<div class="chart-caption">{html.escape(_safe_text(chart_captions[idx]))}</div>',
                unsafe_allow_html=True,
            )

    report_url = turn.get("report_url")

    if report_url:
        try:
            report_bytes = _safe_get(f"{API_HOST}{report_url}")

            st.download_button(
                "Download PDF Report",
                data=report_bytes,
                file_name=f"report_{uuid.uuid4().hex[:8]}.pdf",
                mime="application/pdf",
                key=f"report_{report_url}",
            )

        except Exception as exc:
            st.error(f"Failed to fetch report: {exc}")


_init()


with st.sidebar:
    st.title(" AI Data Analyst")
  

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        current_sig = _file_signature(uploaded)

        if st.session_state.uploaded_signature != current_sig:
            try:
                _upload_dataset(uploaded)
                st.success(f"Loaded {st.session_state.df_name}")
                st.caption(f"{st.session_state.rows:,} rows · {st.session_state.columns} columns")

            except Exception as exc:
                st.error(f"Upload failed: {exc}")

        elif st.session_state.df_name:
            st.success(f"Loaded {st.session_state.df_name}")
            st.caption(f"{st.session_state.rows:,} rows · {st.session_state.columns} columns")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Clear Dataset", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.dataset_id = None
        st.session_state.chat_history = []
        st.session_state.df_name = None
        st.session_state.rows = None
        st.session_state.columns = None
        st.session_state.uploaded_signature = None
        st.session_state["_pending"] = None
        st.rerun()

    st.divider()
    st.markdown("Example questions")

    examples = [
        "Summarize the dataset",
        "Create a full report",
        "Show correlation heatmap",
        "Find missing values",
        "Show distribution of target column",
        "Which columns are important?",
        "Which columns are not important?",
        "Are there duplicate records?",
    ]

    for q in examples:
        if st.button(q, use_container_width=True, key=f"ex_{q}"):
            st.session_state["_pending"] = q
            st.rerun()


st.title(" AI Data Analyst")


if not st.session_state.dataset_id:
    st.info("Upload a CSV file from the sidebar to start.")

else:
    st.markdown(
        f"""
<div class="dataset-card">
  <div><strong>Dataset:</strong> {html.escape(_safe_text(st.session_state.df_name))}</div>
  <div class="dataset-meta">Rows: {st.session_state.rows:,} | Columns: {st.session_state.columns}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            user_html = _message_to_html(turn["content"])
            st.markdown(
                f"""
<div class="msg-user">
  <strong>You</strong>
  <div class="ai-content">
    {user_html}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            _render_assistant(turn)

    pending = st.session_state.pop("_pending", None)
    prompt = st.chat_input("Ask a question about your dataset") or pending

    if prompt:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "session_id": st.session_state.session_id,
            "dataset_id": st.session_state.dataset_id,
            "question": prompt,
        }

        with st.spinner("Analyzing dataset..."):
            try:
                response = requests.post(
                    f"{API_BASE}/ask",
                    json=payload,
                    timeout=180,
                )

                response.raise_for_status()
                data = response.json()

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": data.get("text_response", ""),
                        "intent": data.get("intent", "text"),
                        "chart_urls": data.get("chart_urls", []),
                        "chart_captions": data.get("chart_captions", []),
                        "report_url": data.get("report_url"),
                    }
                )

            except Exception as exc:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": f"Error: {exc}",
                        "intent": "text",
                        "chart_urls": [],
                        "chart_captions": [],
                        "report_url": None,
                    }
                )

        st.rerun()

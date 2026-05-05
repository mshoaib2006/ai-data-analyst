# AI Data Analyst System with FastAPI Backend

## Run backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# add OPENAI_API_KEY in .env
uvicorn main:app --reload
```

## Run frontend
```bash
cd frontend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Backend default URL: `http://127.0.0.1:8000`

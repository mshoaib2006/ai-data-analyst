# backend/routes/upload.py

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

from models.schemas import UploadResponse
from services.store import (
    new_id,
    ensure_session,
    attach_dataset_to_session,
    add_dataset,
)

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...), session_id: str | None = None):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        from io import BytesIO
        df = pd.read_csv(BytesIO(contents))

        # IMPORTANT FIX: clean column names
        df.columns = [str(col).strip() for col in df.columns]

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")

    session_id = ensure_session(session_id)
    dataset_id = new_id("ds")

    add_dataset(dataset_id, df, file.filename)
    attach_dataset_to_session(session_id, dataset_id)

    return UploadResponse(
        session_id=session_id,
        dataset_id=dataset_id,
        filename=file.filename,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
    )
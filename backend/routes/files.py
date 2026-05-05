import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from services.store import CHARTS_DIR, REPORTS_DIR

router = APIRouter()

@router.get("/files/chart/{filename}")
async def get_chart(filename: str):
    path = os.path.join(CHARTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chart not found.")
    return FileResponse(path, media_type="image/png")

@router.get("/files/report/{filename}")
async def get_report(filename: str):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(path, media_type="application/pdf", filename=filename)

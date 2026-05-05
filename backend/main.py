from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.upload import router as upload_router
from routes.ask import router as ask_router
from routes.files import router as files_router

app = FastAPI(title="AI Data Analyst Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api")
app.include_router(ask_router, prefix="/api")
app.include_router(files_router, prefix="/api")

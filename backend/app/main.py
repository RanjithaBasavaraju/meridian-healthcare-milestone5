import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import HTTPException
from sqlalchemy import text

from .database import init_db, engine
from .ml.predictor import get_predictor
from .routers import dashboard, jobs, predict, upload

app = FastAPI(
    title="Healthcare Cost Prediction & Resource Planning API",
    description=(
        "Predicts per-patient healthcare cost from Medicare-claims-style "
        "features, supports single and bulk (async) scoring, and powers "
        "the resource-planning dashboard."
    ),
    version="1.0.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(dashboard.router)


@app.on_event("startup")
def on_startup():
    init_db()
    get_predictor()  # trains/loads the model once at boot so first request isn't slow


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "Healthcare Cost Prediction API is running.",
        "docs": "/docs",
    }
@app.get("/health/live")
def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
def readiness():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "ready",
            "database": "connected"
        }

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Database connection unavailable"
        )
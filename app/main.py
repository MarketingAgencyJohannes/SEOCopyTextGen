from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import create_tables
from app.services.job_store import get_job
from app.routers import health, agent1, agent2, agent3, agent4

app = FastAPI(
    title="SEO Copy Text Generator",
    description="4-agent pipeline: Channel Scraper → Transcriber → SERP Analyzer → Copy Generator",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if settings.api_secret_key and request.url.path not in EXEMPT_PATHS:
        provided = request.headers.get("X-API-Key", "")
        if provided != settings.api_secret_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing X-API-Key header"})
    return await call_next(request)


@app.on_event("startup")
def on_startup():
    create_tables()


@app.get("/jobs/{job_id}", tags=["Jobs"])
def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


app.include_router(health.router, tags=["System"])
app.include_router(agent1.router, prefix="/agent1", tags=["Agent 1 — Channel Scraper"])
app.include_router(agent2.router, prefix="/agent2", tags=["Agent 2 — SERP Analyzer"])
app.include_router(agent3.router, prefix="/agent3", tags=["Agent 3 — Transcriber"])
app.include_router(agent4.router, prefix="/agent4", tags=["Agent 4 — Copy Generator"])

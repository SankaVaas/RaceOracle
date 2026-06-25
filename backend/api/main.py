"""
RaceOracle FastAPI application entry point.

Run:
    uvicorn backend.api.main:app --reload --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from backend.utils.logger import logger

app = FastAPI(
    title       = "RaceOracle API",
    description = "AI-powered horse racing prediction platform with surgical deep learning.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS — allow React frontend on localhost:3000 and any casino white-label domain
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    logger.info("RaceOracle API starting up...")
    logger.info("Docs available at http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown():
    logger.info("RaceOracle API shutting down.")


@app.get("/", tags=["system"])
def root():
    return {
        "name":    "RaceOracle API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
        "demo":    "/api/v1/predict/demo",
    }
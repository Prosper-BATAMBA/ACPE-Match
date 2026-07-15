from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routers import candidates, job_offers, matching, stats, export_csv


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="ACPE Match API",
    description="Backend centralisé pour le matching candidats/offres d'emploi",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(candidates.router)
app.include_router(job_offers.router)
app.include_router(matching.router)
app.include_router(stats.router)
app.include_router(export_csv.router)


@app.get("/")
def root():
    return {"message": "ACPE Match API", "docs": "/docs"}

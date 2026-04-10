import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import settings
from .database import create_tables
from .routers import matches, auth, bets
from .routers import scraper as scraper_router
from .scrapers.scheduler import start_scheduler, stop_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Arranque
    create_tables()
    start_scheduler()
    log.info("Football Predictor API arriba")
    yield
    # Apagado
    stop_scheduler()
    log.info("Football Predictor API apagada")


app = FastAPI(
    title="Football Predictor API",
    description=(
        "Plataforma de análisis y predicción de fútbol. "
        "Datos scrapeados de Sofascore — Premier League, La Liga, "
        "UEFA Champions League y Liga BetPlay DIMAYOR."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,            prefix="/api")
app.include_router(matches.router,         prefix="/api")
app.include_router(bets.router,            prefix="/api")
app.include_router(scraper_router.router,  prefix="/api")


@app.get("/", tags=["health"])
async def root():
    from .database import SessionLocal
    from .models import Match, League, Team, Player
    db = SessionLocal()
    try:
        return {
            "status":  "ok",
            "version": "2.0.0",
            "db": {
                "leagues": db.query(League).count(),
                "teams":   db.query(Team).count(),
                "players": db.query(Player).count(),
                "matches": db.query(Match).count(),
            },
        }
    finally:
        db.close()


@app.get("/api/leagues", tags=["meta"])
async def list_leagues():
    from .database import SessionLocal
    from .models import League, Season
    db = SessionLocal()
    try:
        leagues = db.query(League).filter_by(is_active=True).all()
        result = []
        for lg in leagues:
            season = db.query(Season).filter_by(league_id=lg.id, is_current=True).first()
            result.append({
                "id":         lg.id,
                "sofascore_id": lg.sofascore_id,
                "name":       lg.name,
                "country":    lg.country,
                "logo":       lg.logo_url,
                "avg_goals":  lg.avg_goals,
                "season":     season.year if season else None,
            })
        return result
    finally:
        db.close()

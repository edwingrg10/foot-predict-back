"""
Endpoints para disparar y monitorear el scraping manualmente.
"""
from __future__ import annotations

import threading
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

log = logging.getLogger("router.scraper")
router = APIRouter(prefix="/scraper", tags=["scraper"])


class JobResponse(BaseModel):
    ok: bool
    message: str


# ── Status ───────────────────────────────────────────────────────────────────
@router.get("/status")
async def scraper_status():
    from ..scrapers.scheduler import get_jobs
    from ..database import SessionLocal
    from ..models import ScrapingLog, Match, League, Team, Player

    db = SessionLocal()
    try:
        last_jobs = (
            db.query(ScrapingLog)
            .order_by(ScrapingLog.created_at.desc())
            .limit(10)
            .all()
        )
        return {
            "scheduler_jobs": get_jobs(),
            "db_counts": {
                "leagues": db.query(League).count(),
                "teams":   db.query(Team).count(),
                "players": db.query(Player).count(),
                "matches": db.query(Match).count(),
            },
            "last_jobs": [
                {
                    "type":     j.job_type,
                    "target":   j.target,
                    "status":   j.status,
                    "inserted": j.records_inserted,
                    "started":  str(j.started_at),
                    "finished": str(j.finished_at),
                    "error":    j.error_message,
                }
                for j in last_jobs
            ],
        }
    finally:
        db.close()


# ── Inicialización ────────────────────────────────────────────────────────────
@router.post("/init", response_model=JobResponse)
async def trigger_init(background_tasks: BackgroundTasks):
    """
    Primera ejecución: inicializa ligas, temporadas, standings y partidos de hoy.
    Ejecutar UNA sola vez al arrancar el sistema.
    """
    def _run():
        from ..scrapers.runner import run_init
        run_init()

    background_tasks.add_task(_run)
    return JobResponse(ok=True, message="Init scraping iniciado en background")


# ── Scraping diario manual ────────────────────────────────────────────────────
@router.post("/daily", response_model=JobResponse)
async def trigger_daily(background_tasks: BackgroundTasks):
    """Dispara el scraping diario manualmente."""
    def _run():
        from ..scrapers.runner import run_daily
        run_daily()

    background_tasks.add_task(_run)
    return JobResponse(ok=True, message="Daily scraping iniciado en background")


# ── Carga histórica ───────────────────────────────────────────────────────────
@router.post("/historical", response_model=JobResponse)
async def trigger_historical(
    background_tasks: BackgroundTasks,
    years_back: int = Query(3, ge=1, le=10),
):
    """
    Carga histórica de las últimas N temporadas.
    ADVERTENCIA: puede tardar varias horas. Ejecutar solo una vez.
    """
    def _run():
        from ..scrapers.runner import run_historical
        run_historical(years_back=years_back)

    background_tasks.add_task(_run)
    return JobResponse(
        ok=True,
        message=f"Carga histórica ({years_back} temporadas) iniciada en background"
    )


# ── Scraping por fecha ────────────────────────────────────────────────────────
@router.post("/date/{date_str}", response_model=JobResponse)
async def trigger_date(date_str: str, background_tasks: BackgroundTasks):
    """Scrape partidos de una fecha específica (YYYY-MM-DD)."""
    import re
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        raise HTTPException(400, "Formato de fecha inválido. Usar YYYY-MM-DD")

    def _run():
        from ..scrapers.base import client
        from ..scrapers.matches import scrape_daily
        from ..database import SessionLocal
        client.start()
        db = SessionLocal()
        try:
            n = scrape_daily(db, date_str)
            log.info(f"[Router] Scraping {date_str}: {n} partidos")
        finally:
            db.close()
            client.stop()

    background_tasks.add_task(_run)
    return JobResponse(ok=True, message=f"Scraping de {date_str} iniciado en background")


# ── Detalles de partidos pendientes ──────────────────────────────────────────
@router.post("/details", response_model=JobResponse)
async def trigger_details(
    background_tasks: BackgroundTasks,
    limit: int = Query(50, ge=1, le=200),
):
    """Scrape stats, eventos y alineaciones de partidos terminados pendientes."""
    def _run():
        from ..scrapers.base import client
        from ..scrapers.details import scrape_pending_matches
        from ..database import SessionLocal
        client.start()
        db = SessionLocal()
        try:
            done = scrape_pending_matches(db, limit=limit)
            log.info(f"[Router] Detalles completados: {done}")
        finally:
            db.close()
            client.stop()

    background_tasks.add_task(_run)
    return JobResponse(ok=True, message=f"Scraping de detalles (max {limit}) iniciado en background")

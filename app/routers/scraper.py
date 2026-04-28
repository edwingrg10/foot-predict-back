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

# Evita que dos jobs de scraping corran en paralelo (Playwright es single-instance)
import threading
_scraping_lock = threading.Lock()
_scraping_running = False


def _acquire_scraping() -> bool:
    global _scraping_running
    with _scraping_lock:
        if _scraping_running:
            return False
        _scraping_running = True
        return True


def _release_scraping():
    global _scraping_running
    with _scraping_lock:
        _scraping_running = False


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
    if not _acquire_scraping():
        return JobResponse(ok=False, message="Ya hay un scraping en curso. Espera a que termine.")

    def _run():
        try:
            from ..scrapers.runner import run_init
            run_init()
        finally:
            _release_scraping()

    background_tasks.add_task(_run)
    return JobResponse(ok=True, message="Init scraping iniciado en background")


# ── Scraping diario manual ────────────────────────────────────────────────────
@router.post("/daily", response_model=JobResponse)
async def trigger_daily(background_tasks: BackgroundTasks):
    """Dispara el scraping diario manualmente."""
    if not _acquire_scraping():
        return JobResponse(ok=False, message="Ya hay un scraping en curso. Espera a que termine.")

    def _run():
        try:
            from ..scrapers.runner import run_daily
            run_daily()
        finally:
            _release_scraping()

    background_tasks.add_task(_run)
    return JobResponse(ok=True, message="Daily scraping iniciado en background")


# ── Carga histórica ───────────────────────────────────────────────────────────
@router.post("/historical", response_model=JobResponse)
async def trigger_historical(
    background_tasks: BackgroundTasks,
    years_back: int = Query(3, ge=1, le=10),
    retrain: bool = Query(True, description="Re-entrenar modelos al terminar"),
    league_ids: str = Query(None, description="Sofascore IDs separados por coma, ej: 11536,11539. Vacío = todas las ligas"),
):
    """
    Carga histórica de las últimas N temporadas y re-entrena los modelos.
    Puede tardar varias horas dependiendo de cuántos partidos haya que scrapear.
    """
    if not _acquire_scraping():
        return JobResponse(ok=False, message="Ya hay un scraping en curso. Espera a que termine.")

    parsed_ids = None
    if league_ids:
        try:
            parsed_ids = [int(x.strip()) for x in league_ids.split(",")]
        except ValueError:
            _release_scraping()
            return JobResponse(ok=False, message="league_ids inválido. Usar números separados por coma.")

    def _run():
        try:
            from ..scrapers.runner import run_historical
            from ..database import SessionLocal
            run_historical(years_back=years_back, league_ids=parsed_ids)

            if retrain:
                log.info("[Historical] Scraping terminado — evaluando y re-entrenando modelos...")
                db = SessionLocal()
                try:
                    from ..ml2.evaluator import evaluate_finished_matches
                    from ..ml2.trainer import train_all
                    evaluate_finished_matches(db)
                    result = train_all(db)
                    if result:
                        log.info(f"[Historical] Modelos re-entrenados: {list(result.keys())}")
                    else:
                        log.warning("[Historical] Dataset insuficiente para re-entrenar")
                finally:
                    db.close()
        finally:
            _release_scraping()

    background_tasks.add_task(_run)
    return JobResponse(
        ok=True,
        message=f"Carga histórica ({years_back} temporadas) iniciada en background. "
                f"{'Re-entrenamiento automático al terminar.' if retrain else ''} "
                f"Monitorea el progreso en GET /scraper/status"
    )


# ── Scraping por fecha ────────────────────────────────────────────────────────
@router.post("/date/{date_str}", response_model=JobResponse)
async def trigger_date(date_str: str, background_tasks: BackgroundTasks):
    """Scrape partidos de una fecha específica (YYYY-MM-DD)."""
    import re
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        raise HTTPException(400, "Formato de fecha inválido. Usar YYYY-MM-DD")

    if not _acquire_scraping():
        return JobResponse(ok=False, message="Ya hay un scraping en curso. Espera a que termine.")

    def _run():
        from ..scrapers.base import client
        from ..scrapers.matches import scrape_daily
        from ..database import SessionLocal
        try:
            client.start()
            db = SessionLocal()
            try:
                n = scrape_daily(db, date_str)
                log.info(f"[Router] Scraping {date_str}: {n} partidos")
            finally:
                db.close()
                client.stop()
        finally:
            _release_scraping()

    background_tasks.add_task(_run)
    return JobResponse(ok=True, message=f"Scraping de {date_str} iniciado en background")


# ── Detalles de partidos pendientes ──────────────────────────────────────────
@router.get("/debug-events/{date_str}")
async def debug_events(date_str: str):
    """
    Diagnóstico: devuelve los primeros 5 eventos de Sofascore para una fecha
    y muestra qué campos de torneo vienen en la respuesta.
    """
    from fastapi.concurrency import run_in_threadpool
    from ..scrapers.base import client

    def _fetch():
        from playwright.sync_api import sync_playwright
        results = {}
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx     = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
                locale="es-ES",
            )
            page = ctx.new_page()
            try:
                page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=20000)
            except Exception:
                pass

            url = f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{date_str}"
            raw = page.evaluate(f"""async () => {{
                const r = await fetch('{url}', {{
                    headers: {{
                        'Accept': 'application/json',
                        'Origin': 'https://www.sofascore.com',
                        'Referer': 'https://www.sofascore.com/',
                    }}
                }});
                const status = r.status;
                let body = null;
                try {{ body = await r.json(); }} catch(e) {{}}
                return {{ status, body }};
            }}""")
            browser.close()

        if not raw:
            return {"error": "evaluate devolvió None"}

        status = raw.get("status")
        body   = raw.get("body") or {}

        if status != 200:
            return {"error": f"HTTP {status} de Sofascore", "status": status}

        events = body.get("events", [])
        if not events:
            return {"status": status, "total_events": 0, "message": "Sin eventos para esta fecha"}

        sample = []
        for ev in events[:10]:
            t = ev.get("tournament", {})
            sample.append({
                "match":       f"{ev.get('homeTeam',{}).get('name')} vs {ev.get('awayTeam',{}).get('name')}",
                "ev_status":   ev.get("status", {}).get("type"),
                "tournament":  {
                    "name":             t.get("name"),
                    "uniqueTournament": t.get("uniqueTournament"),
                    "category_name":    t.get("category", {}).get("name"),
                },
            })
        return {"http_status": status, "total_events": len(events), "sample": sample}

    return await run_in_threadpool(_fetch)


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


# ── Evaluación de predicciones ────────────────────────────────────────────────
@router.post("/evaluate", response_model=JobResponse)
async def trigger_evaluate():
    """Evalúa ahora todas las predicciones de partidos ya terminados."""
    from ..database import SessionLocal
    from ..ml2.evaluator import evaluate_finished_matches

    db = SessionLocal()
    try:
        result = evaluate_finished_matches(db)
        evaluated = result.get("evaluated", 0)
        skipped   = result.get("skipped", 0)
        return JobResponse(
            ok=True,
            message=f"Evaluación completada: {evaluated} evaluadas, {skipped} sin datos suficientes.",
        )
    finally:
        db.close()

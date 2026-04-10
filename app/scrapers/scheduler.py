"""
Scheduler — ejecuta el scraping diario automáticamente.
Se lanza al arrancar FastAPI.
"""
from __future__ import annotations

import logging
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

log = logging.getLogger("scraper.scheduler")

_scheduler: BackgroundScheduler | None = None


def _run_daily_safe():
    """Wrapper que captura errores para que el scheduler no muera."""
    try:
        from .runner import run_daily
        run_daily()
    except Exception as e:
        log.error(f"[Scheduler] Error en daily job: {e}", exc_info=True)


def start_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    _scheduler = BackgroundScheduler(timezone="America/Bogota")

    # Scraping completo a las 6:00 AM hora Colombia
    _scheduler.add_job(
        _run_daily_safe,
        CronTrigger(hour=6, minute=0),
        id="daily_morning",
        name="Scraping matutino",
        replace_existing=True,
    )

    # Segunda pasada a las 23:00 para capturar resultados del día
    _scheduler.add_job(
        _run_daily_safe,
        CronTrigger(hour=23, minute=0),
        id="daily_night",
        name="Scraping nocturno (resultados)",
        replace_existing=True,
    )

    _scheduler.start()
    log.info("[Scheduler] Scheduler iniciado — jobs: 6:00 AM y 11:00 PM (Bogotá)")


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("[Scheduler] Scheduler detenido")


def get_jobs() -> list[dict]:
    if not _scheduler:
        return []
    return [
        {
            "id":       job.id,
            "name":     job.name,
            "next_run": str(job.next_run_time),
        }
        for job in _scheduler.get_jobs()
    ]

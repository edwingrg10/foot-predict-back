"""
Runner — orquesta todos los scrapers.

  run_daily()      → partidos hoy/mañana + detalles pendientes + standings
  run_historical() → carga completa de temporadas pasadas
  run_init()       → primera ejecución: ligas + temporadas + carga inicial
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from ..database import SessionLocal
from ..models import ScrapingLog, ScrapingStatus, League, Season, Match, MatchStatus
from .base import client
from .leagues import scrape_all_leagues
from .matches import scrape_daily, scrape_season
from .details import scrape_pending_matches, scrape_match_full
from .standings import scrape_all_standings, scrape_standings
from .config import TARGET_LEAGUES

log = logging.getLogger("scraper.runner")


def _log_job(db, job_type: str, target: str = "") -> ScrapingLog:
    entry = ScrapingLog(job_type=job_type, target=target, status=ScrapingStatus.RUNNING)
    import datetime
    entry.started_at = datetime.datetime.now()
    db.add(entry)
    db.commit()
    return entry


def _finish_log(db, entry: ScrapingLog, inserted: int = 0, error: str = None):
    import datetime
    entry.finished_at       = datetime.datetime.now()
    entry.records_inserted  = inserted
    entry.status = ScrapingStatus.FAILED if error else ScrapingStatus.DONE
    entry.error_message = error
    db.commit()


# ── Job diario ───────────────────────────────────────────────────────────────
def run_daily():
    """
    Ejecutar cada día:
      1. Partidos de hoy y mañana
      2. Detalles (stats, eventos, lineups) de partidos terminados pendientes
      3. Standings actualizados
    """
    log.info("=" * 60)
    log.info("DAILY SCRAPING START")
    log.info("=" * 60)

    client.start()
    db = SessionLocal()
    total = 0

    try:
        today    = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()

        # 1. Partidos
        entry = _log_job(db, "daily_matches", today)
        try:
            n  = scrape_daily(db, today)
            n += scrape_daily(db, tomorrow)
            total += n
            _finish_log(db, entry, inserted=n)
            log.info(f"[Runner] Partidos scrapeados: {n}")
        except Exception as e:
            _finish_log(db, entry, error=str(e))
            log.error(f"[Runner] Error partidos: {e}")

        # 2. Detalles de partidos terminados
        entry = _log_job(db, "match_details", today)
        try:
            done = scrape_pending_matches(db, limit=30)
            _finish_log(db, entry, inserted=done)
            log.info(f"[Runner] Detalles completados: {done}")
        except Exception as e:
            _finish_log(db, entry, error=str(e))
            log.error(f"[Runner] Error detalles: {e}")

        # 3. Standings
        entry = _log_job(db, "standings", today)
        try:
            n = scrape_all_standings(db)
            _finish_log(db, entry, inserted=n)
            log.info(f"[Runner] Standings actualizados: {n}")
        except Exception as e:
            _finish_log(db, entry, error=str(e))
            log.error(f"[Runner] Error standings: {e}")

    finally:
        db.close()
        client.stop()
        log.info(f"DAILY SCRAPING DONE — total={total}")


# ── Inicialización ───────────────────────────────────────────────────────────
def run_init():
    """
    Primera ejecución:
      1. Crear ligas y temporadas en DB
      2. Scrape standings actuales
      3. Scrape partidos de hoy/mañana
    """
    log.info("=" * 60)
    log.info("INIT SCRAPING START")
    log.info("=" * 60)

    client.start()
    db = SessionLocal()

    try:
        # 1. Ligas y temporadas
        entry = _log_job(db, "init_leagues")
        try:
            scrape_all_leagues(db)
            _finish_log(db, entry)
            log.info("[Runner] Ligas y temporadas inicializadas")
        except Exception as e:
            _finish_log(db, entry, error=str(e))
            log.error(f"[Runner] Error init ligas: {e}")

        # 2. Standings
        entry = _log_job(db, "init_standings")
        try:
            n = scrape_all_standings(db)
            _finish_log(db, entry, inserted=n)
        except Exception as e:
            _finish_log(db, entry, error=str(e))

        # 3. Partidos de hoy y mañana
        today    = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        entry = _log_job(db, "init_matches", today)
        try:
            n  = scrape_daily(db, today)
            n += scrape_daily(db, tomorrow)
            _finish_log(db, entry, inserted=n)
            log.info(f"[Runner] Partidos iniciales: {n}")
        except Exception as e:
            _finish_log(db, entry, error=str(e))

    finally:
        db.close()
        client.stop()
        log.info("INIT SCRAPING DONE")


# ── Carga histórica ──────────────────────────────────────────────────────────
def run_historical(years_back: int = 3):
    """
    Carga histórica de las últimas N temporadas de cada liga.
    Esto puede tardar varias horas — ejecutar solo una vez.
    """
    log.info("=" * 60)
    log.info(f"HISTORICAL SCRAPING START (last {years_back} seasons)")
    log.info("=" * 60)

    client.start()
    db = SessionLocal()

    try:
        for cfg in TARGET_LEAGUES:
            league = db.query(League).filter_by(sofascore_id=cfg.sofascore_id).first()
            if not league:
                log.warning(f"[Runner] Liga no encontrada: {cfg.name} — ejecuta run_init() primero")
                continue

            # Obtener temporadas no scrapeadas aún
            seasons = (
                db.query(Season)
                .filter_by(league_id=league.id, scraped_full=False)
                .order_by(Season.id.desc())
                .limit(years_back)
                .all()
            )

            for season in seasons:
                log.info(f"[Runner] Scraping {cfg.name} {season.year}...")
                entry = _log_job(db, "historical", f"{cfg.sofascore_id}/{season.sofascore_id}")
                try:
                    n = scrape_season(db, cfg.sofascore_id, season.sofascore_id)
                    _finish_log(db, entry, inserted=n)
                    log.info(f"[Runner] {cfg.name} {season.year}: {n} partidos")
                except Exception as e:
                    _finish_log(db, entry, error=str(e))
                    log.error(f"[Runner] Error histórico {cfg.name} {season.year}: {e}")

            # Después de cargar partidos, scrape los detalles
            log.info(f"[Runner] Scraping detalles históricos {cfg.name}...")
            entry = _log_job(db, "historical_details", cfg.name)
            try:
                # Hacer en lotes de 50
                total_done = 0
                while True:
                    done = scrape_pending_matches(db, limit=50)
                    total_done += done
                    if done == 0:
                        break
                _finish_log(db, entry, inserted=total_done)
            except Exception as e:
                _finish_log(db, entry, error=str(e))

    finally:
        db.close()
        client.stop()
        log.info("HISTORICAL SCRAPING DONE")

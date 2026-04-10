"""
Scraper de ligas y temporadas.
Pobla las tablas: leagues, seasons
"""
from __future__ import annotations

import logging
from datetime import date
from sqlalchemy.orm import Session

from ..models import League, Season
from .base import client
from .config import TARGET_LEAGUES, LeagueConfig

log = logging.getLogger("scraper.leagues")


def _upsert_league(db: Session, cfg: LeagueConfig) -> League:
    league = db.query(League).filter_by(sofascore_id=cfg.sofascore_id).first()
    if not league:
        league = League(
            sofascore_id=cfg.sofascore_id,
            name=cfg.name,
            country=cfg.country,
            logo_url=cfg.logo_url,
            avg_goals=cfg.avg_goals,
        )
        db.add(league)
        db.flush()
        log.info(f"[Leagues] Creada liga: {cfg.name}")
    else:
        league.name      = cfg.name
        league.logo_url  = cfg.logo_url
        league.avg_goals = cfg.avg_goals
    return league


def _upsert_season(db: Session, league: League, raw: dict) -> Season:
    sf_id = raw["id"]
    season = db.query(Season).filter_by(
        league_id=league.id, sofascore_id=sf_id
    ).first()

    year_str = raw.get("year", str(sf_id))
    # Determinar si es la temporada actual (la más reciente)
    if not season:
        season = Season(
            league_id=league.id,
            sofascore_id=sf_id,
            year=year_str,
        )
        db.add(season)
        log.info(f"[Leagues] Nueva temporada: {league.name} {year_str}")
    return season


def scrape_all_leagues(db: Session) -> dict[int, int]:
    """
    Devuelve {sofascore_league_id: db_league_id} para uso posterior.
    """
    league_map: dict[int, int] = {}

    for cfg in TARGET_LEAGUES:
        log.info(f"[Leagues] Procesando {cfg.name}...")
        league = _upsert_league(db, cfg)

        # Traer temporadas desde Sofascore
        data = client.fetch(f"/unique-tournament/{cfg.sofascore_id}/seasons")
        if not data or "seasons" not in data:
            log.warning(f"[Leagues] Sin temporadas para {cfg.name}")
            db.commit()
            league_map[cfg.sofascore_id] = league.id
            continue

        seasons_raw = data["seasons"]

        # Marcar todas como no-actuales primero
        db.query(Season).filter_by(league_id=league.id).update({"is_current": False})

        # La primera temporada en la lista es la más reciente
        for i, raw in enumerate(seasons_raw):
            season = _upsert_season(db, league, raw)
            if i == 0:
                season.is_current = True

        db.commit()
        league_map[cfg.sofascore_id] = league.id
        log.info(f"[Leagues] {cfg.name}: {len(seasons_raw)} temporadas procesadas")

    return league_map

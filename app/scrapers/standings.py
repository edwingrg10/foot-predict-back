"""
Scraper de clasificaciones (tablas de posiciones).
"""
from __future__ import annotations

import logging
from sqlalchemy.orm import Session

from ..models import League, Season, Team, Standing
from .base import client
from .teams import upsert_team

log = logging.getLogger("scraper.standings")


def scrape_standings(db: Session, sofascore_league_id: int, sofascore_season_id: int) -> int:
    league = db.query(League).filter_by(sofascore_id=sofascore_league_id).first()
    season = db.query(Season).filter_by(
        league_id=league.id if league else None,
        sofascore_id=sofascore_season_id
    ).first() if league else None

    if not league or not season:
        log.error(f"[Standings] Liga/temporada no encontrada: {sofascore_league_id}/{sofascore_season_id}")
        return 0

    data = client.fetch(
        f"/unique-tournament/{sofascore_league_id}/season/{sofascore_season_id}/standings/total"
    )
    if not data or "standings" not in data:
        log.warning(f"[Standings] Sin datos para {league.name}")
        return 0

    count = 0
    for standing_group in data["standings"]:
        for row in standing_group.get("rows", []):
            team_raw = row.get("team", {})
            team = upsert_team(db, team_raw, league_id=league.id)
            if not team:
                continue

            standing = (
                db.query(Standing)
                .filter_by(season_id=season.id, team_id=team.id)
                .first()
            )
            if not standing:
                standing = Standing(
                    league_id=league.id,
                    season_id=season.id,
                    team_id=team.id,
                )
                db.add(standing)

            standing.position       = row.get("position")
            standing.matches_played = row.get("matches")
            standing.wins           = row.get("wins")
            standing.draws          = row.get("draws")
            standing.losses         = row.get("losses")
            standing.goals_for      = row.get("scoresFor")
            standing.goals_against  = row.get("scoresAgainst")
            standing.goal_diff      = (row.get("scoresFor") or 0) - (row.get("scoresAgainst") or 0)
            standing.points         = row.get("points")
            count += 1

    db.commit()
    log.info(f"[Standings] {league.name}: {count} posiciones actualizadas")
    return count


def scrape_all_standings(db: Session) -> int:
    """Actualiza standings de todas las ligas activas (temporada actual)."""
    from .config import TARGET_LEAGUES
    total = 0
    for cfg in TARGET_LEAGUES:
        league = db.query(League).filter_by(sofascore_id=cfg.sofascore_id).first()
        if not league:
            continue
        season = db.query(Season).filter_by(league_id=league.id, is_current=True).first()
        if not season:
            continue
        total += scrape_standings(db, cfg.sofascore_id, season.sofascore_id)
    return total

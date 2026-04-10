"""
Scraper de equipos y jugadores.
Pobla las tablas: teams, players
"""
from __future__ import annotations

import logging
from sqlalchemy.orm import Session

from ..models import Team, Player, League
from .base import client

log = logging.getLogger("scraper.teams")


def upsert_team(db: Session, raw: dict, league_id: int | None = None) -> Team:
    sf_id = raw.get("id")
    if not sf_id:
        return None

    team = db.query(Team).filter_by(sofascore_id=sf_id).first()
    if not team:
        team = Team(
            sofascore_id=sf_id,
            name=raw.get("name", ""),
            short_name=raw.get("shortName") or raw.get("name", "")[:30],
            logo_url=f"https://img.sofascore.com/api/v1/team/{sf_id}/image",
            country=raw.get("country", {}).get("name") if raw.get("country") else None,
            league_id=league_id,
        )
        db.add(team)
        db.flush()
        log.debug(f"[Teams] Nuevo equipo: {team.name}")
    else:
        team.name       = raw.get("name", team.name)
        team.short_name = raw.get("shortName") or team.short_name
        if league_id:
            team.league_id = league_id
    return team


def upsert_player(db: Session, raw: dict, team_id: int | None = None) -> Player | None:
    sf_id = raw.get("id")
    if not sf_id:
        return None

    player = db.query(Player).filter_by(sofascore_id=sf_id).first()
    if not player:
        player = Player(
            sofascore_id=sf_id,
            name=raw.get("name", ""),
            short_name=raw.get("shortName") or raw.get("name", "")[:50],
            logo_url=f"https://img.sofascore.com/api/v1/player/{sf_id}/image",
            team_id=team_id,
            position=_map_position(raw.get("position")),
        )
        db.add(player)
        db.flush()
    else:
        player.name = raw.get("name", player.name)
        if team_id:
            player.team_id = team_id
    return player


def scrape_team_details(db: Session, sofascore_team_id: int) -> Team | None:
    """Enriquece un equipo con info adicional (estadio, fundación, etc.)"""
    data = client.fetch(f"/team/{sofascore_team_id}")
    if not data or "team" not in data:
        return None

    raw  = data["team"]
    team = db.query(Team).filter_by(sofascore_id=sofascore_team_id).first()
    if not team:
        return None

    venue = raw.get("venue") or {}
    team.stadium          = venue.get("name")
    team.stadium_capacity = venue.get("capacity")
    team.founded_year     = raw.get("foundationDateTimestamp")  # puede ser timestamp o año

    db.flush()
    return team


def scrape_squad(db: Session, sofascore_team_id: int, team_id: int) -> int:
    """Scrape la plantilla de un equipo. Retorna cantidad de jugadores."""
    data = client.fetch(f"/team/{sofascore_team_id}/players")
    if not data or "players" not in data:
        return 0

    count = 0
    for item in data["players"]:
        raw_player = item.get("player", item)
        player = upsert_player(db, raw_player, team_id=team_id)
        if player:
            # Actualizar número de camiseta si viene
            player.shirt_number = item.get("shirtNumber")
            player.position     = _map_position(item.get("position") or raw_player.get("position"))
            count += 1

    db.flush()
    log.info(f"[Teams] Equipo {sofascore_team_id}: {count} jugadores")
    return count


def _map_position(pos: str | None) -> str | None:
    if not pos:
        return None
    mapping = {
        "G":  "Goalkeeper",
        "D":  "Defender",
        "M":  "Midfielder",
        "F":  "Forward",
        "GK": "Goalkeeper",
        "DF": "Defender",
        "MF": "Midfielder",
        "FW": "Forward",
        "goalkeeper": "Goalkeeper",
        "defender":   "Defender",
        "midfielder": "Midfielder",
        "forward":    "Forward",
        "attacker":   "Forward",
    }
    return mapping.get(pos, pos)

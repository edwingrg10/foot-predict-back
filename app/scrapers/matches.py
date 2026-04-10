"""
Scraper de partidos.
- scrape_daily(date_str)  → partidos de un día concreto
- scrape_season_round()   → todos los partidos de una temporada por rondas
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, date, timedelta
from typing import Optional

COT = timezone(timedelta(hours=-5))  # Colombia UTC-5

from sqlalchemy.orm import Session

from ..models import Match, League, Season, MatchStatus, MatchImportance
from .base import client
from .config import LEAGUES_BY_ID
from .teams import upsert_team
from .referees import upsert_referee

log = logging.getLogger("scraper.matches")


# ── Mapeo de estados Sofascore → MatchStatus ─────────────────────────────────
_STATUS_MAP = {
    "notstarted":      MatchStatus.SCHEDULED,
    "inprogress":      MatchStatus.LIVE,
    "halftime":        MatchStatus.HALFTIME,
    "finished":        MatchStatus.FINISHED,
    "postponed":       MatchStatus.POSTPONED,
    "canceled":        MatchStatus.CANCELLED,
    "cancelled":       MatchStatus.CANCELLED,
    "abandoned":       MatchStatus.ABANDONED,
    "extra time":      MatchStatus.LIVE,
    "awaiting extra":  MatchStatus.LIVE,
    "penalties":       MatchStatus.LIVE,
}


def _parse_status(ev: dict) -> MatchStatus:
    stype = ev.get("status", {}).get("type", "notstarted").lower()
    return _STATUS_MAP.get(stype, MatchStatus.SCHEDULED)


def _parse_dt(ts: int | None) -> datetime:
    if ts:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    return datetime.now(tz=timezone.utc)


def _get_or_create_season(db: Session, league: League) -> Season | None:
    return (
        db.query(Season)
        .filter_by(league_id=league.id, is_current=True)
        .first()
    )


def _upsert_match(db: Session, ev: dict, league: League, season: Season | None) -> Match | None:
    sf_id = ev.get("id")
    if not sf_id:
        return None

    # Equipos
    home_raw = ev.get("homeTeam", {})
    away_raw = ev.get("awayTeam", {})
    home_team = upsert_team(db, home_raw, league_id=league.id)
    away_team = upsert_team(db, away_raw, league_id=league.id)
    if not home_team or not away_team:
        return None

    # Árbitro (puede no venir en partidos programados)
    ref_raw  = ev.get("referee")
    referee  = upsert_referee(db, ref_raw)

    # Marcador
    home_score    = ev.get("homeScore", {}).get("current")
    away_score    = ev.get("awayScore", {}).get("current")
    home_score_ht = ev.get("homeScore", {}).get("period1")
    away_score_ht = ev.get("awayScore", {}).get("period1")
    home_score_et = ev.get("homeScore", {}).get("extra1")
    away_score_et = ev.get("awayScore", {}).get("extra1")
    home_pen      = ev.get("homeScore", {}).get("penalties")
    away_pen      = ev.get("awayScore", {}).get("penalties")

    # Ronda
    round_info = ev.get("roundInfo", {})
    round_str  = round_info.get("nameKey") or (
        f"Ronda {round_info['round']}" if round_info.get("round") else None
    )

    match = db.query(Match).filter_by(sofascore_id=sf_id).first()
    if not match:
        match = Match(
            sofascore_id=sf_id,
            league_id=league.id,
            season_id=season.id if season else None,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            referee_id=referee.id if referee else None,
            match_date=_parse_dt(ev.get("startTimestamp")),
            round=round_str,
            status=_parse_status(ev),
            home_score=home_score,
            away_score=away_score,
            home_score_ht=home_score_ht,
            away_score_ht=away_score_ht,
            home_score_et=home_score_et,
            away_score_et=away_score_et,
            home_score_pen=home_pen,
            away_score_pen=away_pen,
            venue=ev.get("venue", {}).get("name") if ev.get("venue") else None,
        )
        db.add(match)
        db.flush()
        log.debug(f"[Matches] Nuevo partido: {home_team.name} vs {away_team.name} ({match_date_str(match)})")
    else:
        # Actualizar estado y marcador (cambia en partidos en vivo / terminados)
        match.status        = _parse_status(ev)
        match.home_score    = home_score
        match.away_score    = away_score
        match.home_score_ht = home_score_ht
        match.away_score_ht = away_score_ht
        match.home_score_et = home_score_et
        match.away_score_et = away_score_et
        match.home_score_pen = home_pen
        match.away_score_pen = away_pen
        if referee:
            match.referee_id = referee.id

    return match


def match_date_str(m: Match) -> str:
    return m.match_date.strftime("%Y-%m-%d") if m.match_date else "?"


# ── Scraping diario ──────────────────────────────────────────────────────────
def scrape_daily(db: Session, date_str: str) -> int:
    """
    Scrape todos los partidos de nuestras ligas para una fecha dada.
    Retorna cantidad de partidos insertados/actualizados.
    """
    log.info(f"[Matches] Scraping diario: {date_str}")
    data = client.fetch(f"/sport/football/scheduled-events/{date_str}")
    if not data or "events" not in data:
        log.warning(f"[Matches] Sin eventos para {date_str}")
        return 0

    target_date = date.fromisoformat(date_str)
    count = 0

    for ev in data["events"]:
        # Filtrar por fecha en hora colombiana (no UTC) para incluir
        # partidos nocturnos que en UTC caen al día siguiente
        ts = ev.get("startTimestamp")
        if ts:
            ev_date = datetime.fromtimestamp(ts, tz=COT).date()
            if ev_date != target_date:
                continue

        # Identificar la liga
        tournament = ev.get("tournament", {})
        category   = tournament.get("category", {})
        league_cfg = _match_tournament(tournament, category)
        if not league_cfg:
            continue

        league = db.query(League).filter_by(sofascore_id=league_cfg.sofascore_id).first()
        if not league:
            log.warning(f"[Matches] Liga no encontrada en DB: {league_cfg.name}")
            continue

        season = _get_or_create_season(db, league)
        match  = _upsert_match(db, ev, league, season)
        if match:
            count += 1

    db.commit()
    log.info(f"[Matches] {date_str}: {count} partidos procesados")
    return count


def scrape_today(db: Session) -> int:
    return scrape_daily(db, datetime.now(COT).date().isoformat())


def scrape_tomorrow(db: Session) -> int:
    return scrape_daily(db, (datetime.now(COT).date() + timedelta(days=1)).isoformat())


# ── Scraping histórico por rondas ────────────────────────────────────────────
def scrape_season(db: Session, sofascore_league_id: int, sofascore_season_id: int) -> int:
    """
    Scrape completo de una temporada iterando todas las rondas.
    """
    league = db.query(League).filter_by(sofascore_id=sofascore_league_id).first()
    if not league:
        log.error(f"[Matches] Liga {sofascore_league_id} no encontrada en DB")
        return 0

    season = db.query(Season).filter_by(
        league_id=league.id, sofascore_id=sofascore_season_id
    ).first()
    if not season:
        log.error(f"[Matches] Temporada {sofascore_season_id} no encontrada en DB")
        return 0

    log.info(f"[Matches] Scraping temporada {league.name} {season.year}...")

    total = 0
    round_num = 1
    consecutive_empty = 0

    while consecutive_empty < 3:
        data = client.fetch(
            f"/unique-tournament/{sofascore_league_id}"
            f"/season/{sofascore_season_id}/events/round/{round_num}"
        )
        if not data or not data.get("events"):
            consecutive_empty += 1
            round_num += 1
            continue

        consecutive_empty = 0
        events = data["events"]
        round_count = 0

        for ev in events:
            match = _upsert_match(db, ev, league, season)
            if match:
                round_count += 1

        db.commit()
        log.info(f"[Matches] {league.name} {season.year} — Ronda {round_num}: {round_count} partidos")
        total     += round_count
        round_num += 1

    season.scraped_full = True
    db.commit()
    log.info(f"[Matches] Temporada completa: {total} partidos")
    return total


# ── Helper: identificar liga por torneo ──────────────────────────────────────
def _match_tournament(tournament: dict, category: dict):
    """
    Identifica si un evento pertenece a una de nuestras ligas objetivo.
    Estrategia: primero por uniqueTournament.id (más preciso), luego por keywords.
    """
    from .config import LEAGUES_BY_ID

    # 1) Coincidencia exacta por uniqueTournament.id (la más fiable)
    unique_id = tournament.get("uniqueTournament", {}).get("id")
    if unique_id and unique_id in LEAGUES_BY_ID:
        return LEAGUES_BY_ID[unique_id]

    # 2) Fallback por keywords — más estricto que antes
    t_name   = tournament.get("name", "").lower()
    cat_name = category.get("name", "").lower()

    _STRICT_KEYWORDS = {
        # La Liga Primera División — excluir Hypermotion, SmartBank, etc.
        8:     [("laliga", ["hypermotion", "smartbank", "2", "ii", "promises", "easports"])],
        # Premier League inglesa — excluir Premier League 2, Asia Trophy
        17:    [("premier league", ["2", "ii", "asia", "cup", "summer", "u21", "u23", "international"])],
        # Champions League — excluir Youth, Women
        7:     [("champions league", ["youth", "women", "uefa super"])],
        # Liga BetPlay DIMAYOR
        11536: [("dimayor", []), ("betplay", [])],
    }
    _COUNTRY_REQ = {17: "england", 8: "spain", 7: "europe", 11536: "colombia"}

    for lid, rules in _STRICT_KEYWORDS.items():
        cfg = LEAGUES_BY_ID.get(lid)
        if not cfg:
            continue
        for keyword, excludes in rules:
            if keyword not in t_name:
                continue
            # Verificar país
            required = _COUNTRY_REQ.get(lid)
            if required and required not in cat_name:
                continue
            # Verificar que no contenga términos excluidos
            if any(ex in t_name for ex in excludes):
                continue
            return cfg
    return None

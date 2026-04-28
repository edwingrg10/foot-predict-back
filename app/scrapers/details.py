"""
Scraper de detalles de partido:
  - Estadísticas (MatchStats)
  - Eventos: goles, tarjetas, sustituciones (MatchEvent)
  - Alineaciones + stats individuales (MatchLineup)
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from ..models import Match, MatchStats, MatchEvent, MatchLineup, EventType, MatchStatus
from .base import client
from .teams import upsert_player

log = logging.getLogger("scraper.details")


# ── Estadísticas del partido ─────────────────────────────────────────────────
def scrape_match_stats(db: Session, match: Match) -> bool:
    data = client.fetch(f"/event/{match.sofascore_id}/statistics")
    if not data or "statistics" not in data:
        return False

    # Usar el período ALL (totales)
    period_all = next(
        (p for p in data["statistics"] if p.get("period") == "ALL"),
        data["statistics"][0] if data["statistics"] else None,
    )
    if not period_all:
        return False

    # Aplanar todos los items en un dict {name: {home, away}}
    items: dict[str, dict] = {}
    for group in period_all.get("groups", []):
        for item in group.get("statisticsItems", []):
            key = item.get("key") or item.get("name", "").lower().replace(" ", "_")
            items[key] = {
                "home": _parse_stat(item.get("home")),
                "away": _parse_stat(item.get("away")),
            }

    def h(key):
        return items.get(key, {}).get("home")

    def a(key):
        return items.get(key, {}).get("away")

    stats = db.query(MatchStats).filter_by(match_id=match.id).first()
    if not stats:
        stats = MatchStats(match_id=match.id)
        db.add(stats)

    stats.home_possession         = h("ballPossession") or h("ball_possession")
    stats.away_possession         = a("ballPossession") or a("ball_possession")
    stats.home_shots              = h("totalShots") or h("total_shots")
    stats.away_shots              = a("totalShots") or a("total_shots")
    stats.home_shots_on_target    = h("shotsOnTarget") or h("shots_on_target")
    stats.away_shots_on_target    = a("shotsOnTarget") or a("shots_on_target")
    stats.home_shots_off_target   = h("shotsOffTarget") or h("shots_off_target")
    stats.away_shots_off_target   = a("shotsOffTarget") or a("shots_off_target")
    stats.home_blocked_shots      = h("blockedShots") or h("blocked_shots")
    stats.away_blocked_shots      = a("blockedShots") or a("blocked_shots")
    stats.home_xg                 = h("expectedGoals") or h("expected_goals")
    stats.away_xg                 = a("expectedGoals") or a("expected_goals")
    stats.home_corners            = h("cornerKicks") or h("corner_kicks")
    stats.away_corners            = a("cornerKicks") or a("corner_kicks")
    stats.home_fouls              = h("fouls")
    stats.away_fouls              = a("fouls")
    stats.home_yellow_cards       = h("yellowCards") or h("yellow_cards")
    stats.away_yellow_cards       = a("yellowCards") or a("yellow_cards")
    stats.home_red_cards          = h("redCards") or h("red_cards")
    stats.away_red_cards          = a("redCards") or a("red_cards")
    stats.home_passes             = h("totalPasses") or h("total_passes")
    stats.away_passes             = a("totalPasses") or a("total_passes")
    stats.home_pass_accuracy      = h("accuratePasses") or h("accurate_passes")
    stats.away_pass_accuracy      = a("accuratePasses") or a("accurate_passes")
    stats.home_tackles            = h("tackles")
    stats.away_tackles            = a("tackles")
    stats.home_interceptions      = h("interceptions")
    stats.away_interceptions      = a("interceptions")
    stats.home_attacks            = h("attacks")
    stats.away_attacks            = a("attacks")
    stats.home_dangerous_attacks  = h("dangerousAttacks") or h("dangerous_attacks")
    stats.away_dangerous_attacks  = a("dangerousAttacks") or a("dangerous_attacks")
    stats.home_aerial_won         = h("aerialWon") or h("aerial_won")
    stats.away_aerial_won         = a("aerialWon") or a("aerial_won")

    match.stats_scraped = True
    db.flush()
    log.debug(f"[Details] Stats guardadas: match {match.sofascore_id}")
    return True


def _parse_stat(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.replace("%", "").strip()
        try:
            return float(val)
        except ValueError:
            return None
    return None


# ── Eventos del partido ──────────────────────────────────────────────────────
_INCIDENT_MAP = {
    "goal":             EventType.GOAL,
    "ownGoal":          EventType.OWN_GOAL,
    "penaltyScored":    EventType.PENALTY_GOAL,
    "penaltyMissed":    EventType.PENALTY_MISS,
    "yellowCard":       EventType.YELLOW_CARD,
    "redCard":          EventType.RED_CARD,
    "yellowRedCard":    EventType.YELLOW_RED,
    "substitution":     EventType.SUBSTITUTION,
    "varDecision":      EventType.VAR,
}


def scrape_match_events(db: Session, match: Match) -> int:
    data = client.fetch(f"/event/{match.sofascore_id}/incidents")
    if not data or "incidents" not in data:
        return 0

    # Borrar eventos previos para re-insertarlos limpios
    db.query(MatchEvent).filter_by(match_id=match.id).delete()

    count = 0
    for inc in data["incidents"]:
        incident_type = inc.get("incidentType", "")
        etype = _INCIDENT_MAP.get(incident_type)
        if not etype:
            continue

        player_raw  = inc.get("player")
        player2_raw = inc.get("playerIn") or inc.get("assist1")

        player  = upsert_player(db, player_raw)  if player_raw  else None
        player2 = upsert_player(db, player2_raw) if player2_raw else None

        # Si es sustitución: player = sale, player2 = entra
        if etype == EventType.SUBSTITUTION:
            player_out_raw = inc.get("playerOut")
            player_in_raw  = inc.get("playerIn")
            player  = upsert_player(db, player_out_raw) if player_out_raw else player
            player2 = upsert_player(db, player_in_raw)  if player_in_raw  else player2

        is_home = inc.get("isHome")
        team_id = match.home_team_id if is_home else match.away_team_id

        event = MatchEvent(
            match_id=match.id,
            player_id=player.id  if player  else None,
            player2_id=player2.id if player2 else None,
            team_id=team_id,
            event_type=etype,
            minute=inc.get("time"),
            extra_time=inc.get("addedTime", 0),
            is_home=is_home,
            description=inc.get("description"),
        )
        db.add(event)
        count += 1

    match.events_scraped = True
    db.flush()
    log.debug(f"[Details] {count} eventos guardados: match {match.sofascore_id}")
    return count


# ── Alineaciones ─────────────────────────────────────────────────────────────
def scrape_match_lineups(db: Session, match: Match) -> int:
    data = client.fetch(f"/event/{match.sofascore_id}/lineups")
    if not data:
        return 0

    # Borrar alineaciones previas
    db.query(MatchLineup).filter_by(match_id=match.id).delete()

    count = 0
    for is_home, side_key in [(True, "home"), (False, "away")]:
        side = data.get(side_key, {})
        team_id = match.home_team_id if is_home else match.away_team_id

        players_raw = side.get("players", [])
        for item in players_raw:
            raw_player = item.get("player", {})
            player = upsert_player(db, raw_player, team_id=team_id)
            if not player:
                continue

            stats = item.get("statistics", {})
            lineup = MatchLineup(
                match_id=match.id,
                team_id=team_id,
                player_id=player.id,
                is_home=is_home,
                is_starter=item.get("position") != "S",  # S = suplente
                shirt_number=item.get("shirtNumber"),
                position=item.get("positionName") or item.get("position"),
                minutes_played=stats.get("minutesPlayed"),
                rating=_parse_stat(stats.get("rating")),
                goals=stats.get("goals", 0),
                assists=stats.get("goalAssist", 0),
                shots=stats.get("totalShots", 0) or stats.get("onTargetScoringAttempt", 0),
                shots_on_target=stats.get("onTargetScoringAttempt", 0),
                yellow_cards=stats.get("yellowCard", 0),
                red_cards=stats.get("redCard", 0) or stats.get("redCardSecondYellow", 0),
                passes=stats.get("totalPass", 0),
                pass_accuracy=_parse_stat(stats.get("accuratePassesPercentage")),
                dribbles=stats.get("wonContest", 0),
                tackles=stats.get("challengeWon", 0) or stats.get("totalTackle", 0),
            )
            db.add(lineup)
            count += 1

    match.lineups_scraped = True
    db.flush()
    log.debug(f"[Details] {count} alineaciones guardadas: match {match.sofascore_id}")
    return count


# ── Pipeline completo para un partido ────────────────────────────────────────
def scrape_match_full(db: Session, match: Match) -> dict:
    """Scrape stats + eventos + alineaciones de un partido terminado."""
    if match.status != MatchStatus.FINISHED:
        return {"skipped": "not finished"}

    result = {}
    if not match.stats_scraped:
        result["stats"]   = scrape_match_stats(db, match)
    if not match.events_scraped:
        result["events"]  = scrape_match_events(db, match)
    if not match.lineups_scraped:
        result["lineups"] = scrape_match_lineups(db, match)

    db.commit()
    return result


def scrape_pending_matches(db: Session, limit: int = 50) -> int:
    """
    Scrape detalles de todos los partidos terminados que aún no tienen stats.
    Usa una sesión nueva por partido para evitar que Supabase cierre la conexión
    durante loops largos.
    """
    from ..database import SessionLocal

    pending_ids = (
        db.query(Match.id, Match.sofascore_id)
        .filter(
            Match.status == MatchStatus.FINISHED,
            Match.stats_scraped == False,
        )
        .order_by(Match.match_date.desc())
        .limit(limit)
        .all()
    )

    log.info(f"[Details] {len(pending_ids)} partidos pendientes de detalles")
    done = 0
    for match_id, sofascore_id in pending_ids:
        fresh_db = SessionLocal()
        try:
            match = fresh_db.query(Match).filter_by(id=match_id).first()
            if not match:
                continue
            result = scrape_match_full(fresh_db, match)
            if result.get("stats") or result.get("events"):
                done += 1
                log.info(
                    f"[Details] {sofascore_id} OK — "
                    f"stats={result.get('stats')}, events={result.get('events')}, "
                    f"lineups={result.get('lineups')}"
                )
        except Exception as e:
            log.error(f"[Details] Error en match {sofascore_id}: {e}")
        finally:
            fresh_db.close()

    return done

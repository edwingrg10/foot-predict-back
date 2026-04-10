"""
Router de partidos — lee de SQL Server y aplica predicciones con IA.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Optional

# Colombia = UTC-5
COT = timezone(timedelta(hours=-5))

from fastapi import APIRouter, Query, HTTPException, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Match, League, Season, Team, MatchStatus, Prediction

router = APIRouter(prefix="/matches", tags=["matches"])


# ── Serialización ─────────────────────────────────────────────────────────────
def _serialize_match(match: Match, db: Session) -> dict:
    pred = match.prediction
    home = match.home_team
    away = match.away_team
    lg   = match.league

    out = {
        "id":          match.id,
        "sofascore_id": match.sofascore_id,
        "match_date":  match.match_date.strftime("%Y-%m-%dT%H:%M:%SZ") if match.match_date else None,
        "round":       match.round,
        "status":      match.status.value if match.status else "scheduled",
        "venue":       match.venue,
        "home_score":  match.home_score,
        "away_score":  match.away_score,
        "home_score_ht": match.home_score_ht,
        "away_score_ht": match.away_score_ht,
        "league": {
            "id":      lg.id,
            "name":    lg.name,
            "country": lg.country,
            "logo":    lg.logo_url,
        } if lg else None,
        "home_team": {
            "id":   home.id,
            "name": home.name,
            "logo": home.logo_url,
        } if home else None,
        "away_team": {
            "id":   away.id,
            "name": away.name,
            "logo": away.logo_url,
        } if away else None,
        "prediction": _serialize_prediction(pred) if pred else None,
    }
    return out


def _serialize_prediction(pred: Prediction) -> dict:
    if not pred:
        return None
    value_bets = []
    if pred.value_bets:
        try:
            value_bets = json.loads(pred.value_bets)
        except Exception:
            pass
    return {
        "prob_home_win":          pred.prob_home_win,
        "prob_draw":              pred.prob_draw,
        "prob_away_win":          pred.prob_away_win,
        "prob_over_25":           pred.prob_over_25,
        "prob_under_25":          pred.prob_under_25,
        "prob_over_35":           pred.prob_over_35,
        "prob_under_35":          pred.prob_under_35,
        "prob_btts":              pred.prob_btts,
        "prob_no_btts":           pred.prob_no_btts,
        # Corners
        "prob_over_95_corners":   pred.prob_over_95_corners  or 0,
        "prob_under_95_corners":  pred.prob_under_95_corners or 0,
        "expected_home_corners":  pred.expected_home_corners or 0,
        "expected_away_corners":  pred.expected_away_corners or 0,
        # Tarjetas
        "prob_over_35_cards":     pred.prob_over_35_cards    or 0,
        "prob_under_35_cards":    pred.prob_under_35_cards   or 0,
        "expected_home_cards":    pred.expected_home_cards   or 0,
        "expected_away_cards":    pred.expected_away_cards   or 0,
        # Goles
        "expected_home_goals":    pred.expected_home_goals,
        "expected_away_goals":    pred.expected_away_goals,
        "predicted_score":        pred.predicted_score,
        "confidence_score":       pred.confidence_score,
        "risk_level":             pred.risk_level,
        "value_bets":             value_bets,
        "model_version":          pred.model_version,
    }


def _group_by_league(matches: list[Match], db: Session) -> list[dict]:
    groups: dict[int, dict] = {}
    for m in matches:
        lid = m.league_id
        if lid not in groups:
            lg = m.league
            groups[lid] = {
                "league_id":      lid,
                "league_name":    lg.name    if lg else "",
                "league_country": lg.country if lg else "",
                "league_logo":    lg.logo_url if lg else "",
                "matches": [],
            }
        groups[lid]["matches"].append(_serialize_match(m, db))
    return list(groups.values())


def _get_and_predict(db: Session, matches: list[Match]) -> list[Match]:
    """Para partidos sin predicción, genera una al vuelo."""
    from ..ml2.predictor import predict_and_save, models_exist
    if not models_exist():
        return matches
    for m in matches:
        if not m.prediction and m.status == MatchStatus.SCHEDULED:
            try:
                predict_and_save(db, m)
                db.refresh(m)
            except Exception:
                pass
    return matches


def _date_range_filter(q, date_str: str):
    # 00:00 COT (UTC-5) = 05:00 UTC. Usar naive UTC para compatibilidad con pyodbc.
    d = date.fromisoformat(date_str)
    dt = datetime(d.year, d.month, d.day, 5, 0, 0)  # 00:00 COT = 05:00 UTC naive
    next_dt = dt + timedelta(days=1)
    return q.filter(Match.match_date >= dt, Match.match_date < next_dt)


# ── Endpoints ────────────────────────────────────────────────────────────────
@router.get("/today")
async def get_today_matches(
    league_id: Optional[int] = Query(None),
    group_by_league: bool = Query(True),
    db: Session = Depends(get_db),
):
    today = datetime.now(COT).date().isoformat()
    return _matches_for_date(db, today, league_id, group_by_league)


@router.get("/tomorrow")
async def get_tomorrow_matches(
    league_id: Optional[int] = Query(None),
    group_by_league: bool = Query(True),
    db: Session = Depends(get_db),
):
    tomorrow = (datetime.now(COT).date() + timedelta(days=1)).isoformat()
    return _matches_for_date(db, tomorrow, league_id, group_by_league)


@router.get("/date/{date_str}")
async def get_matches_by_date(
    date_str: str,
    league_id: Optional[int] = Query(None),
    group_by_league: bool = Query(True),
    db: Session = Depends(get_db),
):
    return _matches_for_date(db, date_str, league_id, group_by_league)


def _matches_for_date(db, date_str, league_id, group_by_league):
    d = date.fromisoformat(date_str)
    dt = datetime(d.year, d.month, d.day, 5, 0, 0)  # 00:00 COT = 05:00 UTC naive
    next_dt = dt + timedelta(days=1)

    q = (
        db.query(Match)
        .filter(Match.match_date >= dt, Match.match_date < next_dt)
        .order_by(Match.match_date)
    )
    if league_id:
        q = q.filter(Match.league_id == league_id)

    matches = q.all()
    matches = _get_and_predict(db, matches)

    if group_by_league:
        return {"groups": _group_by_league(matches, db), "total": len(matches)}
    return {"matches": [_serialize_match(m, db) for m in matches], "total": len(matches)}


@router.get("/detail/{match_id}")
async def get_match_detail(match_id: int, db: Session = Depends(get_db)):
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        raise HTTPException(404, "Partido no encontrado")

    _get_and_predict(db, [match])
    db.refresh(match)

    result = _serialize_match(match, db)

    # Agregar stats si existen
    if match.stats:
        s = match.stats
        result["stats"] = {
            "home_possession": s.home_possession,
            "away_possession": s.away_possession,
            "home_shots": s.home_shots, "away_shots": s.away_shots,
            "home_shots_on_target": s.home_shots_on_target,
            "away_shots_on_target": s.away_shots_on_target,
            "home_xg": s.home_xg, "away_xg": s.away_xg,
            "home_corners": s.home_corners, "away_corners": s.away_corners,
            "home_yellow_cards": s.home_yellow_cards,
            "away_yellow_cards": s.away_yellow_cards,
            "home_red_cards": s.home_red_cards,
            "away_red_cards": s.away_red_cards,
        }

    # Agregar eventos
    result["events"] = [
        {
            "type":      e.event_type.value,
            "minute":    e.minute,
            "extra":     e.extra_time,
            "is_home":   e.is_home,
            "player":    e.player.name if e.player else None,
            "player2":   e.player2.name if e.player2 else None,
        }
        for e in sorted(match.events, key=lambda x: x.minute or 0)
    ] if match.events_scraped else []

    return result


@router.get("/search")
async def search_matches(
    q: str = Query(..., min_length=2),
    day: str = Query("today"),
    db: Session = Depends(get_db),
):
    cot_today = datetime.now(COT).date()
    d = cot_today if day == "today" else cot_today + timedelta(days=1)
    dt = datetime(d.year, d.month, d.day, 5, 0, 0)  # 00:00 COT = 05:00 UTC naive
    next_dt = dt + timedelta(days=1)

    matches = (
        db.query(Match)
        .join(Match.home_team)
        .filter(Match.match_date >= dt, Match.match_date < next_dt)
        .all()
    )

    q_lower = q.lower()
    filtered = [
        m for m in matches
        if q_lower in (m.home_team.name or "").lower()
        or q_lower in (m.away_team.name or "").lower()
        or q_lower in (m.league.name or "").lower()
    ]
    return {"groups": _group_by_league(filtered, db), "total": len(filtered)}


# ── Endpoint de predicción / entrenamiento ────────────────────────────────────
@router.post("/train", tags=["ml"])
async def train_models(db: Session = Depends(get_db)):
    """Entrena los modelos de IA con el historial de la DB."""
    from ..ml2.trainer import train_all
    results = train_all(db)
    if not results:
        raise HTTPException(422, "Dataset insuficiente para entrenar. Carga más partidos históricos.")
    return {"ok": True, "models": results}


@router.post("/predict-all", tags=["ml"])
async def predict_all(db: Session = Depends(get_db)):
    """Genera predicciones para todos los partidos próximos."""
    from ..ml2.predictor import predict_all_upcoming
    done = predict_all_upcoming(db)
    return {"ok": True, "predicted": done}

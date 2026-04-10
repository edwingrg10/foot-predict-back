"""
Feature Builder — convierte datos de la DB en vectores para el modelo.

Para cada partido genera ~60 features:
  - Forma reciente (últimos 5 partidos)
  - Promedios de goles casa/fuera (temporada)
  - xG promedio
  - Estadísticas de tarjetas y corners
  - H2H histórico
  - Posición en la tabla
  - Diferencia de calidad entre equipos
  - Factor árbitro
"""
from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..models import (
    Match, MatchStats, MatchEvent, Standing,
    EventType, MatchStatus, League, Season, Team, Referee
)


FEATURE_NAMES = [
    # Goles promedio (temporada actual)
    "home_goals_scored_avg",     "home_goals_conceded_avg",
    "away_goals_scored_avg",     "away_goals_conceded_avg",
    "home_goals_home_avg",       "home_goals_conceded_home_avg",
    "away_goals_away_avg",       "away_goals_conceded_away_avg",

    # xG promedio (últimas 5 jornadas)
    "home_xg_for_avg",           "home_xg_against_avg",
    "away_xg_for_avg",           "away_xg_against_avg",

    # Forma reciente (puntos en últimos 5 partidos, escala 0-15)
    "home_form_5",               "away_form_5",
    "home_form_home_5",          "away_form_away_5",

    # Tiros promedio
    "home_shots_avg",            "home_shots_on_target_avg",
    "away_shots_avg",            "away_shots_on_target_avg",

    # Posesión promedio
    "home_possession_avg",       "away_possession_avg",

    # Corners promedio
    "home_corners_avg",          "away_corners_avg",

    # Tarjetas promedio
    "home_yellow_avg",           "home_red_avg",
    "away_yellow_avg",           "away_red_avg",

    # Posición en tabla (normalizada 0-1, 1=1°)
    "home_table_position",       "away_table_position",
    "home_points",               "away_points",

    # Diferencias (home - away)
    "goal_diff_delta",           "form_delta",
    "xg_delta",                  "position_delta",

    # H2H (últimos 10 enfrentamientos)
    "h2h_home_win_rate",         "h2h_draw_rate",         "h2h_away_win_rate",
    "h2h_avg_goals",             "h2h_btts_rate",

    # Factor árbitro
    "referee_yellow_avg",        "referee_red_avg",
    "referee_home_win_rate",

    # Racha sin perder / sin ganar
    "home_unbeaten_streak",      "away_unbeaten_streak",
    "home_win_streak",           "away_win_streak",

    # Clean sheets recientes
    "home_clean_sheets_5",       "away_clean_sheets_5",

    # Over 2.5 rate reciente
    "home_over25_rate_5",        "away_over25_rate_5",

    # BTTS rate reciente
    "home_btts_rate_5",          "away_btts_rate_5",

    # Liga (one-hot)
    "league_premier",            "league_laliga",
    "league_champions",          "league_dimayor",

    # Corners avanzados (para modelo específico)
    "home_corners_home_avg",     "away_corners_away_avg",
    "h2h_corners_avg",
    "referee_corners_avg",

    # Tarjetas avanzadas (para modelo específico)
    "home_yellow_home_avg",      "away_yellow_away_avg",
    "home_red_home_avg",         "away_red_away_avg",
    "referee_yellow_per_team",   "referee_red_per_team",
    "h2h_cards_avg",
    "match_importance",          "derby_factor",
]


def build_features(db: Session, match: Match) -> Optional[np.ndarray]:
    """
    Construye el vector de features para un partido.
    Retorna None si no hay suficientes datos.
    """
    v: dict[str, float] = {k: 0.0 for k in FEATURE_NAMES}

    home_id   = match.home_team_id
    away_id   = match.away_team_id
    league_id = match.league_id
    match_dt  = match.match_date

    # ── Historial de partidos previos de cada equipo ─────────────────────────
    home_history = _get_team_history(db, home_id, match_dt, limit=20)
    away_history = _get_team_history(db, away_id, match_dt, limit=20)

    if len(home_history) < 3 or len(away_history) < 3:
        return None   # muy pocos datos

    # ── Goles promedio (temporada) ───────────────────────────────────────────
    v["home_goals_scored_avg"]      = _avg_goals_scored(home_history)
    v["home_goals_conceded_avg"]    = _avg_goals_conceded(home_history)
    v["away_goals_scored_avg"]      = _avg_goals_scored(away_history)
    v["away_goals_conceded_avg"]    = _avg_goals_conceded(away_history)

    home_h = [m for m in home_history if m.home_team_id == home_id]
    away_a = [m for m in away_history if m.away_team_id == away_id]

    v["home_goals_home_avg"]        = _avg_goals_scored_side(home_h, as_home=True)
    v["home_goals_conceded_home_avg"] = _avg_goals_conceded_side(home_h, as_home=True)
    v["away_goals_away_avg"]        = _avg_goals_scored_side(away_a, as_home=False)
    v["away_goals_conceded_away_avg"] = _avg_goals_conceded_side(away_a, as_home=False)

    # ── xG (últimas 10) ──────────────────────────────────────────────────────
    v["home_xg_for_avg"],  v["home_xg_against_avg"]  = _avg_xg(db, home_id, home_history[:10], as_home=True)
    v["away_xg_for_avg"],  v["away_xg_against_avg"]  = _avg_xg(db, away_id, away_history[:10], as_home=False)

    # ── Forma reciente (últimos 5) ───────────────────────────────────────────
    v["home_form_5"]      = _form_points(home_id, home_history[:5])
    v["away_form_5"]      = _form_points(away_id, away_history[:5])
    v["home_form_home_5"] = _form_points(home_id, home_h[:5])
    v["away_form_away_5"] = _form_points(away_id, away_a[:5])

    # ── Tiros y posesión (últimas 10) ────────────────────────────────────────
    home_stats = _get_match_stats(db, home_history[:10], home_id)
    away_stats = _get_match_stats(db, away_history[:10], away_id)

    v["home_shots_avg"]           = home_stats.get("shots", 0)
    v["home_shots_on_target_avg"] = home_stats.get("shots_on_target", 0)
    v["away_shots_avg"]           = away_stats.get("shots", 0)
    v["away_shots_on_target_avg"] = away_stats.get("shots_on_target", 0)
    v["home_possession_avg"]      = home_stats.get("possession", 50)
    v["away_possession_avg"]      = away_stats.get("possession", 50)
    v["home_corners_avg"]         = home_stats.get("corners", 0)
    v["away_corners_avg"]         = away_stats.get("corners", 0)
    v["home_yellow_avg"]          = home_stats.get("yellow_cards", 0)
    v["home_red_avg"]             = home_stats.get("red_cards", 0)
    v["away_yellow_avg"]          = away_stats.get("yellow_cards", 0)
    v["away_red_avg"]             = away_stats.get("red_cards", 0)

    # ── Tabla de posiciones ──────────────────────────────────────────────────
    season_id = match.season_id
    home_pos, home_pts, total_teams = _get_standing(db, home_id, season_id, league_id)
    away_pos, away_pts, _           = _get_standing(db, away_id, season_id, league_id)

    if total_teams > 0:
        v["home_table_position"] = 1 - (home_pos - 1) / max(total_teams - 1, 1)
        v["away_table_position"] = 1 - (away_pos - 1) / max(total_teams - 1, 1)
    v["home_points"] = home_pts
    v["away_points"] = away_pts

    # ── Diferencias ──────────────────────────────────────────────────────────
    v["goal_diff_delta"]  = v["home_goals_scored_avg"] - v["away_goals_scored_avg"]
    v["form_delta"]       = v["home_form_5"] - v["away_form_5"]
    v["xg_delta"]         = v["home_xg_for_avg"] - v["away_xg_for_avg"]
    v["position_delta"]   = v["home_table_position"] - v["away_table_position"]

    # ── H2H ──────────────────────────────────────────────────────────────────
    h2h = _get_h2h(db, home_id, away_id, match_dt, limit=10)
    if h2h:
        hw = sum(1 for m in h2h if _winner(m) == "home")
        dw = sum(1 for m in h2h if _winner(m) == "draw")
        aw = sum(1 for m in h2h if _winner(m) == "away")
        total_h2h = len(h2h)

        # Suavizado bayesiano: con pocos partidos, jalamos hacia el prior (0.45/0.27/0.28)
        # Más partidos H2H → más peso al H2H real
        alpha = max(3, 10 - total_h2h)   # prior strength decreases with more data
        v["h2h_home_win_rate"] = (hw + alpha * 0.45) / (total_h2h + alpha)
        v["h2h_draw_rate"]     = (dw + alpha * 0.27) / (total_h2h + alpha)
        v["h2h_away_win_rate"] = (aw + alpha * 0.28) / (total_h2h + alpha)

        goals = [
            (m.home_score or 0) + (m.away_score or 0) for m in h2h
            if m.home_score is not None
        ]
        v["h2h_avg_goals"]  = np.mean(goals) if goals else 2.5
        btts = sum(1 for m in h2h if (m.home_score or 0) > 0 and (m.away_score or 0) > 0)
        v["h2h_btts_rate"]  = btts / total_h2h

    # ── Árbitro ───────────────────────────────────────────────────────────────
    if match.referee_id:
        ref = db.query(Referee).get(match.referee_id)
        if ref and ref.matches_total > 5:
            v["referee_yellow_avg"]   = ref.yellow_cards_avg
            v["referee_red_avg"]      = ref.red_cards_avg
            v["referee_home_win_rate"] = ref.home_win_pct
        else:
            v["referee_yellow_avg"]   = 3.5
            v["referee_home_win_rate"] = 0.45

    # ── Rachas ────────────────────────────────────────────────────────────────
    v["home_unbeaten_streak"] = _unbeaten_streak(home_id, home_history)
    v["away_unbeaten_streak"] = _unbeaten_streak(away_id, away_history)
    v["home_win_streak"]      = _win_streak(home_id, home_history)
    v["away_win_streak"]      = _win_streak(away_id, away_history)

    # ── Rates últimos 5 ──────────────────────────────────────────────────────
    v["home_clean_sheets_5"] = _clean_sheets_rate(home_id, home_history[:5])
    v["away_clean_sheets_5"] = _clean_sheets_rate(away_id, away_history[:5])
    v["home_over25_rate_5"]  = _over25_rate(home_history[:5])
    v["away_over25_rate_5"]  = _over25_rate(away_history[:5])
    v["home_btts_rate_5"]    = _btts_rate(home_history[:5])
    v["away_btts_rate_5"]    = _btts_rate(away_history[:5])

    # ── Liga (one-hot) ────────────────────────────────────────────────────────
    league = db.query(League).get(league_id)
    if league:
        sid = league.sofascore_id
        v["league_premier"]  = 1.0 if sid == 17    else 0.0
        v["league_laliga"]   = 1.0 if sid == 8     else 0.0
        v["league_champions"]= 1.0 if sid == 7     else 0.0
        v["league_dimayor"]  = 1.0 if sid == 11536 else 0.0

    # ── Corners/tarjetas avanzadas ────────────────────────────────────────────
    home_stats_h = _get_match_stats(db, home_h[:10], home_id)
    away_stats_a = _get_match_stats(db, away_a[:10], away_id)

    v["home_corners_home_avg"] = home_stats_h.get("corners", 0)
    v["away_corners_away_avg"] = away_stats_a.get("corners", 0)
    v["home_yellow_home_avg"]  = home_stats_h.get("yellow_cards", 0)
    v["away_yellow_away_avg"]  = away_stats_a.get("yellow_cards", 0)
    v["home_red_home_avg"]     = home_stats_h.get("red_cards", 0)
    v["away_red_away_avg"]     = away_stats_a.get("red_cards", 0)

    # H2H corners y tarjetas
    if h2h:
        h2h_corners, h2h_cards = [], []
        for hm in h2h:
            s = db.query(MatchStats).filter_by(match_id=hm.id).first()
            if s:
                tc = (s.home_corners or 0) + (s.away_corners or 0)
                tk = (s.home_yellow_cards or 0) + (s.away_yellow_cards or 0) + \
                     (s.home_red_cards or 0) + (s.away_red_cards or 0)
                if tc > 0:
                    h2h_corners.append(tc)
                if tk > 0:
                    h2h_cards.append(tk)
        v["h2h_corners_avg"] = float(np.mean(h2h_corners)) if h2h_corners else 9.5
        v["h2h_cards_avg"]   = float(np.mean(h2h_cards))   if h2h_cards   else 3.5

    # Árbitro — corners y tarjetas por equipo
    if match.referee_id:
        ref = db.query(Referee).get(match.referee_id)
        if ref and ref.matches_total and ref.matches_total > 3:
            # referee_yellow_avg ya es por partido total (ambos equipos)
            v["referee_yellow_per_team"] = (ref.yellow_cards_avg or 3.5) / 2
            v["referee_red_per_team"]    = (ref.red_cards_avg    or 0.3) / 2
            v["referee_corners_avg"]     = 9.5  # si Referee no tiene dato de corners, usamos promedio
        else:
            v["referee_yellow_per_team"] = 1.75
            v["referee_red_per_team"]    = 0.15
            v["referee_corners_avg"]     = 9.5

    # Importancia del partido (UCL = alta, derby = alta)
    v["match_importance"] = 1.0 if (league and league.sofascore_id == 7) else 0.5
    # Derby: misma ciudad implica más tarjetas — heurística simple
    v["derby_factor"] = 0.0  # se puede enriquecer con datos geográficos futuros

    return np.array([v[k] for k in FEATURE_NAMES], dtype=np.float32)


# ── Targets para entrenamiento ────────────────────────────────────────────────
def build_targets(match: Match, db: Session = None) -> Optional[dict]:
    """Extrae los labels reales de un partido terminado."""
    if match.status != MatchStatus.FINISHED:
        return None
    if match.home_score is None or match.away_score is None:
        return None

    hs, as_ = match.home_score, match.away_score
    total   = hs + as_

    result_1x2 = 0 if hs > as_ else (1 if hs == as_ else 2)  # 0=H, 1=D, 2=A

    targets = {
        "result_1x2": result_1x2,
        "over_25":    int(total > 2.5),
        "over_35":    int(total > 3.5),
        "btts":       int(hs > 0 and as_ > 0),
        "home_score": hs,
        "away_score": as_,
    }

    # Corners y tarjetas — requieren stats de la DB
    if db is not None and match.stats:
        s = match.stats
        total_corners = (s.home_corners or 0) + (s.away_corners or 0)
        total_yellows  = (s.home_yellow_cards or 0) + (s.away_yellow_cards or 0)
        total_reds     = (s.home_red_cards or 0) + (s.away_red_cards or 0)
        total_cards    = total_yellows + total_reds
        if total_corners > 0:
            targets["corners_over_95"] = int(total_corners > 9.5)
        if total_cards > 0:
            targets["cards_over_35"]   = int(total_cards > 3.5)

    return targets


# ── Helpers internos ──────────────────────────────────────────────────────────
def _get_team_history(db: Session, team_id: int, before: datetime, limit: int) -> list[Match]:
    return (
        db.query(Match)
        .filter(
            Match.status == MatchStatus.FINISHED,
            Match.match_date < before,
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
            Match.home_score.isnot(None),
        )
        .order_by(Match.match_date.desc())
        .limit(limit)
        .all()
    )


def _avg_goals_scored(matches: list[Match]) -> float:
    if not matches:
        return 1.3
    team_id = matches[0].home_team_id  # placeholder — se corrige abajo
    # Calcula promedio sin saber el team_id directamente
    scores = []
    for m in matches:
        if m.home_score is not None:
            scores.append(m.home_score if hasattr(m, '_team_id') else 0)
    return float(np.mean(scores)) if scores else 1.3


def _avg_goals_scored_correct(team_id: int, matches: list[Match]) -> float:
    scores = []
    for m in matches:
        if m.home_score is None:
            continue
        if m.home_team_id == team_id:
            scores.append(m.home_score)
        else:
            scores.append(m.away_score)
    return float(np.mean(scores)) if scores else 1.3


def _avg_goals_conceded_correct(team_id: int, matches: list[Match]) -> float:
    scores = []
    for m in matches:
        if m.home_score is None:
            continue
        if m.home_team_id == team_id:
            scores.append(m.away_score)
        else:
            scores.append(m.home_score)
    return float(np.mean(scores)) if scores else 1.1


# Parche — reemplaza las funciones básicas
def _avg_goals_scored(matches: list[Match]) -> float:
    if not matches:
        return 1.3
    team_id = None
    # Inferir team_id del contexto — tomamos el equipo que aparece más veces
    from collections import Counter
    c = Counter()
    for m in matches:
        c[m.home_team_id] += 1
        c[m.away_team_id] += 1
    team_id = c.most_common(1)[0][0]
    return _avg_goals_scored_correct(team_id, matches)


def _avg_goals_conceded(matches: list[Match]) -> float:
    if not matches:
        return 1.1
    from collections import Counter
    c = Counter()
    for m in matches:
        c[m.home_team_id] += 1
        c[m.away_team_id] += 1
    team_id = c.most_common(1)[0][0]
    return _avg_goals_conceded_correct(team_id, matches)


def _avg_goals_scored_side(matches: list[Match], as_home: bool) -> float:
    scores = []
    for m in matches:
        if m.home_score is None:
            continue
        scores.append(m.home_score if as_home else m.away_score)
    return float(np.mean(scores)) if scores else 1.3


def _avg_goals_conceded_side(matches: list[Match], as_home: bool) -> float:
    scores = []
    for m in matches:
        if m.home_score is None:
            continue
        scores.append(m.away_score if as_home else m.home_score)
    return float(np.mean(scores)) if scores else 1.1


def _avg_xg(db: Session, team_id: int, matches: list[Match], as_home: bool):
    xg_for, xg_ag = [], []
    for m in matches:
        stats = db.query(MatchStats).filter_by(match_id=m.id).first()
        if not stats:
            continue
        is_home = m.home_team_id == team_id
        xg_f = stats.home_xg if is_home else stats.away_xg
        xg_a = stats.away_xg if is_home else stats.home_xg
        if xg_f is not None:
            xg_for.append(xg_f)
        if xg_a is not None:
            xg_ag.append(xg_a)
    return (
        float(np.mean(xg_for)) if xg_for else 1.2,
        float(np.mean(xg_ag)) if xg_ag else 1.0,
    )


def _form_points(team_id: int, matches: list[Match]) -> float:
    pts = 0
    for m in matches:
        if m.home_score is None:
            continue
        w = _winner(m)
        if w == "home" and m.home_team_id == team_id:
            pts += 3
        elif w == "away" and m.away_team_id == team_id:
            pts += 3
        elif w == "draw":
            pts += 1
    return float(pts)


def _winner(m: Match) -> str:
    if m.home_score is None or m.away_score is None:
        return "unknown"
    if m.home_score > m.away_score:
        return "home"
    if m.home_score < m.away_score:
        return "away"
    return "draw"


def _get_match_stats(db: Session, matches: list[Match], team_id: int) -> dict:
    shots, sot, poss, corners, yellows, reds = [], [], [], [], [], []
    for m in matches:
        s = db.query(MatchStats).filter_by(match_id=m.id).first()
        if not s:
            continue
        is_home = m.home_team_id == team_id
        shots.append(s.home_shots    if is_home else s.away_shots)
        sot.append(s.home_shots_on_target if is_home else s.away_shots_on_target)
        poss.append(s.home_possession if is_home else s.away_possession)
        corners.append(s.home_corners if is_home else s.away_corners)
        yellows.append(s.home_yellow_cards if is_home else s.away_yellow_cards)
        reds.append(s.home_red_cards if is_home else s.away_red_cards)

    def safe_mean(lst):
        lst = [x for x in lst if x is not None]
        return float(np.mean(lst)) if lst else 0.0

    return {
        "shots":           safe_mean(shots),
        "shots_on_target": safe_mean(sot),
        "possession":      safe_mean(poss) or 50.0,
        "corners":         safe_mean(corners),
        "yellow_cards":    safe_mean(yellows),
        "red_cards":       safe_mean(reds),
    }


def _get_standing(db: Session, team_id: int, season_id: int, league_id: int):
    st = db.query(Standing).filter_by(team_id=team_id, season_id=season_id).first()
    total = db.query(Standing).filter_by(season_id=season_id).count() if season_id else 20
    if st:
        return st.position or 10, st.points or 0, total or 20
    return 10, 0, total or 20


def _get_h2h(db: Session, home_id: int, away_id: int, before: datetime, limit: int):
    return (
        db.query(Match)
        .filter(
            Match.status == MatchStatus.FINISHED,
            Match.match_date < before,
            Match.home_score.isnot(None),
            or_(
                and_(Match.home_team_id == home_id, Match.away_team_id == away_id),
                and_(Match.home_team_id == away_id, Match.away_team_id == home_id),
            ),
        )
        .order_by(Match.match_date.desc())
        .limit(limit)
        .all()
    )


def _unbeaten_streak(team_id: int, matches: list[Match]) -> float:
    streak = 0
    for m in matches:
        w = _winner(m)
        if (w == "home" and m.away_team_id == team_id) or \
           (w == "away" and m.home_team_id == team_id):
            break
        streak += 1
    return float(streak)


def _win_streak(team_id: int, matches: list[Match]) -> float:
    streak = 0
    for m in matches:
        w = _winner(m)
        if (w == "home" and m.home_team_id == team_id) or \
           (w == "away" and m.away_team_id == team_id):
            streak += 1
        else:
            break
    return float(streak)


def _clean_sheets_rate(team_id: int, matches: list[Match]) -> float:
    if not matches:
        return 0.0
    cs = 0
    for m in matches:
        if m.home_score is None:
            continue
        if m.home_team_id == team_id and m.away_score == 0:
            cs += 1
        elif m.away_team_id == team_id and m.home_score == 0:
            cs += 1
    return cs / len(matches)


def _over25_rate(matches: list[Match]) -> float:
    finished = [m for m in matches if m.home_score is not None]
    if not finished:
        return 0.5
    return sum(1 for m in finished if (m.home_score + m.away_score) > 2.5) / len(finished)


def _btts_rate(matches: list[Match]) -> float:
    finished = [m for m in matches if m.home_score is not None]
    if not finished:
        return 0.5
    return sum(1 for m in finished if m.home_score > 0 and m.away_score > 0) / len(finished)

"""
Evaluador de predicciones post-partido.

evaluate_finished_matches(db) → evalúa todos los partidos terminados con
    predicción aún no evaluada. Calcula métricas de acierto y Brier score.

get_model_stats(db) → retorna resumen de rendimiento del modelo:
    accuracy por mercado, calibración, tendencia reciente, por liga.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from ..models import Match, Prediction, League, MatchStatus

log = logging.getLogger("ml2.evaluator")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _brier_1x2(pred: Prediction, outcome: str) -> float:
    """
    Brier score multiclase para 1X2.
    Rango [0, 2]. 0 = predicción perfecta.
    """
    ph = pred.prob_home_win or 0
    pd = pred.prob_draw     or 0
    pa = pred.prob_away_win or 0
    ah = 1.0 if outcome == "H" else 0.0
    ad = 1.0 if outcome == "D" else 0.0
    aa = 1.0 if outcome == "A" else 0.0
    return round((ph - ah) ** 2 + (pd - ad) ** 2 + (pa - aa) ** 2, 4)


def _predicted_outcome(pred: Prediction) -> str:
    """El resultado más probable según la predicción."""
    probs = {
        "H": pred.prob_home_win or 0,
        "D": pred.prob_draw     or 0,
        "A": pred.prob_away_win or 0,
    }
    return max(probs, key=probs.get)


# ── Evaluación individual ─────────────────────────────────────────────────────

def _evaluate_one(db: Session, match: Match, pred: Prediction) -> bool:
    """
    Evalúa una predicción contra el resultado real.
    Retorna True si se pudo evaluar.
    """
    hs = match.home_score
    as_ = match.away_score
    if hs is None or as_ is None:
        return False

    outcome = _actual_outcome(hs, as_)
    total_goals   = hs + as_
    total_corners = None
    total_home_cards = None
    total_away_cards = None

    if match.stats:
        s = match.stats
        hc = s.home_corners
        ac = s.away_corners
        if hc is not None and ac is not None:
            total_corners = hc + ac
        total_home_cards = s.home_yellow_cards
        total_away_cards = s.away_yellow_cards

    # Resultados reales
    pred.actual_outcome    = outcome
    pred.actual_goals      = total_goals
    pred.actual_corners    = total_corners
    pred.actual_home_cards = total_home_cards
    pred.actual_away_cards = total_away_cards

    # ¿Acertó el resultado 1X2?
    pred.outcome_correct = (_predicted_outcome(pred) == outcome)

    # ¿Acertó Over/Under 2.5?
    if pred.prob_over_25 is not None and pred.prob_under_25 is not None:
        predicted_over25 = (pred.prob_over_25 or 0) >= (pred.prob_under_25 or 0)
        actual_over25    = total_goals > 2
        pred.over25_correct = (predicted_over25 == actual_over25)

    # ¿Acertó BTTS?
    if pred.prob_btts is not None:
        predicted_btts = (pred.prob_btts or 0) >= 0.5
        actual_btts    = hs > 0 and as_ > 0
        pred.btts_correct = (predicted_btts == actual_btts)

    # ¿Acertó Corners Over 9.5?
    if total_corners is not None and pred.prob_over_95_corners is not None:
        predicted_over_corners = (pred.prob_over_95_corners or 0) >= 0.5
        actual_over_corners    = total_corners > 9
        pred.corners_correct = (predicted_over_corners == actual_over_corners)

    # ¿Acertó Tarjetas Over 3.5?
    if total_home_cards is not None and total_away_cards is not None:
        total_cards = total_home_cards + total_away_cards
        if pred.prob_over_35_cards is not None:
            predicted_over_cards = (pred.prob_over_35_cards or 0) >= 0.5
            actual_over_cards    = total_cards > 3
            pred.cards_correct = (predicted_over_cards == actual_over_cards)

    # Brier score
    pred.brier_1x2   = _brier_1x2(pred, outcome)

    # ¿La apuesta recomendada (smart_bet) ganó?
    pred.smart_bet_correct = _evaluate_smart_bet(pred, outcome, total_goals,
                                                  total_corners, total_home_cards,
                                                  total_away_cards, hs, as_)
    pred.evaluated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    return True


def _evaluate_smart_bet(pred, outcome, total_goals, total_corners,
                        total_home_cards, total_away_cards, hs, as_) -> bool | None:
    """Evalúa si todos los picks de la smart_bet fueron correctos."""
    import json
    if not pred.smart_bet:
        return None
    try:
        sb = json.loads(pred.smart_bet)
    except Exception:
        return None

    picks = sb.get("picks", [])
    if not picks:
        return None

    for pick in picks:
        market = pick.get("market", "")
        label  = pick.get("label", "")
        ok = _pick_correct(market, label, outcome, total_goals,
                           total_corners, total_home_cards, total_away_cards, hs, as_)
        if ok is None or not ok:
            return ok  # None = sin datos, False = falló
    return True


def _pick_correct(market, label, outcome, total_goals, total_corners,
                  total_home_cards, total_away_cards, hs, as_) -> bool | None:
    label_l = label.lower()
    if market == "1X2":
        if "local" in label_l or "gana " == label_l[:6] and "visitante" not in label_l:
            # "Gana {home}" — si no dice visitante asumimos local
            if "visitante" in label_l:
                return outcome == "A"
            if "empate" in label_l:
                return outcome == "D"
            return outcome == "H"
        if "empate" in label_l:
            return outcome == "D"
        return outcome == "A"
    if market == "Goles":
        if total_goals is None:
            return None
        if "más" in label_l and "2.5" in label_l:
            return total_goals > 2
        if "menos" in label_l and "2.5" in label_l:
            return total_goals <= 2
        if "más" in label_l and "3.5" in label_l:
            return total_goals > 3
        if "menos" in label_l and "3.5" in label_l:
            return total_goals <= 3
    if market == "BTTS":
        if hs is None or as_ is None:
            return None
        btts = hs > 0 and as_ > 0
        return btts if "no " not in label_l else not btts
    if market == "Corners":
        if total_corners is None:
            return None
        if "más" in label_l:
            return total_corners > 9
        return total_corners <= 9
    if market == "Tarjetas":
        if total_home_cards is None or total_away_cards is None:
            return None
        total = total_home_cards + total_away_cards
        if "más" in label_l:
            return total > 3
        return total <= 3
    return None


# ── Evaluación batch ──────────────────────────────────────────────────────────

def evaluate_finished_matches(db: Session) -> dict:
    """
    Evalúa todos los partidos FINISHED que tienen predicción pero aún
    no han sido evaluados. Retorna resumen de la ejecución.
    """
    pending = (
        db.query(Prediction)
        .join(Match, Prediction.match_id == Match.id)
        .filter(
            Match.status == MatchStatus.FINISHED,
            Prediction.evaluated_at == None,  # noqa: E711
        )
        .all()
    )

    evaluated = 0
    skipped   = 0

    for pred in pending:
        match = pred.match
        ok = _evaluate_one(db, match, pred)
        if ok:
            evaluated += 1
        else:
            skipped += 1

    if evaluated:
        db.commit()
        log.info(f"[Evaluator] {evaluated} predicciones evaluadas, {skipped} sin datos.")

    return {"evaluated": evaluated, "skipped": skipped}


# ── Estadísticas del modelo ───────────────────────────────────────────────────

def get_model_stats(db: Session) -> dict:
    """
    Calcula métricas globales y por liga del modelo.
    Solo considera predicciones ya evaluadas.
    """
    preds = (
        db.query(Prediction)
        .join(Match, Prediction.match_id == Match.id)
        .filter(Prediction.evaluated_at != None)  # noqa: E711
        .all()
    )

    if not preds:
        return {"total_evaluated": 0, "message": "Sin predicciones evaluadas aún."}

    total = len(preds)

    def _pct(lst):
        valid = [x for x in lst if x is not None]
        if not valid:
            return None
        return round(sum(valid) / len(valid) * 100, 1)

    def _avg(lst):
        valid = [x for x in lst if x is not None]
        if not valid:
            return None
        return round(sum(valid) / len(valid), 4)

    # Métricas globales
    global_stats = {
        "total_evaluated":    total,
        "outcome_accuracy":   _pct([p.outcome_correct    for p in preds]),
        "over25_accuracy":    _pct([p.over25_correct     for p in preds]),
        "btts_accuracy":      _pct([p.btts_correct       for p in preds]),
        "corners_accuracy":   _pct([p.corners_correct    for p in preds]),
        "cards_accuracy":     _pct([p.cards_correct      for p in preds]),
        "smart_bet_accuracy": _pct([p.smart_bet_correct  for p in preds]),
        "avg_brier_1x2":      _avg([p.brier_1x2          for p in preds]),
    }

    # Últimos 20 partidos (tendencia reciente)
    recent = sorted(preds, key=lambda p: p.evaluated_at or datetime.min)[-20:]
    global_stats["recent_20"] = {
        "outcome_accuracy": _pct([p.outcome_correct for p in recent]),
        "avg_brier_1x2":    _avg([p.brier_1x2       for p in recent]),
    }

    # Por liga
    by_league: dict[str, list] = {}
    for pred in preds:
        league_name = pred.match.league.name if pred.match and pred.match.league else "Desconocida"
        by_league.setdefault(league_name, []).append(pred)

    league_stats = []
    for league_name, lpreds in by_league.items():
        league_stats.append({
            "league":            league_name,
            "total":             len(lpreds),
            "outcome_accuracy":  _pct([p.outcome_correct   for p in lpreds]),
            "over25_accuracy":   _pct([p.over25_correct    for p in lpreds]),
            "btts_accuracy":     _pct([p.btts_correct      for p in lpreds]),
            "smart_bet_accuracy":_pct([p.smart_bet_correct for p in lpreds]),
            "avg_brier_1x2":     _avg([p.brier_1x2         for p in lpreds]),
        })
    league_stats.sort(key=lambda x: x["total"], reverse=True)

    # Calibración por rango de probabilidad (¿cuando decimos X% cuánto acierta?)
    calibration = _calibration_buckets(preds)

    return {
        **global_stats,
        "by_league":   league_stats,
        "calibration": calibration,
    }


def _calibration_buckets(preds: list[Prediction]) -> list[dict]:
    """
    Agrupa predicciones por rango de probabilidad del resultado predicho
    y calcula qué % realmente ocurrió.

    Ej: cuando decimos 70-80%, ¿gana el favorito el 75% de las veces?
    """
    buckets = {
        "50-60": {"predicted": [], "correct": []},
        "60-70": {"predicted": [], "correct": []},
        "70-80": {"predicted": [], "correct": []},
        "80-88": {"predicted": [], "correct": []},
    }

    for pred in preds:
        if pred.outcome_correct is None:
            continue
        best_prob = max(
            pred.prob_home_win or 0,
            pred.prob_draw     or 0,
            pred.prob_away_win or 0,
        ) * 100

        if 50 <= best_prob < 60:
            key = "50-60"
        elif 60 <= best_prob < 70:
            key = "60-70"
        elif 70 <= best_prob < 80:
            key = "70-80"
        elif best_prob >= 80:
            key = "80-88"
        else:
            continue

        buckets[key]["predicted"].append(best_prob)
        buckets[key]["correct"].append(int(pred.outcome_correct))

    result = []
    for label, data in buckets.items():
        n = len(data["correct"])
        if n == 0:
            continue
        result.append({
            "range":        label + "%",
            "total":        n,
            "avg_predicted": round(sum(data["predicted"]) / n, 1),
            "actual_pct":   round(sum(data["correct"]) / n * 100, 1),
        })
    return result

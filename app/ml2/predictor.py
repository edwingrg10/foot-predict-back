"""
Predictor — usa los modelos entrenados para predecir un partido.

Combina:
  - Probabilidades del modelo XGBoost
  - Modelo de Poisson (Dixon-Coles) para scoreline
  - Análisis de value bets
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sqlalchemy.orm import Session

from ..models import Match, Prediction, MatchStatus
from .features import build_features
from .trainer import load_models, models_exist

log = logging.getLogger("ml2.predictor")

# Modelos cargados en memoria (singleton)
_models: dict = {}


def _get_models() -> dict:
    global _models
    if not _models:
        _models = load_models()
    return _models


@dataclass
class MatchPrediction:
    match_id: int

    # 1X2
    prob_home_win: float = 0.0
    prob_draw:     float = 0.0
    prob_away_win: float = 0.0

    # Goles
    prob_over_25:  float = 0.0
    prob_under_25: float = 0.0
    prob_over_35:  float = 0.0
    prob_under_35: float = 0.0

    # BTTS
    prob_btts:    float = 0.0
    prob_no_btts: float = 0.0

    # Corners
    prob_over_95_corners:  float = 0.0
    prob_under_95_corners: float = 0.0
    expected_home_corners: float = 0.0
    expected_away_corners: float = 0.0

    # Tarjetas
    prob_over_35_cards:  float = 0.0
    prob_under_35_cards: float = 0.0
    expected_home_cards: float = 0.0
    expected_away_cards: float = 0.0

    # Score esperado (Poisson)
    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0
    predicted_score:     str   = "1-1"

    # Distribución de resultados (top 10 scores)
    score_distribution: list[dict] = field(default_factory=list)

    # Calidad de predicción
    confidence_score: float = 0.0
    risk_level:       str   = "medium"
    value_bets:       list[dict] = field(default_factory=list)
    analysis_notes:   list[str]  = field(default_factory=list)

    # Modelo usado
    model_version: str = "2.0-xgb"


def predict_match(db: Session, match: Match) -> Optional[MatchPrediction]:
    """
    Genera una predicción completa para un partido.
    Retorna None si no hay suficientes datos.
    """
    if match.status == MatchStatus.FINISHED:
        log.debug(f"[Predictor] Partido {match.sofascore_id} ya terminado — skipping")
        return None

    # Construir features
    feats = build_features(db, match)
    if feats is None:
        log.debug(f"[Predictor] Sin features suficientes para match {match.sofascore_id}")
        return _fallback_prediction(db, match)

    if np.isnan(feats).any():
        feats = np.nan_to_num(feats, nan=0.0)

    models = _get_models()
    pred   = MatchPrediction(match_id=match.id)

    # ── 1X2 ──────────────────────────────────────────────────────────────────
    if "result_1x2" in models:
        proba = models["result_1x2"].predict_proba([feats])[0]
        # Clases: 0=home, 1=draw, 2=away
        # Cap: ningún resultado puede superar 88% (fútbol siempre es incierto)
        p_h = min(float(proba[0]), 0.88)
        p_d = min(float(proba[1]), 0.88)
        p_a = min(float(proba[2]), 0.88)
        # Re-normalizar para que sumen 1
        total = p_h + p_d + p_a
        pred.prob_home_win = round(p_h / total, 4)
        pred.prob_draw     = round(p_d / total, 4)
        pred.prob_away_win = round(p_a / total, 4)
    else:
        pred.prob_home_win, pred.prob_draw, pred.prob_away_win = 0.45, 0.27, 0.28

    # ── Over/Under ───────────────────────────────────────────────────────────
    if "over_25" in models:
        p_o25 = float(models["over_25"].predict_proba([feats])[0][1])
        pred.prob_over_25  = round(p_o25, 4)
        pred.prob_under_25 = round(1 - p_o25, 4)
    if "over_35" in models:
        p_o35 = float(models["over_35"].predict_proba([feats])[0][1])
        pred.prob_over_35  = round(p_o35, 4)
        pred.prob_under_35 = round(1 - p_o35, 4)

    # ── BTTS ─────────────────────────────────────────────────────────────────
    if "btts" in models:
        p_btts = float(models["btts"].predict_proba([feats])[0][1])
        pred.prob_btts    = round(p_btts, 4)
        pred.prob_no_btts = round(1 - p_btts, 4)

    # ── Corners ──────────────────────────────────────────────────────────────
    if "corners_over_95" in models:
        p_c = float(models["corners_over_95"].predict_proba([feats])[0][1])
        pred.prob_over_95_corners  = round(p_c, 4)
        pred.prob_under_95_corners = round(1 - p_c, 4)
    else:
        # Estimación estadística sin modelo
        hc = feats[FEAT("home_corners_avg")]
        ac = feats[FEAT("away_corners_avg")]
        total_c = max(5.0, hc + ac)
        pred.prob_over_95_corners  = round(min(0.85, max(0.15, (total_c - 7) / 8)), 4)
        pred.prob_under_95_corners = round(1 - pred.prob_over_95_corners, 4)

    # Corners esperados por equipo
    hc_avg = feats[FEAT("home_corners_avg")]
    ac_avg = feats[FEAT("away_corners_avg")]
    pred.expected_home_corners = round(max(3.0, hc_avg if hc_avg > 0 else 4.5), 2)
    pred.expected_away_corners = round(max(2.5, ac_avg if ac_avg > 0 else 4.0), 2)

    # ── Tarjetas ─────────────────────────────────────────────────────────────
    if "cards_over_35" in models:
        p_k = float(models["cards_over_35"].predict_proba([feats])[0][1])
        pred.prob_over_35_cards  = round(p_k, 4)
        pred.prob_under_35_cards = round(1 - p_k, 4)
    else:
        # Estimación: árbitro + promedio equipos
        ref_y = feats[FEAT("referee_yellow_avg")]
        hy    = feats[FEAT("home_yellow_avg")]
        ay    = feats[FEAT("away_yellow_avg")]
        # Promedio ponderado
        est_cards = ref_y * 0.5 + (hy + ay) * 0.5
        est_cards = max(1.5, est_cards)
        pred.prob_over_35_cards  = round(min(0.85, max(0.15, (est_cards - 2.5) / 4)), 4)
        pred.prob_under_35_cards = round(1 - pred.prob_over_35_cards, 4)

    # Tarjetas esperadas por equipo (árbitro pesa mucho)
    ref_y_per_team = feats[FEAT("referee_yellow_per_team")]
    hy_avg         = feats[FEAT("home_yellow_avg")]
    ay_avg         = feats[FEAT("away_yellow_avg")]
    pred.expected_home_cards = round(max(0.5, hy_avg * 0.6 + ref_y_per_team * 0.4), 2)
    pred.expected_away_cards = round(max(0.5, ay_avg * 0.6 + ref_y_per_team * 0.4), 2)

    # ── Goles esperados (Poisson) ─────────────────────────────────────────────
    home_atk  = feats[FEAT("home_goals_scored_avg")]
    home_def  = feats[FEAT("home_goals_conceded_avg")]
    away_atk  = feats[FEAT("away_goals_scored_avg")]
    away_def  = feats[FEAT("away_goals_conceded_avg")]

    home_xg = max(0.3, home_atk * 0.5 + away_def * 0.3 + feats[FEAT("home_xg_for_avg")] * 0.2)
    away_xg = max(0.2, away_atk * 0.5 + home_def * 0.3 + feats[FEAT("away_xg_for_avg")] * 0.2)

    # Calibrar con la probabilidad 1X2 del modelo
    home_xg, away_xg = _calibrate_goals(home_xg, away_xg, pred.prob_home_win, pred.prob_draw, pred.prob_away_win)

    pred.expected_home_goals = round(home_xg, 2)
    pred.expected_away_goals = round(away_xg, 2)

    # ── Distribución de scores (Poisson bivariado) ───────────────────────────
    pred.score_distribution = _poisson_score_dist(home_xg, away_xg)
    if pred.score_distribution:
        top = pred.score_distribution[0]
        pred.predicted_score = f"{top['home']}-{top['away']}"

    # ── Confidence ───────────────────────────────────────────────────────────
    max_prob = max(pred.prob_home_win, pred.prob_draw, pred.prob_away_win)
    pred.confidence_score = round(_compute_confidence(feats, max_prob), 2)
    pred.risk_level = (
        "low"    if pred.confidence_score >= 0.70 else
        "medium" if pred.confidence_score >= 0.55 else
        "high"
    )

    # ── Análisis ─────────────────────────────────────────────────────────────
    pred.analysis_notes = _build_notes(feats, pred)

    # ── Value bets ───────────────────────────────────────────────────────────
    pred.value_bets = _find_value_bets(pred)

    return pred


def predict_and_save(db: Session, match: Match) -> Optional[Prediction]:
    """Predice y guarda en la tabla predictions."""
    pred_data = predict_match(db, match)
    if not pred_data:
        return None

    import json
    prediction = db.query(Prediction).filter_by(match_id=match.id).first()
    if not prediction:
        prediction = Prediction(match_id=match.id)
        db.add(prediction)

    f = float  # pyodbc necesita float nativo, no numpy.float32
    prediction.prob_home_win          = f(pred_data.prob_home_win)
    prediction.prob_draw              = f(pred_data.prob_draw)
    prediction.prob_away_win          = f(pred_data.prob_away_win)
    prediction.prob_over_25           = f(pred_data.prob_over_25)
    prediction.prob_under_25          = f(pred_data.prob_under_25)
    prediction.prob_over_35           = f(pred_data.prob_over_35)
    prediction.prob_under_35          = f(pred_data.prob_under_35)
    prediction.prob_btts              = f(pred_data.prob_btts)
    prediction.prob_no_btts           = f(pred_data.prob_no_btts)
    prediction.prob_over_95_corners   = f(pred_data.prob_over_95_corners)
    prediction.prob_under_95_corners  = f(pred_data.prob_under_95_corners)
    prediction.expected_home_corners  = f(pred_data.expected_home_corners)
    prediction.expected_away_corners  = f(pred_data.expected_away_corners)
    prediction.prob_over_35_cards     = f(pred_data.prob_over_35_cards)
    prediction.prob_under_35_cards    = f(pred_data.prob_under_35_cards)
    prediction.expected_home_cards    = f(pred_data.expected_home_cards)
    prediction.expected_away_cards    = f(pred_data.expected_away_cards)
    prediction.expected_home_goals    = f(pred_data.expected_home_goals)
    prediction.expected_away_goals    = f(pred_data.expected_away_goals)
    prediction.predicted_score        = pred_data.predicted_score
    prediction.confidence_score       = f(pred_data.confidence_score)
    prediction.risk_level             = pred_data.risk_level
    prediction.value_bets             = json.dumps(pred_data.value_bets)
    prediction.model_version          = pred_data.model_version

    db.commit()
    return prediction


def predict_all_upcoming(db: Session) -> int:
    """Genera predicciones para todos los partidos próximos sin predicción."""
    from ..models import MatchStatus
    matches = (
        db.query(Match)
        .filter(Match.status == MatchStatus.SCHEDULED)
        .all()
    )
    done = 0
    for match in matches:
        try:
            p = predict_and_save(db, match)
            if p:
                done += 1
        except Exception as e:
            log.error(f"[Predictor] Error en match {match.id}: {e}")
    log.info(f"[Predictor] Predicciones generadas: {done}/{len(matches)}")
    return done


# ── Helpers ───────────────────────────────────────────────────────────────────
from .features import FEATURE_NAMES as _FN

def FEAT(name: str) -> int:
    return _FN.index(name)


def _calibrate_goals(home_xg, away_xg, p_home, p_draw, p_away):
    """Ajusta xG para que sean consistentes con las probabilidades 1X2."""
    from scipy.stats import poisson
    # Iteraciones simples de calibración
    for _ in range(20):
        p_h = sum(
            poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            for h in range(8) for a in range(8) if h > a
        )
        p_d = sum(
            poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            for h in range(8) for a in range(8) if h == a
        )
        p_a = sum(
            poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            for h in range(8) for a in range(8) if h < a
        )
        if p_h > 0 and p_a > 0:
            home_xg *= (p_home / p_h) ** 0.1
            away_xg *= (p_away / p_a) ** 0.1
    return max(0.2, home_xg), max(0.1, away_xg)


def _poisson_score_dist(home_xg: float, away_xg: float) -> list[dict]:
    from scipy.stats import poisson
    scores = []
    for h in range(7):
        for a in range(7):
            prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            scores.append({"home": h, "away": a, "prob": round(prob, 4)})
    scores.sort(key=lambda x: -x["prob"])
    return scores[:10]


def _compute_confidence(feats: np.ndarray, max_prob: float) -> float:
    """Confianza basada en: cuán determinante es el favorito + cantidad de datos."""
    form_diff = abs(feats[FEAT("form_delta")])
    pos_diff  = abs(feats[FEAT("position_delta")])
    xg_diff   = abs(feats[FEAT("xg_delta")])
    base = max_prob * 0.6 + min(form_diff / 15, 0.2) + min(pos_diff, 0.2)
    return min(0.95, max(0.35, base))


def _build_notes(feats: np.ndarray, pred: MatchPrediction) -> list[str]:
    notes = []

    # Forma
    hf = feats[FEAT("home_form_5")]
    af = feats[FEAT("away_form_5")]
    if hf > 10:
        notes.append(f"Local en excelente forma ({hf:.0f}/15 pts últimos 5 partidos)")
    if af > 10:
        notes.append(f"Visitante en excelente forma ({af:.0f}/15 pts últimos 5 partidos)")
    if hf < 4:
        notes.append("Local en mala racha reciente")
    if af < 4:
        notes.append("Visitante en mala racha reciente")

    # H2H
    h2h_hw = feats[FEAT("h2h_home_win_rate")]
    if h2h_hw > 0.6:
        notes.append(f"Local domina el H2H ({h2h_hw*100:.0f}% victorias)")
    elif h2h_hw < 0.25:
        notes.append(f"Visitante tiene ventaja histórica en el H2H")

    # Goles
    avg_g = feats[FEAT("home_goals_scored_avg")] + feats[FEAT("away_goals_scored_avg")]
    if avg_g > 3.5:
        notes.append("Ambos equipos con alta producción goleadora")
    if pred.prob_over_25 > 0.65:
        notes.append(f"Alta probabilidad de más de 2.5 goles ({pred.prob_over_25*100:.0f}%)")
    if pred.prob_btts > 0.65:
        notes.append(f"Alta probabilidad de que ambos anoten ({pred.prob_btts*100:.0f}%)")

    # Posición
    pd = feats[FEAT("position_delta")]
    if pd > 0.4:
        notes.append("Gran diferencia de posición en tabla a favor del local")
    elif pd < -0.4:
        notes.append("Gran diferencia de posición en tabla a favor del visitante")

    # Árbitro
    ref_y = feats[FEAT("referee_yellow_avg")]
    if ref_y > 5:
        notes.append(f"Árbitro con historial de muchas tarjetas ({ref_y:.1f} amarillas/partido)")
    elif ref_y < 2.5:
        notes.append(f"Árbitro permisivo ({ref_y:.1f} amarillas/partido en promedio)")

    # Corners
    hc = feats[FEAT("home_corners_avg")]
    ac = feats[FEAT("away_corners_avg")]
    total_c = hc + ac
    if total_c > 11:
        notes.append(f"Ambos equipos generan muchos corners (prom {total_c:.1f}/partido)")
    if pred.prob_over_95_corners > 0.65:
        notes.append(f"Alta probabilidad de más de 9.5 corners ({pred.prob_over_95_corners*100:.0f}%)")

    # Tarjetas
    if pred.prob_over_35_cards > 0.65:
        notes.append(f"Partido propenso a tarjetas: más de 3.5 ({pred.prob_over_35_cards*100:.0f}%)")

    return notes[:8]


def _find_value_bets(pred: MatchPrediction) -> list[dict]:
    """
    Identifica value bets para todos los mercados disponibles:
    1X2, Over/Under goles, BTTS, Corners, Tarjetas.
    Edge threshold: nuestra prob > mercado implícito + 8% y prob > 38%.
    """
    bets = []
    MARGIN = 0.92   # margen bookmaker ~8%
    MIN_EDGE = 0.08
    MIN_PROB = 0.38

    checks = [
        # (mercado, pick_label, our_prob, margin)
        ("1X2",      "Victoria Local",          pred.prob_home_win,         MARGIN),
        ("1X2",      "Empate",                  pred.prob_draw,             MARGIN),
        ("1X2",      "Victoria Visitante",      pred.prob_away_win,         MARGIN),
        ("Goles",    "Más de 2.5 goles",        pred.prob_over_25,          MARGIN),
        ("Goles",    "Menos de 2.5 goles",      pred.prob_under_25,         MARGIN),
        ("Goles",    "Más de 3.5 goles",        pred.prob_over_35,          MARGIN),
        ("BTTS",     "Ambos Anotan",            pred.prob_btts,             MARGIN),
        ("BTTS",     "No Ambos Anotan",         pred.prob_no_btts,          MARGIN),
        ("Corners",  "Más de 9.5 corners",      pred.prob_over_95_corners,  MARGIN),
        ("Corners",  "Menos de 9.5 corners",    pred.prob_under_95_corners, MARGIN),
        ("Tarjetas", "Más de 3.5 tarjetas",     pred.prob_over_35_cards,    MARGIN),
        ("Tarjetas", "Menos de 3.5 tarjetas",   pred.prob_under_35_cards,   MARGIN),
    ]

    for market, label, our_prob, margin in checks:
        if our_prob <= 0:
            continue
        mkt_implied = our_prob * margin
        edge = our_prob - mkt_implied
        if edge >= MIN_EDGE and our_prob >= MIN_PROB:
            bets.append({
                "market":    market,
                "pick":      label,
                "our_prob":  round(our_prob, 3),
                "fair_odds": round(1 / our_prob, 2),
                "edge":      round(edge * 100, 1),
            })

    bets.sort(key=lambda x: -x["edge"])
    return bets[:6]


def _fallback_prediction(db: Session, match: Match) -> Optional[MatchPrediction]:
    """Predicción básica cuando no hay suficientes datos históricos."""
    pred = MatchPrediction(match_id=match.id)
    pred.prob_home_win          = 0.44
    pred.prob_draw              = 0.26
    pred.prob_away_win          = 0.30
    pred.prob_over_25           = 0.52
    pred.prob_under_25          = 0.48
    pred.prob_over_35           = 0.30
    pred.prob_under_35          = 0.70
    pred.prob_btts              = 0.50
    pred.prob_no_btts           = 0.50
    pred.prob_over_95_corners   = 0.52
    pred.prob_under_95_corners  = 0.48
    pred.expected_home_corners  = 4.5
    pred.expected_away_corners  = 4.0
    pred.prob_over_35_cards     = 0.48
    pred.prob_under_35_cards    = 0.52
    pred.expected_home_cards    = 1.8
    pred.expected_away_cards    = 1.7
    pred.expected_home_goals    = 1.4
    pred.expected_away_goals    = 1.1
    pred.predicted_score        = "1-1"
    pred.confidence_score       = 0.40
    pred.risk_level             = "high"
    pred.score_distribution     = _poisson_score_dist(1.4, 1.1)
    pred.analysis_notes         = ["Datos insuficientes — predicción basada en promedios de liga"]
    pred.model_version          = "2.0-fallback"
    return pred

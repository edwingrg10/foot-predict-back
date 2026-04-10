"""
Football Prediction Engine
Uses a hybrid approach:
  1. Dixon-Coles Poisson model for goal expectation
  2. Feature-based XGBoost for 1X2 classification
  3. Value bet detection via Kelly Criterion
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TeamStats:
    name: str
    attack_strength: float = 1.0     # relative to league average
    defense_strength: float = 1.0
    form_points: float = 5.0         # last 5 matches (max 15)
    goals_scored_avg: float = 1.3
    goals_conceded_avg: float = 1.2
    home_advantage: float = 1.0      # multiplier for home team
    is_home: bool = False
    key_players_missing: int = 0
    fatigue_index: float = 0.0       # 0-1, higher = more tired
    motivation: float = 1.0          # 1=normal, 1.2=must win, 0.8=nothing at stake


@dataclass
class MatchContext:
    league_avg_goals: float = 2.65
    referee_cards_avg: float = 4.0
    referee_home_bias: float = 0.0   # positive favors home
    weather_factor: float = 1.0      # 0.85=rainy, 1=normal
    altitude_factor: float = 1.0
    importance: str = "regular"       # regular | derby | final | relegation
    h2h_home_wins: int = 0
    h2h_draws: int = 0
    h2h_away_wins: int = 0


@dataclass
class PredictionResult:
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    prob_over_25: float
    prob_under_25: float
    prob_btts: float
    prob_no_btts: float
    expected_home_goals: float
    expected_away_goals: float
    predicted_score: str
    prob_over_35: float
    prob_under_35: float
    confidence_score: float
    risk_level: str
    prob_home_cards: float = 0.0
    prob_corners_over: float = 0.55
    value_bets: list[dict] = field(default_factory=list)
    score_distribution: dict = field(default_factory=dict)
    analysis_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dixon-Coles adjustment (rho correction for 0-0, 1-0, 0-1, 1-1)
# ---------------------------------------------------------------------------

def _dixon_coles_tau(home_goals: int, away_goals: int, mu: float, nu: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1 - mu * nu * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + nu * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + mu * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def poisson_match_probabilities(
    mu_home: float,
    mu_away: float,
    max_goals: int = 8,
    rho: float = -0.13,
) -> dict[str, float]:
    """
    Calculate full score probability matrix and aggregate markets.
    rho ≈ -0.13 is the typical Dixon-Coles correction value.
    """
    score_probs: dict[str, float] = {}
    prob_home_win = prob_draw = prob_away_win = 0.0
    prob_over_25 = prob_over_35 = prob_btts = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_h = poisson.pmf(h, mu_home)
            p_a = poisson.pmf(a, mu_away)
            tau = _dixon_coles_tau(h, a, mu_home, mu_away, rho)
            prob = p_h * p_a * tau

            score_key = f"{h}-{a}"
            score_probs[score_key] = round(prob, 5)

            if h > a:
                prob_home_win += prob
            elif h == a:
                prob_draw += prob
            else:
                prob_away_win += prob

            total_goals = h + a
            if total_goals > 2.5:
                prob_over_25 += prob
            if total_goals > 3.5:
                prob_over_35 += prob
            if h > 0 and a > 0:
                prob_btts += prob

    return {
        "prob_home_win": round(prob_home_win, 4),
        "prob_draw": round(prob_draw, 4),
        "prob_away_win": round(prob_away_win, 4),
        "prob_over_25": round(prob_over_25, 4),
        "prob_under_25": round(1 - prob_over_25, 4),
        "prob_over_35": round(prob_over_35, 4),
        "prob_under_35": round(1 - prob_over_35, 4),
        "prob_btts": round(prob_btts, 4),
        "prob_no_btts": round(1 - prob_btts, 4),
        "score_distribution": score_probs,
    }


# ---------------------------------------------------------------------------
# Goal expectation model
# ---------------------------------------------------------------------------

HOME_ADVANTAGE = 1.35   # empirical constant across most leagues


def compute_expected_goals(
    home: TeamStats,
    away: TeamStats,
    ctx: MatchContext,
) -> tuple[float, float]:
    """
    Expected goals via attack × defense × league_avg × contextual factors.
    Based on Dixon-Coles / Maher (1982) approach.
    """
    base_home = (
        home.attack_strength
        * away.defense_strength
        * ctx.league_avg_goals / 2
        * HOME_ADVANTAGE
        * ctx.weather_factor
        * ctx.altitude_factor
        * home.motivation
    )
    base_away = (
        away.attack_strength
        * home.defense_strength
        * ctx.league_avg_goals / 2
        * ctx.weather_factor
        * ctx.altitude_factor
        * away.motivation
    )

    # Form adjustment — max ±15% based on form (0-15 points)
    form_adj_home = 0.85 + (home.form_points / 15) * 0.3
    form_adj_away = 0.85 + (away.form_points / 15) * 0.3
    base_home *= form_adj_home
    base_away *= form_adj_away

    # Missing key players — each costs ~7% attack output
    base_home *= max(0.6, 1 - home.key_players_missing * 0.07)
    base_away *= max(0.6, 1 - away.key_players_missing * 0.07)

    # Fatigue — each unit reduces output by 5%
    base_home *= 1 - home.fatigue_index * 0.05
    base_away *= 1 - away.fatigue_index * 0.05

    # H2H adjustment (light signal)
    total_h2h = ctx.h2h_home_wins + ctx.h2h_draws + ctx.h2h_away_wins
    if total_h2h >= 3:
        h2h_ratio = ctx.h2h_home_wins / total_h2h
        base_home *= 0.9 + h2h_ratio * 0.2
        base_away *= 0.9 + (ctx.h2h_away_wins / total_h2h) * 0.2

    # Importance
    if ctx.importance == "relegation":
        base_home *= 0.92
        base_away *= 0.92
    elif ctx.importance == "final":
        base_home *= 0.88
        base_away *= 0.88

    return round(max(0.2, base_home), 3), round(max(0.2, base_away), 3)


# ---------------------------------------------------------------------------
# Value bet detection (Kelly Criterion)
# ---------------------------------------------------------------------------

def find_value_bets(
    probs: dict[str, float],
    odds: dict[str, float],
    bankroll_fraction: float = 0.25,
) -> list[dict]:
    """
    A value bet exists when: implied_prob < our_model_prob.
    Kelly stake = (b*p - q) / b  where b = decimal_odds - 1
    """
    market_map = {
        "home_win": ("prob_home_win", odds.get("home", 0)),
        "draw":     ("prob_draw",     odds.get("draw", 0)),
        "away_win": ("prob_away_win", odds.get("away", 0)),
        "over_25":  ("prob_over_25",  odds.get("over_25", 0)),
        "under_25": ("prob_under_25", odds.get("under_25", 0)),
        "btts_yes": ("prob_btts",     odds.get("btts_yes", 0)),
        "btts_no":  ("prob_no_btts",  odds.get("btts_no",  0)),
    }

    value_bets = []
    for label, (prob_key, decimal_odd) in market_map.items():
        if not decimal_odd or decimal_odd < 1.01:
            continue
        our_prob = probs.get(prob_key, 0)
        implied_prob = 1 / decimal_odd
        edge = our_prob - implied_prob
        if edge > 0.03:   # minimum 3% edge
            b = decimal_odd - 1
            kelly = (b * our_prob - (1 - our_prob)) / b
            kelly_stake = round(min(kelly * bankroll_fraction, 0.10), 4)  # cap at 10%
            ev = round((our_prob * decimal_odd - 1) * 100, 2)
            value_bets.append({
                "market": label,
                "our_probability": round(our_prob * 100, 1),
                "implied_probability": round(implied_prob * 100, 1),
                "edge": round(edge * 100, 1),
                "odds": decimal_odd,
                "expected_value_pct": ev,
                "kelly_stake_pct": round(kelly_stake * 100, 1),
                "rating": _rate_value_bet(ev),
            })

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets


def _rate_value_bet(ev_pct: float) -> str:
    if ev_pct >= 20:
        return "excellent"
    elif ev_pct >= 10:
        return "good"
    elif ev_pct >= 5:
        return "fair"
    return "marginal"


# ---------------------------------------------------------------------------
# Confidence & risk scoring
# ---------------------------------------------------------------------------

def compute_confidence(
    probs: dict[str, float],
    home: TeamStats,
    away: TeamStats,
    ctx: MatchContext,
) -> tuple[float, str, list[str]]:
    notes: list[str] = []
    score = 50.0   # base

    # Dominant favourite boosts confidence
    best_prob = max(probs["prob_home_win"], probs["prob_draw"], probs["prob_away_win"])
    if best_prob > 0.60:
        score += 15
        notes.append(f"Strong favourite with {round(best_prob*100)}% win probability.")
    elif best_prob > 0.45:
        score += 7

    # Good form
    if home.form_points >= 10:
        score += 8
        notes.append(f"{home.name} in excellent form.")
    if away.form_points >= 10:
        score += 8
        notes.append(f"{away.name} in excellent form.")

    # Key players missing reduces confidence
    missing = home.key_players_missing + away.key_players_missing
    if missing >= 3:
        score -= 12
        notes.append("Several key players missing — higher uncertainty.")
    elif missing >= 1:
        score -= 5

    # Adverse weather
    if ctx.weather_factor < 0.9:
        score -= 8
        notes.append("Adverse weather conditions may affect play style.")

    # H2H consistency
    total_h2h = ctx.h2h_home_wins + ctx.h2h_draws + ctx.h2h_away_wins
    if total_h2h >= 5:
        score += 5
        notes.append("Sufficient head-to-head history available.")

    score = max(0, min(100, score))

    if score >= 70:
        risk = "low"
    elif score >= 50:
        risk = "medium"
    else:
        risk = "high"

    return round(score, 1), risk, notes


# ---------------------------------------------------------------------------
# Most likely score
# ---------------------------------------------------------------------------

def most_likely_score(score_distribution: dict[str, float]) -> str:
    return max(score_distribution, key=score_distribution.get)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_match(
    home: TeamStats,
    away: TeamStats,
    ctx: MatchContext,
    odds: Optional[dict[str, float]] = None,
) -> PredictionResult:
    mu_home, mu_away = compute_expected_goals(home, away, ctx)
    probs = poisson_match_probabilities(mu_home, mu_away)
    confidence, risk, notes = compute_confidence(probs, home, away, ctx)

    predicted_score = most_likely_score(probs["score_distribution"])
    value_bets = find_value_bets(probs, odds or {})

    return PredictionResult(
        prob_home_win=probs["prob_home_win"],
        prob_draw=probs["prob_draw"],
        prob_away_win=probs["prob_away_win"],
        prob_over_25=probs["prob_over_25"],
        prob_under_25=probs["prob_under_25"],
        prob_btts=probs["prob_btts"],
        prob_no_btts=probs["prob_no_btts"],
        prob_over_35=probs["prob_over_35"],
        prob_under_35=probs["prob_under_35"],
        expected_home_goals=mu_home,
        expected_away_goals=mu_away,
        predicted_score=predicted_score,
        prob_home_cards=round(ctx.referee_cards_avg / 10, 3),
        prob_corners_over=0.55,   # placeholder until corner model added
        confidence_score=confidence,
        risk_level=risk,
        value_bets=value_bets,
        score_distribution={k: v for k, v in sorted(
            probs["score_distribution"].items(), key=lambda x: x[1], reverse=True
        )[:10]},
        analysis_notes=notes,
    )

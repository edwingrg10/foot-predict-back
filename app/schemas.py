from pydantic import BaseModel, EmailStr
from typing import Optional, Any
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    username: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None


# ── Predictions ───────────────────────────────────────────────────────────────

class PredictionOut(BaseModel):
    match_id: int
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    prob_over_25: float
    prob_under_25: float
    prob_btts: float
    prob_no_btts: float
    prob_over_35: float
    prob_under_35: float
    expected_home_goals: float
    expected_away_goals: float
    predicted_score: str
    confidence_score: float
    risk_level: str
    value_bets: list[dict]
    score_distribution: dict
    analysis_notes: list[str]


# ── Matches ───────────────────────────────────────────────────────────────────

class TeamInfo(BaseModel):
    id: int
    name: str
    logo: Optional[str]
    form: Optional[str]
    form_points: Optional[float]
    attack: Optional[float]
    defense: Optional[float]


class LeagueInfo(BaseModel):
    id: int
    name: str
    country: Optional[str]
    logo: Optional[str]
    avg_goals: Optional[float]


class OddsInfo(BaseModel):
    home: Optional[float]
    draw: Optional[float]
    away: Optional[float]
    over_25: Optional[float]
    under_25: Optional[float]
    btts_yes: Optional[float]
    btts_no: Optional[float]


class H2HInfo(BaseModel):
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0
    last_meetings: list[dict] = []


class MissingPlayers(BaseModel):
    home: int = 0
    away: int = 0


class MatchOut(BaseModel):
    id: int
    league: LeagueInfo
    home_team: TeamInfo
    away_team: TeamInfo
    match_date: str
    status: str
    venue: Optional[str]
    referee: Optional[str]
    weather: Optional[str]
    importance: Optional[str]
    odds: Optional[OddsInfo]
    h2h: Optional[H2HInfo]
    missing_players: Optional[MissingPlayers]
    prediction: Optional[PredictionOut]


# ── Saved Bets ────────────────────────────────────────────────────────────────

class SavedBetCreate(BaseModel):
    match_id: int
    bet_type: str
    bet_pick: str
    odds: Optional[float]
    stake: Optional[float] = 0
    notes: Optional[str]


class SavedBetOut(SavedBetCreate):
    id: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

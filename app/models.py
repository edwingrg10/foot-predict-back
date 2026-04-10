"""
Schema completo — Football Predictor
SQL Server 2019 via SQLAlchemy + pyodbc
"""
from __future__ import annotations

from sqlalchemy import (
    Column, Integer, BigInteger, SmallInteger, String, Float, Boolean,
    DateTime, Date, ForeignKey, Text, Enum, UniqueConstraint, Index,
)
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.sql import func
import enum


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────
class MatchStatus(str, enum.Enum):
    SCHEDULED  = "scheduled"
    LIVE       = "live"
    HALFTIME   = "halftime"
    FINISHED   = "finished"
    POSTPONED  = "postponed"
    CANCELLED  = "cancelled"
    ABANDONED  = "abandoned"

class MatchImportance(str, enum.Enum):
    REGULAR    = "regular"
    DERBY      = "derby"
    FINAL      = "final"
    RELEGATION = "relegation"
    CUP        = "cup"

class EventType(str, enum.Enum):
    GOAL          = "goal"
    OWN_GOAL      = "own_goal"
    PENALTY_GOAL  = "penalty_goal"
    PENALTY_MISS  = "penalty_miss"
    YELLOW_CARD   = "yellow_card"
    RED_CARD      = "red_card"
    YELLOW_RED    = "yellow_red"
    SUBSTITUTION  = "substitution"
    VAR           = "var"

class BetStatus(str, enum.Enum):
    PENDING = "pending"
    WON     = "won"
    LOST    = "lost"
    VOID    = "void"

class ScrapingStatus(str, enum.Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"


# ─────────────────────────────────────────────
# LIGAS
# ─────────────────────────────────────────────
class League(Base):
    __tablename__ = "leagues"

    id             = Column(Integer, primary_key=True)
    sofascore_id   = Column(Integer, unique=True, nullable=False, index=True)  # unique-tournament id
    name           = Column(String(100), nullable=False)
    country        = Column(String(60), nullable=False)
    logo_url       = Column(String(300))
    avg_goals      = Column(Float, default=2.5)
    is_active      = Column(Boolean, default=True)
    created_at     = Column(DateTime, server_default=func.now())

    seasons   = relationship("Season",  back_populates="league", cascade="all, delete-orphan")
    teams     = relationship("Team",    back_populates="league")
    matches   = relationship("Match",   back_populates="league")
    standings = relationship("Standing", back_populates="league")


# ─────────────────────────────────────────────
# TEMPORADAS
# ─────────────────────────────────────────────
class Season(Base):
    __tablename__ = "seasons"

    id           = Column(Integer, primary_key=True)
    league_id    = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    sofascore_id = Column(Integer, nullable=False)          # season id en Sofascore
    year         = Column(String(10), nullable=False)       # ej: "2024/2025"
    start_date   = Column(Date)
    end_date     = Column(Date)
    is_current   = Column(Boolean, default=False)
    scraped_full = Column(Boolean, default=False)           # ¿ya se cargó el histórico?

    __table_args__ = (UniqueConstraint("league_id", "sofascore_id"),)

    league   = relationship("League",  back_populates="seasons")
    matches  = relationship("Match",   back_populates="season")
    standings = relationship("Standing", back_populates="season")


# ─────────────────────────────────────────────
# EQUIPOS
# ─────────────────────────────────────────────
class Team(Base):
    __tablename__ = "teams"

    id           = Column(Integer, primary_key=True)
    sofascore_id = Column(Integer, unique=True, nullable=False, index=True)
    name         = Column(String(100), nullable=False)
    short_name   = Column(String(30))
    logo_url     = Column(String(300))
    country      = Column(String(60))
    league_id    = Column(Integer, ForeignKey("leagues.id"))
    founded_year = Column(SmallInteger)
    stadium      = Column(String(100))
    stadium_capacity = Column(Integer)
    created_at   = Column(DateTime, server_default=func.now())
    updated_at   = Column(DateTime, server_default=func.now(), onupdate=func.now())

    league         = relationship("League", back_populates="teams")
    season_stats   = relationship("TeamSeasonStats", back_populates="team", cascade="all, delete-orphan")
    home_matches   = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches   = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    players        = relationship("Player", back_populates="team")
    standings      = relationship("Standing", back_populates="team")


# ─────────────────────────────────────────────
# ESTADÍSTICAS DE EQUIPO POR TEMPORADA
# ─────────────────────────────────────────────
class TeamSeasonStats(Base):
    __tablename__ = "team_season_stats"

    id           = Column(Integer, primary_key=True)
    team_id      = Column(Integer, ForeignKey("teams.id"), nullable=False)
    season_id    = Column(Integer, ForeignKey("seasons.id"), nullable=False)

    # Forma
    form_string  = Column(String(20))   # ej: "WWDLW"
    form_points  = Column(Float, default=0)

    # Goles
    goals_scored       = Column(Integer, default=0)
    goals_conceded     = Column(Integer, default=0)
    goals_scored_avg   = Column(Float,   default=0)
    goals_conceded_avg = Column(Float,   default=0)
    goals_scored_home_avg   = Column(Float, default=0)
    goals_conceded_home_avg = Column(Float, default=0)
    goals_scored_away_avg   = Column(Float, default=0)
    goals_conceded_away_avg = Column(Float, default=0)

    # Resultados generales
    matches_played = Column(Integer, default=0)
    wins           = Column(Integer, default=0)
    draws          = Column(Integer, default=0)
    losses         = Column(Integer, default=0)
    points         = Column(Integer, default=0)

    # Casa
    home_played = Column(Integer, default=0)
    home_wins   = Column(Integer, default=0)
    home_draws  = Column(Integer, default=0)
    home_losses = Column(Integer, default=0)

    # Fuera
    away_played = Column(Integer, default=0)
    away_wins   = Column(Integer, default=0)
    away_draws  = Column(Integer, default=0)
    away_losses = Column(Integer, default=0)

    # Métricas avanzadas
    xg_for        = Column(Float, default=0)   # expected goals a favor
    xg_against    = Column(Float, default=0)
    shots_avg     = Column(Float, default=0)
    shots_on_target_avg = Column(Float, default=0)
    possession_avg      = Column(Float, default=0)
    corners_avg         = Column(Float, default=0)
    yellow_cards_avg    = Column(Float, default=0)
    red_cards_avg       = Column(Float, default=0)
    clean_sheets        = Column(Integer, default=0)
    failed_to_score     = Column(Integer, default=0)
    btts_count          = Column(Integer, default=0)   # ambos anotaron
    over25_count        = Column(Integer, default=0)

    # Índices calculados para el modelo
    attack_strength  = Column(Float, default=1.0)
    defense_strength = Column(Float, default=1.0)

    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (UniqueConstraint("team_id", "season_id"),)

    team   = relationship("Team",   back_populates="season_stats")
    season = relationship("Season")


# ─────────────────────────────────────────────
# JUGADORES
# ─────────────────────────────────────────────
class Player(Base):
    __tablename__ = "players"

    id           = Column(Integer, primary_key=True)
    sofascore_id = Column(Integer, unique=True, nullable=False, index=True)
    name         = Column(String(100), nullable=False)
    short_name   = Column(String(50))
    team_id      = Column(Integer, ForeignKey("teams.id"))
    nationality  = Column(String(60))
    position     = Column(String(30))   # Goalkeeper, Defender, Midfielder, Forward
    shirt_number = Column(SmallInteger)
    date_of_birth = Column(Date)
    height_cm    = Column(SmallInteger)
    logo_url     = Column(String(300))
    created_at   = Column(DateTime, server_default=func.now())
    updated_at   = Column(DateTime, server_default=func.now(), onupdate=func.now())

    team          = relationship("Team", back_populates="players")
    season_stats  = relationship("PlayerSeasonStats", back_populates="player", cascade="all, delete-orphan")
    match_lineups = relationship("MatchLineup", back_populates="player")
    match_events  = relationship("MatchEvent", foreign_keys="MatchEvent.player_id", back_populates="player")


# ─────────────────────────────────────────────
# ESTADÍSTICAS DE JUGADOR POR TEMPORADA
# ─────────────────────────────────────────────
class PlayerSeasonStats(Base):
    __tablename__ = "player_season_stats"

    id        = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=False)

    appearances  = Column(Integer, default=0)
    starts       = Column(Integer, default=0)
    minutes      = Column(Integer, default=0)
    goals        = Column(Integer, default=0)
    assists      = Column(Integer, default=0)
    yellow_cards = Column(Integer, default=0)
    red_cards    = Column(Integer, default=0)
    shots        = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    dribbles_completed = Column(Integer, default=0)
    pass_accuracy   = Column(Float, default=0)
    rating_avg      = Column(Float, default=0)    # rating promedio Sofascore

    __table_args__ = (UniqueConstraint("player_id", "season_id"),)

    player = relationship("Player", back_populates="season_stats")
    season = relationship("Season")


# ─────────────────────────────────────────────
# ÁRBITROS
# ─────────────────────────────────────────────
class Referee(Base):
    __tablename__ = "referees"

    id           = Column(Integer, primary_key=True)
    sofascore_id = Column(Integer, unique=True, index=True)
    name         = Column(String(100), nullable=False)
    nationality  = Column(String(60))

    # Estadísticas históricas calculadas
    matches_total     = Column(Integer, default=0)
    yellow_cards_avg  = Column(Float, default=0)
    red_cards_avg     = Column(Float, default=0)
    penalties_avg     = Column(Float, default=0)
    home_win_pct      = Column(Float, default=0)
    draw_pct          = Column(Float, default=0)
    away_win_pct      = Column(Float, default=0)
    goals_avg         = Column(Float, default=0)

    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    matches = relationship("Match", back_populates="referee")


# ─────────────────────────────────────────────
# PARTIDOS
# ─────────────────────────────────────────────
class Match(Base):
    __tablename__ = "matches"

    id           = Column(Integer, primary_key=True)
    sofascore_id = Column(BigInteger, unique=True, nullable=False, index=True)
    league_id    = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    season_id    = Column(Integer, ForeignKey("seasons.id"))
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    referee_id   = Column(Integer, ForeignKey("referees.id"))

    match_date   = Column(DateTime, nullable=False, index=True)
    round        = Column(String(50))          # "Jornada 12", "Quarter-final"
    status       = Column(Enum(MatchStatus),   default=MatchStatus.SCHEDULED)
    importance   = Column(Enum(MatchImportance), default=MatchImportance.REGULAR)

    # Resultado
    home_score         = Column(SmallInteger)
    away_score         = Column(SmallInteger)
    home_score_ht      = Column(SmallInteger)   # resultado al descanso
    away_score_ht      = Column(SmallInteger)
    home_score_et      = Column(SmallInteger)   # tiempo extra
    away_score_et      = Column(SmallInteger)
    home_score_pen     = Column(SmallInteger)   # penales
    away_score_pen     = Column(SmallInteger)

    # Estadio / clima
    venue        = Column(String(100))
    city         = Column(String(60))
    attendance   = Column(Integer)
    weather      = Column(String(50))
    temperature  = Column(Float)

    # Flags de scraping
    stats_scraped    = Column(Boolean, default=False)   # ¿ya se scraparon las stats?
    events_scraped   = Column(Boolean, default=False)
    lineups_scraped  = Column(Boolean, default=False)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    league     = relationship("League",   back_populates="matches")
    season     = relationship("Season",   back_populates="matches")
    home_team  = relationship("Team",     foreign_keys=[home_team_id], back_populates="home_matches")
    away_team  = relationship("Team",     foreign_keys=[away_team_id], back_populates="away_matches")
    referee    = relationship("Referee",  back_populates="matches")
    stats      = relationship("MatchStats",   back_populates="match",  uselist=False, cascade="all, delete-orphan")
    events     = relationship("MatchEvent",   back_populates="match",  cascade="all, delete-orphan")
    lineups    = relationship("MatchLineup",  back_populates="match",  cascade="all, delete-orphan")
    prediction = relationship("Prediction",   back_populates="match",  uselist=False, cascade="all, delete-orphan")


# ─────────────────────────────────────────────
# ESTADÍSTICAS DEL PARTIDO
# ─────────────────────────────────────────────
class MatchStats(Base):
    __tablename__ = "match_stats"

    id       = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), unique=True, nullable=False)

    # Posesión
    home_possession = Column(Float)
    away_possession = Column(Float)

    # Tiros
    home_shots            = Column(SmallInteger)
    away_shots            = Column(SmallInteger)
    home_shots_on_target  = Column(SmallInteger)
    away_shots_on_target  = Column(SmallInteger)
    home_shots_off_target = Column(SmallInteger)
    away_shots_off_target = Column(SmallInteger)
    home_blocked_shots    = Column(SmallInteger)
    away_blocked_shots    = Column(SmallInteger)

    # Goles esperados
    home_xg = Column(Float)
    away_xg = Column(Float)

    # Corners
    home_corners = Column(SmallInteger)
    away_corners = Column(SmallInteger)

    # Faltas y tarjetas
    home_fouls        = Column(SmallInteger)
    away_fouls        = Column(SmallInteger)
    home_yellow_cards = Column(SmallInteger)
    away_yellow_cards = Column(SmallInteger)
    home_red_cards    = Column(SmallInteger)
    away_red_cards    = Column(SmallInteger)

    # Penales
    home_penalties_awarded = Column(SmallInteger, default=0)
    away_penalties_awarded = Column(SmallInteger, default=0)

    # Pases
    home_passes          = Column(Integer)
    away_passes          = Column(Integer)
    home_pass_accuracy   = Column(Float)
    away_pass_accuracy   = Column(Float)
    home_key_passes      = Column(SmallInteger)
    away_key_passes      = Column(SmallInteger)

    # Duelos
    home_tackles         = Column(SmallInteger)
    away_tackles         = Column(SmallInteger)
    home_interceptions   = Column(SmallInteger)
    away_interceptions   = Column(SmallInteger)
    home_aerial_won      = Column(SmallInteger)
    away_aerial_won      = Column(SmallInteger)

    # Ataques
    home_attacks         = Column(SmallInteger)
    away_attacks         = Column(SmallInteger)
    home_dangerous_attacks = Column(SmallInteger)
    away_dangerous_attacks = Column(SmallInteger)

    # Distancia recorrida (km)
    home_distance_covered = Column(Float)
    away_distance_covered = Column(Float)

    match = relationship("Match", back_populates="stats")


# ─────────────────────────────────────────────
# EVENTOS DEL PARTIDO (goles, tarjetas, subs)
# ─────────────────────────────────────────────
class MatchEvent(Base):
    __tablename__ = "match_events"

    id         = Column(Integer, primary_key=True)
    match_id   = Column(Integer, ForeignKey("matches.id"), nullable=False)
    player_id  = Column(Integer, ForeignKey("players.id"))
    player2_id = Column(Integer, ForeignKey("players.id"))  # asistencia / jugador que entra
    team_id    = Column(Integer, ForeignKey("teams.id"))

    event_type = Column(Enum(EventType), nullable=False)
    minute     = Column(SmallInteger)
    extra_time = Column(SmallInteger, default=0)   # minuto adicional (ej: 90+3)
    is_home    = Column(Boolean)                   # True = equipo local
    description = Column(String(200))

    Index("ix_match_events_match", "match_id")

    match   = relationship("Match",  back_populates="events")
    player  = relationship("Player", foreign_keys=[player_id], back_populates="match_events")
    player2 = relationship("Player", foreign_keys=[player2_id])
    team    = relationship("Team")


# ─────────────────────────────────────────────
# ALINEACIONES
# ─────────────────────────────────────────────
class MatchLineup(Base):
    __tablename__ = "match_lineups"

    id         = Column(Integer, primary_key=True)
    match_id   = Column(Integer, ForeignKey("matches.id"), nullable=False)
    team_id    = Column(Integer, ForeignKey("teams.id"),   nullable=False)
    player_id  = Column(Integer, ForeignKey("players.id"), nullable=False)

    is_home       = Column(Boolean)
    is_starter    = Column(Boolean, default=True)
    shirt_number  = Column(SmallInteger)
    position      = Column(String(30))
    minutes_played = Column(SmallInteger)
    rating        = Column(Float)          # rating Sofascore del partido

    # Stats individuales del partido
    goals         = Column(SmallInteger, default=0)
    assists       = Column(SmallInteger, default=0)
    shots         = Column(SmallInteger, default=0)
    shots_on_target = Column(SmallInteger, default=0)
    yellow_cards  = Column(SmallInteger, default=0)
    red_cards     = Column(SmallInteger, default=0)
    passes        = Column(SmallInteger, default=0)
    pass_accuracy = Column(Float)
    dribbles      = Column(SmallInteger, default=0)
    tackles       = Column(SmallInteger, default=0)

    __table_args__ = (UniqueConstraint("match_id", "player_id"),)

    match  = relationship("Match",  back_populates="lineups")
    team   = relationship("Team")
    player = relationship("Player", back_populates="match_lineups")


# ─────────────────────────────────────────────
# CLASIFICACIONES
# ─────────────────────────────────────────────
class Standing(Base):
    __tablename__ = "standings"

    id        = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=False)
    team_id   = Column(Integer, ForeignKey("teams.id"),   nullable=False)

    position      = Column(SmallInteger)
    matches_played = Column(SmallInteger, default=0)
    wins          = Column(SmallInteger, default=0)
    draws         = Column(SmallInteger, default=0)
    losses        = Column(SmallInteger, default=0)
    goals_for     = Column(SmallInteger, default=0)
    goals_against = Column(SmallInteger, default=0)
    goal_diff     = Column(SmallInteger, default=0)
    points        = Column(SmallInteger, default=0)
    updated_at    = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (UniqueConstraint("season_id", "team_id"),)

    league = relationship("League", back_populates="standings")
    season = relationship("Season", back_populates="standings")
    team   = relationship("Team",   back_populates="standings")


# ─────────────────────────────────────────────
# PREDICCIONES
# ─────────────────────────────────────────────
class Prediction(Base):
    __tablename__ = "predictions"

    id       = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), unique=True)

    prob_home_win   = Column(Float, default=0)
    prob_draw       = Column(Float, default=0)
    prob_away_win   = Column(Float, default=0)
    prob_over_25    = Column(Float, default=0)
    prob_under_25   = Column(Float, default=0)
    prob_over_35    = Column(Float, default=0)
    prob_under_35   = Column(Float, default=0)
    prob_btts       = Column(Float, default=0)
    prob_no_btts    = Column(Float, default=0)

    # Corners
    prob_over_95_corners  = Column(Float, default=0)
    prob_under_95_corners = Column(Float, default=0)
    expected_home_corners = Column(Float, default=0)
    expected_away_corners = Column(Float, default=0)

    # Tarjetas
    prob_over_35_cards    = Column(Float, default=0)
    prob_under_35_cards   = Column(Float, default=0)
    expected_home_cards   = Column(Float, default=0)
    expected_away_cards   = Column(Float, default=0)

    expected_home_goals = Column(Float, default=0)
    expected_away_goals = Column(Float, default=0)
    predicted_score     = Column(String(10))        # ej: "2-1"
    confidence_score    = Column(Float, default=0)
    risk_level          = Column(String(10), default="medium")
    value_bets          = Column(Text)              # JSON string
    match_summary       = Column(Text)              # Resumen narrativo del partido
    smart_bet           = Column(Text)              # JSON: apuesta recomendada (simple o combinada)
    model_version       = Column(String(20), default="2.0")

    # ── Evaluación post-partido ───────────────────────────────────────────────
    actual_outcome    = Column(String(1))      # 'H', 'D', 'A'
    actual_goals      = Column(SmallInteger)   # goles totales reales
    actual_corners    = Column(SmallInteger)   # corners totales reales
    actual_home_cards = Column(SmallInteger)   # tarjetas amarillas local
    actual_away_cards = Column(SmallInteger)   # tarjetas amarillas visitante

    outcome_correct   = Column(Boolean)        # ¿acertó 1X2?
    over25_correct    = Column(Boolean)        # ¿acertó over/under 2.5?
    btts_correct      = Column(Boolean)        # ¿acertó BTTS?
    corners_correct   = Column(Boolean)        # ¿acertó corners over 9.5?
    cards_correct     = Column(Boolean)        # ¿acertó tarjetas over 3.5?
    brier_1x2         = Column(Float)          # Brier score 1X2 (0=perfecto, 2=pésimo)
    smart_bet_correct = Column(Boolean)        # ¿la apuesta recomendada ganó?
    evaluated_at      = Column(DateTime)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    match = relationship("Match", back_populates="prediction")


# ─────────────────────────────────────────────
# LOG DE SCRAPING
# ─────────────────────────────────────────────
class ScrapingLog(Base):
    __tablename__ = "scraping_log"

    id         = Column(Integer, primary_key=True)
    job_type   = Column(String(50), nullable=False)   # "daily_matches", "match_stats", "standings", "historical"
    target     = Column(String(100))                  # fecha, league_id, match_id, etc.
    status     = Column(Enum(ScrapingStatus), default=ScrapingStatus.PENDING)
    records_inserted = Column(Integer, default=0)
    records_updated  = Column(Integer, default=0)
    error_message    = Column(Text)
    started_at  = Column(DateTime)
    finished_at = Column(DateTime)
    created_at  = Column(DateTime, server_default=func.now())


# ─────────────────────────────────────────────
# USUARIOS Y APUESTAS GUARDADAS
# ─────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True)
    email           = Column(String(150), unique=True, nullable=False, index=True)
    username        = Column(String(50),  unique=True, nullable=False, index=True)
    hashed_password = Column(String(200), nullable=False)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, server_default=func.now())

    saved_bets         = relationship("SavedBet",         back_populates="user")
    prediction_history = relationship("PredictionHistory", back_populates="user")


class SavedBet(Base):
    __tablename__ = "saved_bets"

    id       = Column(Integer, primary_key=True)
    user_id  = Column(Integer, ForeignKey("users.id"), nullable=False)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    bet_type = Column(String(30), nullable=False)   # "1X2", "over_under", "btts", "handicap"
    bet_pick = Column(String(30), nullable=False)   # "home", "draw", "away", "over", etc.
    odds     = Column(Float)
    stake    = Column(Float, default=0)
    status   = Column(Enum(BetStatus), default=BetStatus.PENDING)
    notes    = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

    user  = relationship("User",  back_populates="saved_bets")
    match = relationship("Match")


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id                  = Column(Integer, primary_key=True)
    user_id             = Column(Integer, ForeignKey("users.id"), nullable=False)
    match_id            = Column(Integer, ForeignKey("matches.id"), nullable=False)
    prediction_snapshot = Column(Text)   # JSON snapshot de la predicción
    actual_result       = Column(String(10))
    was_correct         = Column(Boolean)
    created_at          = Column(DateTime, server_default=func.now())

    user  = relationship("User",  back_populates="prediction_history")
    match = relationship("Match")

"""
Ligas objetivo y sus IDs en Sofascore.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LeagueConfig:
    sofascore_id: int        # unique-tournament id en Sofascore
    name: str
    country: str
    logo_url: str
    avg_goals: float
    category_slug: str       # para validar al parsear eventos diarios


TARGET_LEAGUES: list[LeagueConfig] = [
    LeagueConfig(
        sofascore_id=17,
        name="Premier League",
        country="England",
        logo_url="https://img.sofascore.com/api/v1/unique-tournament/17/image",
        avg_goals=2.72,
        category_slug="england",
    ),
    LeagueConfig(
        sofascore_id=8,
        name="La Liga",
        country="Spain",
        logo_url="https://img.sofascore.com/api/v1/unique-tournament/8/image",
        avg_goals=2.54,
        category_slug="spain",
    ),
    LeagueConfig(
        sofascore_id=7,
        name="UEFA Champions League",
        country="Europe",
        logo_url="https://img.sofascore.com/api/v1/unique-tournament/7/image",
        avg_goals=2.85,
        category_slug="europe",
    ),
    LeagueConfig(
        sofascore_id=11536,
        name="Liga BetPlay DIMAYOR",
        country="Colombia",
        logo_url="https://img.sofascore.com/api/v1/unique-tournament/11536/image",
        avg_goals=2.30,
        category_slug="colombia",
    ),
    # 2026: DIMAYOR cambió de ID a 11539 (Liga DIMAYOR, Apertura)
    LeagueConfig(
        sofascore_id=11539,
        name="Liga BetPlay DIMAYOR",
        country="Colombia",
        logo_url="https://img.sofascore.com/api/v1/unique-tournament/11536/image",
        avg_goals=2.30,
        category_slug="colombia",
    ),
]

# Para acceso rápido por sofascore_id
LEAGUES_BY_ID: dict[int, LeagueConfig] = {lg.sofascore_id: lg for lg in TARGET_LEAGUES}

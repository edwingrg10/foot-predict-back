"""
Data service — wrapper sobre SofascoreService.
Mantiene la interfaz async que usan los routers.
"""
from __future__ import annotations
from datetime import date, timedelta
from typing import Optional
from .sofascore_service import sofascore_service, TARGET_LEAGUES

# Exporta LEAGUES para compatibilidad con routers existentes
LEAGUES: dict[int, dict] = {cfg["id"]: cfg for cfg in TARGET_LEAGUES.values()}


class FootballDataService:

    async def get_today_matches(self, league_id: Optional[int] = None) -> list[dict]:
        all_matches = sofascore_service.fetch_today()
        if league_id:
            return [m for m in all_matches if m.get("league", {}).get("id") == league_id]
        return all_matches

    async def get_tomorrow_matches(self, league_id: Optional[int] = None) -> list[dict]:
        all_matches = sofascore_service.fetch_tomorrow()
        if league_id:
            return [m for m in all_matches if m.get("league", {}).get("id") == league_id]
        return all_matches

    async def get_match_detail(self, fixture_id: int) -> Optional[dict]:
        today_str    = date.today().isoformat()
        tomorrow_str = (date.today() + timedelta(days=1)).isoformat()
        for date_str in [today_str, tomorrow_str]:
            matches = sofascore_service.fetch_date(date_str)
            found = next((m for m in matches if m.get("id") == fixture_id), None)
            if found:
                return found
        return None

    def get_today_leagues(self) -> list[dict]:
        return sofascore_service.get_leagues_for_date(date.today().isoformat())

    def get_tomorrow_leagues(self) -> list[dict]:
        return sofascore_service.get_leagues_for_date(
            (date.today() + timedelta(days=1)).isoformat()
        )

    def invalidate_cache(self) -> None:
        sofascore_service.invalidate_cache()


data_service = FootballDataService()

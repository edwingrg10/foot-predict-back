"""
Sofascore scraper — usa Playwright para bypassear protección anti-bot.
Endpoint real: www.sofascore.com/api/v1/sport/football/scheduled-events/{date}
Solo 5 ligas: Premier, La Liga, Champions, Libertadores, Sudamericana.
"""
from __future__ import annotations

import threading
from datetime import date, datetime, timedelta, timezone
from typing import Optional

# ── Ligas objetivo ───────────────────────────────────────────────────────────
TARGET_LEAGUES: dict[str, dict] = {
    # keyword → config   (keyword se busca en "nombre_torneo categoria")
    "premier league":   {"id": 39,  "name": "Premier League",        "country": "England",       "avg_goals": 2.72},
    "laliga":           {"id": 140, "name": "La Liga",               "country": "Spain",         "avg_goals": 2.54},
    "champions league": {"id": 2,   "name": "UEFA Champions League", "country": "Europe",        "avg_goals": 2.85},
    "libertadores":     {"id": 13,  "name": "CONMEBOL Libertadores", "country": "South America", "avg_goals": 2.48},
    "sudamericana":     {"id": 11,  "name": "CONMEBOL Sudamericana", "country": "South America", "avg_goals": 2.35},
    "dimayor":          {"id": 239, "name": "Liga BetPlay DIMAYOR",  "country": "Colombia",      "avg_goals": 2.30},
}

LEAGUE_LOGOS: dict[int, str] = {
    39:  "https://img.sofascore.com/api/v1/unique-tournament/17/image",
    140: "https://img.sofascore.com/api/v1/unique-tournament/8/image",
    2:   "https://img.sofascore.com/api/v1/unique-tournament/7/image",
    13:  "https://img.sofascore.com/api/v1/unique-tournament/384/image",
    11:  "https://img.sofascore.com/api/v1/unique-tournament/480/image",
    239: "https://img.sofascore.com/api/v1/unique-tournament/11536/image",
}

_SF_STATUS: dict[str, str] = {
    "notstarted":     "NS",
    "inprogress":     "LIVE",
    "halftime":       "HT",
    "finished":       "FT",
    "postponed":      "PST",
    "canceled":       "CANC",
    "cancelled":      "CANC",
    "abandoned":      "ABD",
    "extra time":     "ET",
    "penalties":      "PEN",
    "awaiting extra": "BT",
}

# ── Cache ────────────────────────────────────────────────────────────────────
_cache: dict[str, list[dict]] = {}
_lock  = threading.Lock()

BASE_URL = "https://www.sofascore.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# Para ligas cuyo keyword es genérico, exigimos que el slug de categoría coincida
_LEAGUE_COUNTRY_REQUIRED: dict[str, str] = {
    "premier league":   "england",
    "laliga":           "spain",
    "dimayor":          "colombia",
    "champions league": "europe",   # excluye CAF/AFC Champions League
}


def _match_league(tournament_name: str, category_name: str = "") -> Optional[dict]:
    t_lower   = tournament_name.lower()
    cat_lower = category_name.lower()
    full      = f"{t_lower} {cat_lower}"

    for kw, cfg in TARGET_LEAGUES.items():
        if kw in full:
            required = _LEAGUE_COUNTRY_REQUIRED.get(kw)
            if required and required not in cat_lower:
                continue
            return cfg
    return None


def _parse_status(event: dict) -> tuple[str, Optional[int], Optional[int]]:
    status_obj = event.get("status", {})
    stype = status_obj.get("type", "notstarted").lower().replace(" ", "")
    desc  = status_obj.get("description", "").lower()
    code  = _SF_STATUS.get(stype) or _SF_STATUS.get(desc, "NS")

    home_score = event.get("homeScore", {}).get("current")
    away_score = event.get("awayScore", {}).get("current")

    if code == "LIVE":
        if "1st" in desc or "1er" in desc:
            code = "1H"
        elif "2nd" in desc or "2do" in desc:
            code = "2H"

    return code, home_score, away_score


def _parse_match_time(event: dict) -> str:
    ts = event.get("startTimestamp")
    if ts:
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            pass
    return f"{date.today().isoformat()}T00:00:00+00:00"


def _parse_event(ev: dict) -> Optional[dict]:
    tournament = ev.get("tournament", {})
    category   = tournament.get("category", {})
    cfg = _match_league(tournament.get("name", ""), category.get("name", ""))
    if not cfg:
        return None

    lid = cfg["id"]
    status, home_score, away_score = _parse_status(ev)
    home = ev.get("homeTeam", {})
    away = ev.get("awayTeam", {})

    return {
        "id":        ev.get("id"),
        "league": {
            "id":        lid,
            "name":      cfg["name"],
            "country":   cfg["country"],
            "logo":      LEAGUE_LOGOS.get(lid, ""),
            "avg_goals": cfg["avg_goals"],
        },
        "home_team": {
            "id":   home.get("id"),
            "name": home.get("name", ""),
            "logo": f"https://img.sofascore.com/api/v1/team/{home['id']}/image" if home.get("id") else "",
        },
        "away_team": {
            "id":   away.get("id"),
            "name": away.get("name", ""),
            "logo": f"https://img.sofascore.com/api/v1/team/{away['id']}/image" if away.get("id") else "",
        },
        "match_date":  _parse_match_time(ev),
        "status":      status,
        "home_score":  home_score,
        "away_score":  away_score,
        "venue":       ev.get("venue", {}).get("name") if ev.get("venue") else None,
        "referee":     None,
        "weather":     None,
        "importance":  "regular",
        "odds":        {},
        "h2h":         {"home_wins": 0, "draws": 0, "away_wins": 0, "last_meetings": []},
        "missing_players": {"home": 0, "away": 0},
    }


def _fetch_with_playwright(date_str: str) -> list[dict]:
    """Usa Playwright para hacer el request con cookies reales del browser."""
    from playwright.sync_api import sync_playwright

    print(f"[Sofascore] Playwright fetch {date_str}...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=USER_AGENT,
                locale="es-ES",
            )
            page = ctx.new_page()

            # Visitar la página principal para obtener cookies de sesión
            try:
                page.goto(f"{BASE_URL}/", wait_until="domcontentloaded", timeout=20000)
            except Exception:
                pass
            page.wait_for_timeout(1000)

            # Fetch de eventos con las cookies del browser
            result = page.evaluate(f"""async () => {{
                const r = await fetch(
                    '{BASE_URL}/api/v1/sport/football/scheduled-events/{date_str}',
                    {{
                        headers: {{
                            'Accept': 'application/json',
                            'Origin': '{BASE_URL}',
                            'Referer': '{BASE_URL}/',
                        }}
                    }}
                );
                if (!r.ok) return {{ error: r.status, events: [] }};
                const data = await r.json();
                return {{ events: data.events || [] }};
            }}""")

            browser.close()

            raw = result.get("events", [])
            print(f"[Sofascore] {len(raw)} eventos totales de Sofascore para {date_str}")

            # Filtrar por fecha real (timestamp)
            target = date.fromisoformat(date_str)
            def on_target_date(ev: dict) -> bool:
                ts = ev.get("startTimestamp")
                if not ts:
                    return True
                return datetime.fromtimestamp(ts, tz=timezone.utc).date() == target

            filtered = [ev for ev in raw if on_target_date(ev)]

            matches = [m for ev in filtered if (m := _parse_event(ev))]
            print(f"[Sofascore] {len(matches)} partidos en ligas objetivo para {date_str}")
            return matches

    except Exception as e:
        print(f"[Sofascore] Playwright error: {e}")
        return []


class SofascoreService:

    def fetch_date(self, date_str: str) -> list[dict]:
        with _lock:
            if date_str in _cache:
                print(f"[Sofascore] Cache hit {date_str} — {len(_cache[date_str])} partidos")
                return _cache[date_str]

        matches = _fetch_with_playwright(date_str)

        with _lock:
            _cache[date_str] = matches
        return matches

    def fetch_today(self) -> list[dict]:
        return self.fetch_date(date.today().isoformat())

    def fetch_tomorrow(self) -> list[dict]:
        return self.fetch_date((date.today() + timedelta(days=1)).isoformat())

    def get_leagues_for_date(self, date_str: str) -> list[dict]:
        matches = self.fetch_date(date_str)
        seen: dict[int, dict] = {}
        for m in matches:
            lg = m.get("league", {})
            lid = lg.get("id")
            if lid and lid not in seen:
                seen[lid] = {
                    "id":        lid,
                    "name":      lg.get("name", ""),
                    "country":   lg.get("country", ""),
                    "logo":      lg.get("logo", ""),
                    "avg_goals": lg.get("avg_goals", 2.5),
                }
        return sorted(seen.values(), key=lambda x: x["name"])

    def invalidate_cache(self, date_str: Optional[str] = None) -> None:
        with _lock:
            if date_str:
                _cache.pop(date_str, None)
            else:
                _cache.clear()


sofascore_service = SofascoreService()

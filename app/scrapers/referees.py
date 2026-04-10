"""
Scraper de árbitros.
"""
from __future__ import annotations

import logging
from sqlalchemy.orm import Session

from ..models import Referee

log = logging.getLogger("scraper.referees")


def upsert_referee(db: Session, raw: dict | None) -> Referee | None:
    if not raw or not raw.get("id"):
        return None

    sf_id = raw["id"]
    ref   = db.query(Referee).filter_by(sofascore_id=sf_id).first()
    if not ref:
        ref = Referee(
            sofascore_id=sf_id,
            name=raw.get("name", ""),
            nationality=raw.get("country", {}).get("name") if raw.get("country") else None,
        )
        db.add(ref)
        db.flush()
        log.debug(f"[Referees] Nuevo árbitro: {ref.name}")
    return ref

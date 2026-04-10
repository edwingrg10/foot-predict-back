from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import SavedBet, User
from ..schemas import SavedBetCreate, SavedBetOut
from ..auth import get_current_user
from typing import List

router = APIRouter(prefix="/bets", tags=["bets"])


@router.post("/", response_model=SavedBetOut, status_code=201)
def save_bet(
    payload: SavedBetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    bet = SavedBet(**payload.model_dump(), user_id=current_user.id)
    db.add(bet)
    db.commit()
    db.refresh(bet)
    return bet


@router.get("/", response_model=List[SavedBetOut])
def list_bets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(SavedBet).filter(SavedBet.user_id == current_user.id).all()


@router.delete("/{bet_id}", status_code=204)
def delete_bet(
    bet_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    bet = db.query(SavedBet).filter(
        SavedBet.id == bet_id, SavedBet.user_id == current_user.id
    ).first()
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")
    db.delete(bet)
    db.commit()

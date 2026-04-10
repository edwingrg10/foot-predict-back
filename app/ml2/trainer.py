"""
Trainer — entrena modelos XGBoost con el historial de partidos.

Modelos:
  - result_1x2    : XGBClassifier (3 clases: home/draw/away)
  - over_25       : XGBClassifier (binario)
  - over_35       : XGBClassifier (binario)
  - btts          : XGBClassifier (binario)
"""
from __future__ import annotations

import os
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from ..models import Match, MatchStatus
from .features import build_features, build_targets, FEATURE_NAMES

log = logging.getLogger("ml2.trainer")

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_FILES = {
    "result_1x2":    MODELS_DIR / "model_1x2.pkl",
    "over_25":       MODELS_DIR / "model_over25.pkl",
    "over_35":       MODELS_DIR / "model_over35.pkl",
    "btts":          MODELS_DIR / "model_btts.pkl",
    "corners_over_95": MODELS_DIR / "model_corners_over95.pkl",
    "cards_over_35":   MODELS_DIR / "model_cards_over35.pkl",
}


def build_dataset(db: Session) -> tuple[np.ndarray, dict]:
    """
    Recorre todos los partidos terminados con stats y construye X, y.
    Retorna (X, {target_name: y_array})
    """
    log.info("[Trainer] Construyendo dataset...")

    finished = (
        db.query(Match)
        .filter(
            Match.status == MatchStatus.FINISHED,
            Match.stats_scraped == True,
            Match.home_score.isnot(None),
        )
        .order_by(Match.match_date)
        .all()
    )
    log.info(f"[Trainer] Partidos terminados con stats: {len(finished)}")

    X_rows       = []
    y_1x2        = []
    y_o25        = []
    y_o35        = []
    y_btts       = []
    y_corners    = []   # puede tener menos muestras (solo si hay stats)
    y_cards      = []
    X_corners    = []   # feature vectors para partidos con stats de corners
    X_cards      = []

    for match in finished:
        feats   = build_features(db, match)
        targets = build_targets(match, db=db)
        if feats is None or targets is None:
            continue
        if np.isnan(feats).any():
            continue

        X_rows.append(feats)
        y_1x2.append(targets["result_1x2"])
        y_o25.append(targets["over_25"])
        y_o35.append(targets["over_35"])
        y_btts.append(targets["btts"])

        if "corners_over_95" in targets:
            X_corners.append(feats)
            y_corners.append(targets["corners_over_95"])
        if "cards_over_35" in targets:
            X_cards.append(feats)
            y_cards.append(targets["cards_over_35"])

    if not X_rows:
        return None, {}

    X = np.vstack(X_rows)
    log.info(f"[Trainer] Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    log.info(f"[Trainer] Corners samples: {len(y_corners)} | Cards samples: {len(y_cards)}")

    result = {
        "result_1x2": np.array(y_1x2),
        "over_25":    np.array(y_o25),
        "over_35":    np.array(y_o35),
        "btts":       np.array(y_btts),
    }
    if len(y_corners) >= 20:
        result["corners_over_95"] = (np.vstack(X_corners), np.array(y_corners))
    if len(y_cards) >= 20:
        result["cards_over_35"]   = (np.vstack(X_cards),   np.array(y_cards))

    return X, result


def train_all(db: Session) -> dict:
    """Entrena todos los modelos y los guarda en disco."""
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError as e:
        log.error(f"[Trainer] Dependencia faltante: {e}")
        return {}

    X, targets = build_dataset(db)
    if X is None or len(X) < 30:
        log.warning(f"[Trainer] Dataset insuficiente ({len(X) if X is not None else 0} samples). Se necesitan al menos 30.")
        return {}

    results = {}

    for target_name, payload in targets.items():
        # payload es np.ndarray (y) para modelos base, o (X_specific, y) para corners/cards
        if isinstance(payload, tuple):
            X_fit, y = payload
        else:
            X_fit, y = X, payload

        log.info(f"[Trainer] Entrenando modelo: {target_name} ({len(y)} samples)...")

        n_classes = len(np.unique(y))
        from sklearn.calibration import CalibratedClassifierCV

        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=4,          # reducido para evitar sobreajuste
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,   # regularización extra
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )),
        ])
        # Calibración isotónica: corrige las probabilidades extremas del XGBoost
        cv_folds = min(3, len(y) // 30)
        model = CalibratedClassifierCV(base, method="isotonic", cv=max(2, cv_folds))

        # Cross-validation para medir accuracy
        try:
            cv_scores = cross_val_score(model, X_fit, y, cv=min(5, len(y) // 10), scoring="accuracy")
            acc = cv_scores.mean()
            log.info(f"[Trainer] {target_name} — CV accuracy: {acc:.3f} ± {cv_scores.std():.3f}")
        except Exception as e:
            log.warning(f"[Trainer] CV falló para {target_name}: {e}")
            acc = 0.0

        # Entrenar con todos los datos
        model.fit(X_fit, y)

        # Guardar modelo
        path = MODEL_FILES.get(target_name)
        if path:
            with open(path, "wb") as f:
                pickle.dump(model, f)

        results[target_name] = {
            "samples":  len(y),
            "accuracy": round(acc, 4),
            "classes":  n_classes,
        }
        log.info(f"[Trainer] {target_name} guardado")

    # Guardar feature names para validación
    with open(MODELS_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(FEATURE_NAMES, f)

    log.info(f"[Trainer] Entrenamiento completo: {list(results.keys())}")
    return results


def load_models() -> dict:
    """Carga los modelos desde disco."""
    models = {}
    for name, path in MODEL_FILES.items():
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            log.info(f"[Trainer] Modelo cargado: {name}")
        else:
            log.warning(f"[Trainer] Modelo no encontrado: {path}")
    return models


REQUIRED_MODELS = {"result_1x2", "over_25", "over_35", "btts"}

def models_exist() -> bool:
    return all(MODEL_FILES[k].exists() for k in REQUIRED_MODELS)

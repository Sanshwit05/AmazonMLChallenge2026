"""
Stage E — Matching Model (LightGBM)
Stage F — Threshold Tuning & Conflict Resolution

Pairwise binary classification: given features for an (S1, candidate) pair,
predict match probability. Then apply threshold tuning optimized for F_0.5,
plus optional injective constraint (one S2/S3 record per S1 entity maximum).
"""

import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from .config import LGBM_PARAMS, MATCH_THRESHOLD, RANDOM_SEED
from .features import FEATURE_COLUMNS


def get_feature_columns(features_df: pd.DataFrame = None) -> List[str]:
    """Return the list of feature column names (everything except IDs and label)."""
    if features_df is not None:
        return [c for c in features_df.columns if c not in ("s1_id", "cand_id", "label", "match_prob")]
    return FEATURE_COLUMNS + ["best_alias_score"]


# ──────────────────────────────────────────────────────────────
# F_0.5 metric
# ──────────────────────────────────────────────────────────────

def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta score."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, float, float]:
    """Compute macro-averaged F_0.5 score across all S1 entities.

    Both predictions and ground_truth map S1 entity_id → set of matched entity_ids.
    Singletons have empty sets.

    Returns: (macro_f05, macro_precision, macro_recall)
    """
    f05_scores = []
    precisions = []
    recalls = []

    for s1_id in ground_truth:
        pred_set = predictions.get(s1_id, set())
        true_set = ground_truth[s1_id]

        if len(true_set) == 0 and len(pred_set) == 0:
            # True singleton, correctly predicted as singleton
            f05_scores.append(1.0)
            precisions.append(1.0)
            recalls.append(1.0)
        elif len(true_set) == 0 and len(pred_set) > 0:
            # True singleton, but we predicted matches → 0
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(1.0)  # vacuously
        elif len(true_set) > 0 and len(pred_set) == 0:
            # True matches exist but we predicted none
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
        else:
            tp = len(pred_set & true_set)
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(true_set) if true_set else 0.0
            f05 = f_beta_score(precision, recall, beta=0.5)
            f05_scores.append(f05)
            precisions.append(precision)
            recalls.append(recall)

    macro_f05 = np.mean(f05_scores) if f05_scores else 0.0
    macro_p = np.mean(precisions) if precisions else 0.0
    macro_r = np.mean(recalls) if recalls else 0.0

    return macro_f05, macro_p, macro_r


# ──────────────────────────────────────────────────────────────
# Train LightGBM model
# ──────────────────────────────────────────────────────────────

def train_model(
    features_df: pd.DataFrame,
    model_save_path: Optional[str] = None,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier on the feature dataframe.

    features_df must have columns: s1_id, cand_id, label, and all feature columns.
    """
    print("[Stage E] Training LightGBM matching model...")

    feat_cols = get_feature_columns(features_df)
    X = features_df[feat_cols]
    y = features_df["label"].values

    print(f"  Features: {len(feat_cols)}, Samples: {len(y)}")
    print(f"  Positive rate: {y.mean():.4f} ({int(y.sum())}/{len(y)})")

    params = LGBM_PARAMS.copy()
    params["scale_pos_weight"] = 1.0  # Optimize for precision / F0.5

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X, y,
        eval_set=[(X, y)],
    )

    print("  Training complete.")

    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(feat_cols, importance), key=lambda x: -x[1])
    print("  Top 15 features by importance:")
    for name, imp in feat_imp[:15]:
        print(f"    {name}: {imp}")

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, "wb") as f:
            pickle.dump((model, feat_cols), f)
        print(f"  Model saved to {model_save_path}")

    return model


def train_model_with_validation(
    features_df: pd.DataFrame,
    val_features_df: pd.DataFrame,
    model_save_path: Optional[str] = None,
) -> Tuple[lgb.LGBMClassifier, List[str]]:
    """Train with early stopping on validation set.

    Returns (model, feature_columns).
    """
    print("[Stage E] Training LightGBM with validation...")

    feat_cols = get_feature_columns(features_df)
    X_train = features_df[feat_cols]
    y_train = features_df["label"].values
    X_val = val_features_df[feat_cols]
    y_val = val_features_df["label"].values

    print(f"  Train: {len(y_train)} samples, pos rate: {y_train.mean():.4f}")
    print(f"  Val:   {len(y_val)} samples, pos rate: {y_val.mean():.4f}")

    params = LGBM_PARAMS.copy()
    params["scale_pos_weight"] = 1.0  # Optimize for precision / F0.5 instead of recall
    params["n_estimators"] = 1500  # higher, rely on early stopping

    model = lgb.LGBMClassifier(**params)

    callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    print(f"  Best iteration: {model.best_iteration_}")

    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(feat_cols, importance), key=lambda x: -x[1])
    print("  Top 15 features by importance:")
    for name, imp in feat_imp[:15]:
        print(f"    {name}: {imp}")

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, "wb") as f:
            pickle.dump((model, feat_cols), f)
        print(f"  Model saved to {model_save_path}")

    return model, feat_cols


# ──────────────────────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────────────────────

def predict_probabilities(
    model: lgb.LGBMClassifier,
    features_df: pd.DataFrame,
    feat_cols: List[str],
) -> pd.DataFrame:
    """Add match probability predictions to the features dataframe."""
    X = features_df[feat_cols]
    probs = model.predict_proba(X)[:, 1]
    # Do not copy to save memory
    features_df["match_prob"] = probs
    return features_df


# ──────────────────────────────────────────────────────────────
# Threshold tuning (Stage F)
# ──────────────────────────────────────────────────────────────

def tune_threshold(
    features_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """Sweep thresholds on validation data, optimize for macro F_0.5.

    features_df must have columns: s1_id, cand_id, match_prob, (optionally label).

    Returns: (best_threshold, best_f05)
    """
    print("[Stage F] Tuning threshold for F_0.5...")

    if thresholds is None:
        thresholds = [i / 100 for i in range(30, 96)]

    best_threshold = 0.5
    best_f05 = 0.0

    for t in thresholds:
        predictions = apply_threshold(features_df, threshold=t)
        f05, prec, rec = compute_macro_f05(predictions, ground_truth)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t

    # Fine-grained search around best
    fine_thresholds = [best_threshold + i / 1000 for i in range(-50, 51)]
    fine_thresholds = [t for t in fine_thresholds if 0 < t < 1]
    for t in fine_thresholds:
        predictions = apply_threshold(features_df, threshold=t)
        f05, prec, rec = compute_macro_f05(predictions, ground_truth)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t

    # Print results at best threshold
    predictions = apply_threshold(features_df, threshold=best_threshold)
    f05, prec, rec = compute_macro_f05(predictions, ground_truth)
    print(f"  Best threshold: {best_threshold:.3f}")
    print(f"  F_0.5: {f05:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

    return best_threshold, best_f05


def apply_threshold(
    features_df: pd.DataFrame,
    threshold: float = MATCH_THRESHOLD,
) -> Dict[str, Set[str]]:
    """Apply threshold to match_prob and return predictions dict.

    Returns: dict mapping S1 entity_id → set of matched entity_ids
    """
    matches = features_df[features_df["match_prob"] >= threshold]
    predictions: Dict[str, Set[str]] = {}

    for _, row in matches.iterrows():
        s1_id = row["s1_id"]
        cand_id = row["cand_id"]
        if s1_id not in predictions:
            predictions[s1_id] = set()
        predictions[s1_id].add(cand_id)

    return predictions


# ──────────────────────────────────────────────────────────────
# Conflict resolution (injective constraint)
# ──────────────────────────────────────────────────────────────

def resolve_conflicts(
    features_df: pd.DataFrame,
    threshold: float,
    enforce_injective: bool = True,
) -> Dict[str, Set[str]]:
    """Apply threshold and optionally enforce the injective constraint:
    each S2/S3 record belongs to at most one S1 entity.

    When a candidate exceeds threshold for multiple S1 entities, keep it
    only for the S1 entity with the highest match probability.
    """
    matches = features_df[features_df["match_prob"] >= threshold].copy()

    if enforce_injective and len(matches) > 0:
        # Sort by match_prob descending
        matches = matches.sort_values("match_prob", ascending=False)
        # Keep first occurrence of each cand_id (highest prob)
        matches = matches.drop_duplicates(subset=["cand_id"], keep="first")

    predictions: Dict[str, Set[str]] = {}
    for _, row in matches.iterrows():
        s1_id = row["s1_id"]
        cand_id = row["cand_id"]
        if s1_id not in predictions:
            predictions[s1_id] = set()
        predictions[s1_id].add(cand_id)

    return predictions

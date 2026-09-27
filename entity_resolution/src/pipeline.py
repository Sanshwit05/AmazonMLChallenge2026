"""
Main Entity Resolution Pipeline

Orchestrates all stages:
  A. EDA (optional, run separately)
  B. Preprocessing
  C. Blocking / Candidate Generation
  D. Feature Engineering
  E. Model Training
  F. Threshold Tuning & Conflict Resolution
  G. Output & Validation

Usage:
  uv run python -m src.pipeline --mode train    # train on training data
  uv run python -m src.pipeline --mode predict  # generate test predictions
  uv run python -m src.pipeline --mode full     # train + predict end-to-end
"""

import argparse
import gc
import os
import pickle
import sys
import time
from typing import Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_DIR, MATCHING_RESULTS, CANDIDATE_PAIRS,
    TFIDF_TOP_K, VAL_FRACTION, RANDOM_SEED,
)
from .preprocessing import preprocess_dataframe
from .blocking import generate_candidates
from .features import generate_features_for_pairs
from .model import (
    get_feature_columns,
    train_model_with_validation,
    predict_probabilities,
    tune_threshold,
    apply_threshold,
    resolve_conflicts,
    compute_macro_f05,
)
from .io_utils import (
    load_source_file,
    load_ground_truth,
    write_matching_results,
    write_candidate_pairs,
    assign_labels,
)


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.pkl")


def _sample_training_data(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt: Dict[str, Set[str]],
    max_s1: int = 50000,
) -> tuple:
    """Sample a subset of training data for tractable model training.

    Stratified: keep all S1 entities with matches (up to limit), fill rest
    with singletons. Also filter S2/S3 to only records referenced by
    sampled S1 entities (as ground truth matches or blocking candidates).
    """
    print(f"  Sampling training data: max {max_s1} S1 entities...")

    s1_ids = s1_df["entity_id"].values.tolist()

    # Split into entities with matches and singletons
    with_matches = [sid for sid in s1_ids if len(gt.get(sid, set())) > 0]
    singletons = [sid for sid in s1_ids if len(gt.get(sid, set())) == 0]

    np.random.seed(RANDOM_SEED)

    # Keep proportional mix
    total = len(s1_ids)
    match_rate = len(with_matches) / total
    n_match = min(int(max_s1 * match_rate), len(with_matches))
    n_single = min(max_s1 - n_match, len(singletons))

    sampled_match = list(np.random.choice(with_matches, size=n_match, replace=False))
    sampled_single = list(np.random.choice(singletons, size=n_single, replace=False))
    sampled_s1_ids = set(sampled_match + sampled_single)

    # Get all S2/S3 ids referenced in ground truth for sampled S1s
    referenced_ids: Set[str] = set()
    for sid in sampled_s1_ids:
        referenced_ids.update(gt.get(sid, set()))

    s1_sampled = s1_df[s1_df["entity_id"].isin(sampled_s1_ids)].reset_index(drop=True)

    # For S2/S3: keep all referenced records + a random sample for negatives
    s2_referenced = s2_df[s2_df["entity_id"].isin(referenced_ids)]
    s3_referenced = s3_df[s3_df["entity_id"].isin(referenced_ids)]

    # Add random negative samples from S2/S3
    s2_remaining = s2_df[~s2_df["entity_id"].isin(referenced_ids)]
    s3_remaining = s3_df[~s3_df["entity_id"].isin(referenced_ids)]
    n_neg_s2 = min(len(s2_remaining), max(len(s2_referenced) * 3, 50000))
    n_neg_s3 = min(len(s3_remaining), max(len(s3_referenced) * 3, 50000))

    s2_neg_sample = s2_remaining.sample(n=n_neg_s2, random_state=RANDOM_SEED)
    s3_neg_sample = s3_remaining.sample(n=n_neg_s3, random_state=RANDOM_SEED)

    s2_sampled = pd.concat([s2_referenced, s2_neg_sample], ignore_index=True)
    s3_sampled = pd.concat([s3_referenced, s3_neg_sample], ignore_index=True)

    # Filter GT to sampled S1 ids
    gt_sampled = {k: v for k, v in gt.items() if k in sampled_s1_ids}

    print(f"    Sampled S1: {len(s1_sampled)} ({n_match} with matches, {n_single} singletons)")
    print(f"    Sampled S2: {len(s2_sampled)} ({len(s2_referenced)} referenced)")
    print(f"    Sampled S3: {len(s3_sampled)} ({len(s3_referenced)} referenced)")

    return s1_sampled, s2_sampled, s3_sampled, gt_sampled


def _split_train_val(
    s1_df: pd.DataFrame,
    gt: Dict[str, Set[str]],
    val_fraction: float = VAL_FRACTION,
) -> tuple:
    """Split S1 entities into train/val groups (no entity leakage)."""
    s1_ids = s1_df["entity_id"].values.tolist()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(s1_ids)

    n_val = int(len(s1_ids) * val_fraction)
    val_ids = set(s1_ids[:n_val])
    train_ids = set(s1_ids[n_val:])

    return train_ids, val_ids


# ══════════════════════════════════════════════════════════════
# TRAIN MODE
# ══════════════════════════════════════════════════════════════

def run_training(max_s1: int = 50000):
    """Full training pipeline: load data, preprocess, block, featurize,
    train, tune threshold, evaluate on validation."""
    start_time = time.time()

    # ── Load data ──
    print("\n" + "=" * 60)
    print("STAGE A: Loading training data")
    print("=" * 60)
    s1_raw = load_source_file(TRAIN_S1)
    s2_raw = load_source_file(TRAIN_S2)
    s3_raw = load_source_file(TRAIN_S3)
    gt = load_ground_truth(TRAIN_GT)

    # ── Sample for tractability ──
    print("\n" + "=" * 60)
    print("Sampling for tractable training")
    print("=" * 60)
    s1_raw, s2_raw, s3_raw, gt = _sample_training_data(
        s1_raw, s2_raw, s3_raw, gt, max_s1=max_s1
    )

    # ── Preprocess ──
    print("\n" + "=" * 60)
    print("STAGE B: Preprocessing")
    print("=" * 60)
    print("  Preprocessing S1...")
    s1 = preprocess_dataframe(s1_raw)
    print("  Preprocessing S2...")
    s2 = preprocess_dataframe(s2_raw)
    print("  Preprocessing S3...")
    s3 = preprocess_dataframe(s3_raw)

    del s1_raw, s2_raw, s3_raw
    gc.collect()

    # ── Train/Val split ──
    print("\n  Splitting train/val...")
    train_ids, val_ids = _split_train_val(s1, gt)
    print(f"  Train S1 entities: {len(train_ids)}")
    print(f"  Val S1 entities: {len(val_ids)}")

    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    # ── Blocking ──
    print("\n" + "=" * 60)
    print("STAGE C: Blocking (training set)")
    print("=" * 60)
    train_candidates = generate_candidates(s1_train, s2, s3)

    # Measure blocking recall on train set
    total_true = 0
    found_true = 0
    for s1_id in train_ids:
        true_matches = gt.get(s1_id, set())
        cand_set = train_candidates.get(s1_id, set())
        total_true += len(true_matches)
        found_true += len(true_matches & cand_set)
    block_recall = found_true / max(total_true, 1)
    print(f"  TRAIN Blocking recall: {block_recall:.4f} ({found_true}/{total_true})")

    # ── Blocking for validation ──
    print("\nSTAGE C: Blocking (validation set)")
    val_candidates = generate_candidates(s1_val, s2, s3)

    total_true_v = 0
    found_true_v = 0
    for s1_id in val_ids:
        true_matches = gt.get(s1_id, set())
        cand_set = val_candidates.get(s1_id, set())
        total_true_v += len(true_matches)
        found_true_v += len(true_matches & cand_set)
    block_recall_v = found_true_v / max(total_true_v, 1)
    print(f"  VAL Blocking recall: {block_recall_v:.4f} ({found_true_v}/{total_true_v})")

    # ── Feature engineering ──
    print("\n" + "=" * 60)
    print("STAGE D: Feature Engineering (training pairs)")
    print("=" * 60)
    candidates_pool = pd.concat([s2, s3], ignore_index=True)

    train_features = generate_features_for_pairs(s1_train, candidates_pool, train_candidates)
    train_features = assign_labels(train_features, gt)

    print("\nSTAGE D: Feature Engineering (validation pairs)")
    val_features = generate_features_for_pairs(s1_val, candidates_pool, val_candidates)
    val_features = assign_labels(val_features, gt)

    # ── Model training ──
    print("\n" + "=" * 60)
    print("STAGE E: Model Training")
    print("=" * 60)
    model, feat_cols = train_model_with_validation(
        train_features, val_features, model_save_path=MODEL_PATH
    )

    # ── Threshold tuning ──
    print("\n" + "=" * 60)
    print("STAGE F: Threshold Tuning")
    print("=" * 60)
    val_features = predict_probabilities(model, val_features, feat_cols)

    # Build val ground truth (including singletons for val S1 ids)
    val_gt = {s1_id: gt.get(s1_id, set()) for s1_id in val_ids}

    best_threshold, best_f05 = tune_threshold(val_features, val_gt)

    # Also evaluate with conflict resolution
    val_predictions = resolve_conflicts(val_features, best_threshold, enforce_injective=True)
    # Add singletons
    for s1_id in val_ids:
        if s1_id not in val_predictions:
            val_predictions[s1_id] = set()

    f05, prec, rec = compute_macro_f05(val_predictions, val_gt)
    print(f"\n  Final validation F_0.5 (with conflict resolution): {f05:.4f}")
    print(f"  Precision: {prec:.4f} | Recall: {rec:.4f}")

    # Save threshold
    threshold_path = os.path.join(MODEL_DIR, "threshold.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(threshold_path, "wb") as f:
        pickle.dump(best_threshold, f)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    return model, feat_cols, best_threshold


# ══════════════════════════════════════════════════════════════
# PREDICT MODE (Test set inference)
# ══════════════════════════════════════════════════════════════

def run_prediction(
    model=None,
    feat_cols=None,
    threshold=None,
    chunk_size: int = 50000,
):
    """Generate predictions for the test set.

    If model/threshold not provided, loads from saved files.
    Processes test S1 entities in chunks to manage memory.
    """
    start_time = time.time()

    # Load model if not provided
    if model is None:
        print("  Loading saved model...")
        with open(MODEL_PATH, "rb") as f:
            model, feat_cols = pickle.load(f)
        threshold_path = os.path.join(MODEL_DIR, "threshold.pkl")
        with open(threshold_path, "rb") as f:
            threshold = pickle.load(f)
        print(f"  Threshold: {threshold:.3f}")

    # ── Load test data ──
    print("\n" + "=" * 60)
    print("Loading test data")
    print("=" * 60)
    s1_raw = load_source_file(TEST_S1)
    s2_raw = load_source_file(TEST_S2)
    s3_raw = load_source_file(TEST_S3)

    all_s1_ids = s1_raw["entity_id"].values.tolist()

    # ── Preprocess ──
    print("\n" + "=" * 60)
    print("STAGE B: Preprocessing test data")
    print("=" * 60)
    print("  Preprocessing S1...")
    s1 = preprocess_dataframe(s1_raw)
    print("  Preprocessing S2...")
    s2 = preprocess_dataframe(s2_raw)
    print("  Preprocessing S3...")
    s3 = preprocess_dataframe(s3_raw)

    del s1_raw, s2_raw, s3_raw
    gc.collect()

    # Combine S2+S3 as candidate pool
    candidates_pool = pd.concat([s2, s3], ignore_index=True)

    # ── Process in chunks ──
    all_predictions: Dict[str, Set[str]] = {}
    all_candidates: Dict[str, Set[str]] = {}

    n_chunks = (len(s1) + chunk_size - 1) // chunk_size
    print(f"\nProcessing {len(s1)} S1 entities in {n_chunks} chunks of {chunk_size}")

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len(s1))
        s1_chunk = s1.iloc[chunk_start:chunk_end].reset_index(drop=True)

        print(f"\n{'─'*40}")
        print(f"Chunk {chunk_idx+1}/{n_chunks} ({len(s1_chunk)} S1 entities)")
        print(f"{'─'*40}")

        # Stage C: Blocking
        chunk_candidates = generate_candidates(s1_chunk, s2, s3)
        all_candidates.update(chunk_candidates)

        # Check if any candidates exist
        total_cands = sum(len(v) for v in chunk_candidates.values())
        if total_cands == 0:
            print("  No candidates found in this chunk, all singletons.")
            for s1_id in s1_chunk["entity_id"].values:
                all_predictions[s1_id] = set()
            continue

        # Stage D: Feature engineering
        chunk_features = generate_features_for_pairs(
            s1_chunk, candidates_pool, chunk_candidates
        )

        if len(chunk_features) == 0:
            for s1_id in s1_chunk["entity_id"].values:
                all_predictions[s1_id] = set()
            continue

        # Stage E: Predict
        chunk_features = predict_probabilities(model, chunk_features, feat_cols)

        # Stage F: Threshold + conflict resolution
        chunk_preds = resolve_conflicts(chunk_features, threshold, enforce_injective=True)
        all_predictions.update(chunk_preds)

        # Free memory
        del chunk_features, chunk_candidates
        gc.collect()

    # Ensure every S1 entity has an entry
    for s1_id in all_s1_ids:
        if s1_id not in all_predictions:
            all_predictions[s1_id] = set()
        if s1_id not in all_candidates:
            all_candidates[s1_id] = set()

    # ── Stage G: Write outputs ──
    print("\n" + "=" * 60)
    print("STAGE G: Writing output files")
    print("=" * 60)
    write_matching_results(all_predictions, all_s1_ids)
    write_candidate_pairs(all_candidates, all_s1_ids)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Prediction complete in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "full"],
        default="full",
        help="Pipeline mode: train, predict, or full (train+predict)",
    )
    parser.add_argument(
        "--max-s1",
        type=int,
        default=50000,
        help="Maximum S1 entities to sample for training (default: 50000)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Chunk size for test prediction (default: 50000)",
    )
    args = parser.parse_args()

    if args.mode in ("train", "full"):
        model, feat_cols, threshold = run_training(max_s1=args.max_s1)

    if args.mode in ("predict", "full"):
        if args.mode == "full":
            run_prediction(model=model, feat_cols=feat_cols, threshold=threshold,
                          chunk_size=args.chunk_size)
        else:
            run_prediction(chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()

"""
Stage G — Output & Validation

Handles reading/writing TSV files and generating the final submission files:
  - matching_results.tsv
  - candidate_pairs.tsv
"""

import os
from typing import Dict, List, Set

import pandas as pd

from .config import OUTPUT_DIR, MATCHING_RESULTS, CANDIDATE_PAIRS


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_source_file(path: str) -> pd.DataFrame:
    """Load a source TSV file with proper tab separator."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert "entity_id" in df.columns, f"Missing 'entity_id' column in {path}"
    assert "business_name" in df.columns, f"Missing 'business_name' column in {path}"
    assert "business_address" in df.columns, f"Missing 'business_address' column in {path}"
    assert "country" in df.columns, f"Missing 'country' column in {path}"
    print(f"  Loaded {path}: {len(df)} records")
    return df


def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Load ground truth TSV and return as dict: S1 id → set of matched ids.

    Empty matched_entity_ids means the entity is a singleton.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if matched and matched.strip():
            gt[s1_id] = set(matched.strip().split(","))
        else:
            gt[s1_id] = set()
    print(f"  Loaded ground truth: {len(gt)} S1 entities")
    singletons = sum(1 for v in gt.values() if len(v) == 0)
    with_matches = len(gt) - singletons
    print(f"    Singletons: {singletons} ({100*singletons/len(gt):.1f}%)")
    print(f"    With matches: {with_matches} ({100*with_matches/len(gt):.1f}%)")
    return gt


# ──────────────────────────────────────────────────────────────
# Output writing
# ──────────────────────────────────────────────────────────────

def write_matching_results(
    predictions: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_path: str = MATCHING_RESULTS,
) -> None:
    """Write matching_results.tsv with every S1 entity having exactly one row.

    Singletons get empty matched_entity_ids.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for s1_id in all_s1_ids:
        matched = predictions.get(s1_id, set())
        matched_str = ",".join(sorted(matched)) if matched else ""
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"  Written matching results: {output_path} ({len(df)} rows)")

    # Stats
    non_empty = sum(1 for r in rows if r["matched_entity_ids"])
    print(f"    Non-singleton predictions: {non_empty}")
    print(f"    Singleton predictions: {len(rows) - non_empty}")


def write_candidate_pairs(
    candidates: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_path: str = CANDIDATE_PAIRS,
) -> None:
    """Write candidate_pairs.tsv."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for s1_id in all_s1_ids:
        cands = candidates.get(s1_id, set())
        cands_str = ",".join(sorted(cands)) if cands else ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cands_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"  Written candidate pairs: {output_path} ({len(df)} rows)")


# ──────────────────────────────────────────────────────────────
# Label assignment (for training)
# ──────────────────────────────────────────────────────────────

def assign_labels(
    features_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
) -> pd.DataFrame:
    """Add binary 'label' column based on ground truth."""
    import numpy as np
    
    # Create a fast lookup set of true match pairs
    true_pairs = set()
    for s1_id, matches in ground_truth.items():
        for cand_id in matches:
            true_pairs.add((s1_id, cand_id))
            
    s1_ids = features_df["s1_id"].values
    cand_ids = features_df["cand_id"].values
    labels = [1 if (s, c) in true_pairs else 0 for s, c in zip(s1_ids, cand_ids)]
    
    features_df["label"] = np.array(labels, dtype=np.int8)
    pos = sum(labels)
    neg = len(labels) - pos
    print(f"  Labels assigned: {pos} positive, {neg} negative ({100*pos/max(len(labels),1):.2f}% positive rate)")
    return features_df

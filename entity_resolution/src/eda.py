"""
Stage A — Exploratory Data Analysis

Quick EDA on the training data to understand:
  - Schema validation and null rates
  - Distribution of match count per S1 entity (singleton rate, multi-match rate)
  - Whether any S2/S3 id appears as a match for multiple S1 entities (injective check)
  - Field completeness by source/country
  - String length and token-count distributions
  - Sample true-positive pairs for noise pattern inspection

Usage:
  uv run python -m src.eda
"""

import sys
import os
import pandas as pd
import numpy as np
from collections import Counter

from .config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT


def run_eda():
    print("=" * 70)
    print("ENTITY RESOLUTION -- EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # ── Load data ──
    print("\n1. Loading data (this may take a while for large files)...")
    # Only load a subset for EDA to keep things fast
    s1 = pd.read_csv(TRAIN_S1, sep="\t", dtype=str, keep_default_na=False, nrows=100000)
    s2 = pd.read_csv(TRAIN_S2, sep="\t", dtype=str, keep_default_na=False, nrows=100000)
    s3 = pd.read_csv(TRAIN_S3, sep="\t", dtype=str, keep_default_na=False, nrows=100000)
    gt = pd.read_csv(TRAIN_GT, sep="\t", dtype=str, keep_default_na=False)

    print(f"  S1 sample: {len(s1)} rows")
    print(f"  S2 sample: {len(s2)} rows")
    print(f"  S3 sample: {len(s3)} rows")
    print(f"  Ground truth: {len(gt)} rows")

    # ── Schema check ──
    print(f"\n2. Schema check:")
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(f"  {name} columns: {list(df.columns)}")
        for col in df.columns:
            empty_rate = (df[col] == "").mean()
            print(f"    {col}: {empty_rate*100:.1f}% empty")

    # ── Ground truth analysis ──
    print(f"\n3. Ground truth analysis:")
    gt_match_counts = []
    for _, row in gt.iterrows():
        matched = row.get("matched_entity_ids", "")
        if matched and matched.strip():
            ids = matched.strip().split(",")
            gt_match_counts.append(len(ids))
        else:
            gt_match_counts.append(0)

    gt["match_count"] = gt_match_counts
    print(f"  Total S1 entities in GT: {len(gt)}")
    print(f"  Singletons (0 matches): {(gt['match_count']==0).sum()} ({(gt['match_count']==0).mean()*100:.1f}%)")
    print(f"  1 match: {(gt['match_count']==1).sum()} ({(gt['match_count']==1).mean()*100:.1f}%)")
    print(f"  2 matches: {(gt['match_count']==2).sum()} ({(gt['match_count']==2).mean()*100:.1f}%)")
    print(f"  3+ matches: {(gt['match_count']>=3).sum()} ({(gt['match_count']>=3).mean()*100:.1f}%)")
    print(f"  Max matches: {gt['match_count'].max()}")
    print(f"  Mean matches (non-singleton): {gt[gt['match_count']>0]['match_count'].mean():.2f}")

    # Match count distribution
    print(f"\n  Match count distribution (top 10):")
    dist = gt["match_count"].value_counts().sort_index()
    for count, freq in list(dist.items())[:10]:
        print(f"    {count} matches: {freq} entities ({freq/len(gt)*100:.2f}%)")

    # ── Injective check: does any S2/S3 id appear for multiple S1 entities? ──
    print(f"\n4. Injective constraint check:")
    cand_to_s1 = Counter()
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if matched and matched.strip():
            for cid in matched.strip().split(","):
                cand_to_s1[cid] += 1

    multi_assigned = {k: v for k, v in cand_to_s1.items() if v > 1}
    print(f"  Total unique S2/S3 ids in GT: {len(cand_to_s1)}")
    print(f"  S2/S3 ids assigned to >1 S1 entity: {len(multi_assigned)}")
    if multi_assigned:
        print(f"  WARNING: Injective constraint VIOLATED -- some S2/S3 ids match multiple S1 entities")
        # Show a few examples
        for cid, count in list(multi_assigned.items())[:5]:
            print(f"    {cid}: matched to {count} S1 entities")
    else:
        print(f"  OK: Injective constraint holds -- each S2/S3 id matches at most one S1 entity")

    # ── S2/S3 split in matches ──
    print(f"\n5. Source distribution in matches:")
    s2_matches = sum(1 for cid in cand_to_s1 if cid.startswith("S2-"))
    s3_matches = sum(1 for cid in cand_to_s1 if cid.startswith("S3-"))
    print(f"  S2 ids in matches: {s2_matches}")
    print(f"  S3 ids in matches: {s3_matches}")

    # ── Country distribution ──
    print(f"\n6. Country distribution (from samples):")
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        country_dist = df["country"].value_counts()
        print(f"  {name}:")
        for country, count in country_dist.items():
            print(f"    {country}: {count} ({count/len(df)*100:.1f}%)")

    # ── Name/Address statistics ──
    print(f"\n7. Field statistics (from S1 sample):")
    s1["name_len"] = s1["business_name"].str.len()
    s1["addr_len"] = s1["business_address"].str.len()
    s1["name_tokens"] = s1["business_name"].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    s1["addr_tokens"] = s1["business_address"].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)

    print(f"  Name length: mean={s1['name_len'].mean():.0f}, median={s1['name_len'].median():.0f}, max={s1['name_len'].max()}")
    print(f"  Address length: mean={s1['addr_len'].mean():.0f}, median={s1['addr_len'].median():.0f}, max={s1['addr_len'].max()}")
    print(f"  Name tokens: mean={s1['name_tokens'].mean():.1f}, median={s1['name_tokens'].median():.1f}")
    print(f"  Address tokens: mean={s1['addr_tokens'].mean():.1f}, median={s1['addr_tokens'].median():.1f}")

    # ── Sample true-positive pairs ──
    print(f"\n8. Sample true-positive pairs (first 10 with matches):")
    # Load S1/S2/S3 lookups (from sample)
    s1_lookup = {row["entity_id"]: row for _, row in s1.iterrows()}
    s2_lookup = {row["entity_id"]: row for _, row in s2.iterrows()}
    s3_lookup = {row["entity_id"]: row for _, row in s3.iterrows()}
    all_lookup = {**s1_lookup, **s2_lookup, **s3_lookup}

    shown = 0
    for _, row in gt.iterrows():
        if shown >= 10:
            break
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if not matched or not matched.strip():
            continue
        if s1_id not in s1_lookup:
            continue

        s1_rec = s1_lookup[s1_id]
        match_ids = matched.strip().split(",")
        # Only show if we have the candidate in our sample
        for mid in match_ids[:2]:
            cand_rec = all_lookup.get(mid)
            if cand_rec is not None:
                print(f"\n  --- Pair {shown+1} ---")
                print(f"  S1 [{s1_id}]: name='{s1_rec['business_name']}' | addr='{s1_rec['business_address']}' | country={s1_rec['country']}")
                print(f"  Cand [{mid}]:  name='{cand_rec['business_name']}' | addr='{cand_rec['business_address']}' | country={cand_rec['country']}")
                shown += 1
                break

    print(f"\n{'='*70}")
    print("EDA complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_eda()

"""
Stage D — Feature Engineering (Optimized for scale)

Computes pairwise features for each (S1 entity, candidate) pair.
Uses dict-based lookups and batch processing for speed.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# String similarity helpers
# ──────────────────────────────────────────────────────────────

def _lev_ratio(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return Levenshtein.normalized_similarity(s1, s2)


def _jw_sim(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return JaroWinkler.normalized_similarity(s1, s2)


def _token_set(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.token_set_ratio(s1, s2) / 100.0


def _token_sort(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.token_sort_ratio(s1, s2) / 100.0


def _partial(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.partial_ratio(s1, s2) / 100.0


def _jaccard(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 and not set2:
        return 1.0
    inter = set1 & set2
    union = set1 | set2
    return len(inter) / len(union) if union else 0.0


def _containment(s1: str, s2: str) -> float:
    if not s1:
        return 1.0 if not s2 else 0.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1:
        return 1.0
    return len(set1 & set2) / len(set1)


def _char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    s1, s2 = s1.replace(" ", ""), s2.replace(" ", "")
    n1, n2 = len(s1), len(s2)
    ngrams1 = set(s1[i:i+n] for i in range(n1-n+1)) if n1 >= n else set([s1])
    ngrams2 = set(s2[i:i+n] for i in range(n2-n+1)) if n2 >= n else set([s2])
    if not ngrams1 and not ngrams2:
        return 1.0
    inter = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(inter) / len(union) if union else 0.0


_NUMBERS_RE = re.compile(r"\d+")

def _number_overlap(s1: str, s2: str) -> float:
    nums1 = set(_NUMBERS_RE.findall(s1)) if s1 else set()
    nums2 = set(_NUMBERS_RE.findall(s2)) if s2 else set()
    if not nums1 and not nums2:
        return 1.0
    if not nums1 or not nums2:
        return 0.0
    return len(nums1 & nums2) / len(nums1 | nums2)


def _acronym_match(name1: str, name2: str) -> float:
    if not name1 or not name2:
        return 0.0
    tokens1 = name1.split()
    tokens2 = name2.split()
    acr1 = "".join(t[0] for t in tokens1 if t)
    acr2 = "".join(t[0] for t in tokens2 if t)
    if acr1 and (acr1 == name2.replace(" ", "") or acr1 == acr2):
        return 1.0
    if acr2 and (acr2 == name1.replace(" ", "")):
        return 1.0
    return 0.0


# ──────────────────────────────────────────────────────────────
# Compute all features for one pair (dict-based, no pandas overhead)
# ──────────────────────────────────────────────────────────────

def _compute_pair(s1: dict, cand: dict) -> list:
    """Compute features for one (S1, candidate) pair. Returns a flat list of floats."""

    n1 = s1["name_normalized"]
    n2 = cand["name_normalized"]
    n1_ns = s1["name_no_suffix"]
    n2_ns = cand["name_no_suffix"]
    a1 = s1["addr_clean"]
    a2 = cand["addr_clean"]
    pc1 = s1["postal_code"]
    pc2 = cand["postal_code"]
    sn1 = s1["street_number"]
    sn2 = cand["street_number"]
    ar1 = s1["addr_residual"]
    ar2 = cand["addr_residual"]
    c1 = s1["country_clean"]
    c2 = cand["country_clean"]
    comb1 = s1["combined_text"]
    comb2 = cand["combined_text"]

    return [
        # ── Name features (14) ──
        float(n1 == n2 and n1 != ""),               # name_exact_match
        _lev_ratio(n1, n2),                          # name_levenshtein
        _jw_sim(n1, n2),                             # name_jaro_winkler
        _token_set(n1, n2),                          # name_token_set
        _token_sort(n1, n2),                         # name_token_sort
        _partial(n1, n2),                            # name_partial
        _jaccard(n1, n2),                            # name_jaccard
        _containment(n1, n2),                        # name_containment_s1
        _containment(n2, n1),                        # name_containment_cand
        _lev_ratio(n1_ns, n2_ns),                    # name_ns_levenshtein
        _jw_sim(n1_ns, n2_ns),                       # name_ns_jaro_winkler
        _token_set(n1_ns, n2_ns),                    # name_ns_token_set
        _jaccard(n1_ns, n2_ns),                      # name_ns_jaccard
        _acronym_match(n1_ns, n2_ns),                # name_acronym_match
        # ── Name meta (3) ──
        abs(len(n1) - len(n2)),                      # name_len_diff
        abs(len(n1.split()) - len(n2.split())),      # name_token_count_diff
        min(len(n1), len(n2)) / max(len(n1), len(n2), 1),  # name_len_ratio
        # ── Address features (10) ──
        float(a1 == a2 and a1 != ""),                # addr_exact_match
        _lev_ratio(a1, a2),                          # addr_levenshtein
        _jw_sim(a1, a2),                             # addr_jaro_winkler
        _token_set(a1, a2),                          # addr_token_set
        _token_sort(a1, a2),                         # addr_token_sort
        _jaccard(a1, a2),                            # addr_jaccard
        _partial(a1, a2),                            # addr_partial
        _containment(a1, a2),                        # addr_containment_s1
        _containment(a2, a1),                        # addr_containment_cand
        _number_overlap(a1, a2),                     # addr_number_overlap
        # ── Address component features (5) ──
        float(pc1 == pc2 and pc1 != ""),             # postal_code_match
        _lev_ratio(pc1, pc2) if pc1 and pc2 else 0.0,  # postal_code_partial
        float(sn1 == sn2 and sn1 != ""),             # street_number_match
        float(not pc1 or not pc2),                   # postal_code_missing
        float(not sn1 or not sn2),                   # street_number_missing
        # ── Address residual (2) ──
        _token_set(ar1, ar2),                        # addr_residual_token_set
        _jaccard(ar1, ar2),                          # addr_residual_jaccard
        # ── Address meta (3) ──
        abs(len(a1) - len(a2)),                      # addr_len_diff
        float(not a1 and not a2),                    # addr_both_empty
        float(bool(a1) != bool(a2)),                 # addr_one_empty
        # ── Country (2) ──
        float(c1 == c2 and c1 != ""),                # country_match
        float(c1 != c2 and c1 != "" and c2 != ""),   # country_mismatch
        # ── Combined (4) ──
        _token_set(comb1, comb2),                    # combined_token_set
        _jaccard(comb1, comb2),                      # combined_jaccard
        _char_ngram_jaccard(n1, n2),                 # name_char_ngram_jaccard
        _char_ngram_jaccard(a1, a2),                 # addr_char_ngram_jaccard
    ]


# Feature column names (must match order in _compute_pair)
FEATURE_COLUMNS = [
    "name_exact_match", "name_levenshtein", "name_jaro_winkler",
    "name_token_set", "name_token_sort", "name_partial",
    "name_jaccard", "name_containment_s1", "name_containment_cand",
    "name_ns_levenshtein", "name_ns_jaro_winkler", "name_ns_token_set",
    "name_ns_jaccard", "name_acronym_match",
    "name_len_diff", "name_token_count_diff", "name_len_ratio",
    "addr_exact_match", "addr_levenshtein", "addr_jaro_winkler",
    "addr_token_set", "addr_token_sort", "addr_jaccard", "addr_partial",
    "addr_containment_s1", "addr_containment_cand", "addr_number_overlap",
    "postal_code_match", "postal_code_partial", "street_number_match",
    "postal_code_missing", "street_number_missing",
    "addr_residual_token_set", "addr_residual_jaccard",
    "addr_len_diff", "addr_both_empty", "addr_one_empty",
    "country_match", "country_mismatch",
    "combined_token_set", "combined_jaccard",
    "name_char_ngram_jaccard", "addr_char_ngram_jaccard",
]


def get_feature_columns(features_df: pd.DataFrame = None) -> List[str]:
    """Return the list of feature column names."""
    return FEATURE_COLUMNS


# ──────────────────────────────────────────────────────────────
# Best alias match
# ──────────────────────────────────────────────────────────────

def _best_alias_score(aliases: list, cand_name: str) -> float:
    if not aliases or not cand_name:
        return 0.0
    best = 0.0
    for alias in aliases:
        if alias:
            score = _token_set(alias.lower(), cand_name)
            if score > best:
                best = score
    return best


# ──────────────────────────────────────────────────────────────
# Batch feature generation
# ──────────────────────────────────────────────────────────────

def generate_features_for_pairs(
    s1_df: pd.DataFrame,
    candidates_pool: pd.DataFrame,
    candidate_dict: Dict[str, Set[str]],
) -> pd.DataFrame:
    """Generate features for all (S1, candidate) pairs from the blocking stage.

    Uses dict-based lookups for fast access, avoids iterrows on the data frames.
    """
    print("[Stage D] Generating features for candidate pairs...")

    # Build lookup dicts: entity_id → dict of preprocessed fields
    # Only keep the columns we actually need
    keep_cols = [
        "entity_id", "name_normalized", "name_no_suffix", "name_clean",
        "addr_clean", "postal_code", "street_number", "addr_residual",
        "country_clean", "combined_text", "aliases",
    ]

    s1_lookup = {}
    for _, row in s1_df[keep_cols].iterrows():
        s1_lookup[row["entity_id"]] = {
            "name_normalized": row["name_normalized"] or "",
            "name_no_suffix": row["name_no_suffix"] or "",
            "name_clean": row["name_clean"] or "",
            "addr_clean": row["addr_clean"] or "",
            "postal_code": row["postal_code"] or "",
            "street_number": row["street_number"] or "",
            "addr_residual": row["addr_residual"] or "",
            "country_clean": row["country_clean"] or "",
            "combined_text": row["combined_text"] or "",
            "aliases": row["aliases"] if isinstance(row["aliases"], list) else [""],
        }

    # For candidates, we only need the subset referenced in candidate_dict
    all_cand_ids = set()
    for cands in candidate_dict.values():
        all_cand_ids.update(cands)

    cand_lookup = {}
    cand_subset = candidates_pool[candidates_pool["entity_id"].isin(all_cand_ids)]
    for _, row in cand_subset[keep_cols].iterrows():
        cand_lookup[row["entity_id"]] = {
            "name_normalized": row["name_normalized"] or "",
            "name_no_suffix": row["name_no_suffix"] or "",
            "name_clean": row["name_clean"] or "",
            "addr_clean": row["addr_clean"] or "",
            "postal_code": row["postal_code"] or "",
            "street_number": row["street_number"] or "",
            "addr_residual": row["addr_residual"] or "",
            "country_clean": row["country_clean"] or "",
            "combined_text": row["combined_text"] or "",
            "aliases": row["aliases"] if isinstance(row["aliases"], list) else [""],
        }

    total_pairs = sum(len(v) for v in candidate_dict.values())
    print(f"  Total pairs to featurize: {total_pairs}")

    if total_pairs == 0:
        # Return empty DataFrame with correct columns
        cols = ["s1_id", "cand_id"] + FEATURE_COLUMNS + ["best_alias_score"]
        return pd.DataFrame(columns=cols)

    # Pre-allocate numpy arrays to prevent MemoryError with massive lists
    s1_id_arr = np.empty(total_pairs, dtype=object)
    cand_id_arr = np.empty(total_pairs, dtype=object)
    feature_matrix = np.empty((total_pairs, len(FEATURE_COLUMNS)), dtype=np.float32)
    alias_scores_arr = np.empty(total_pairs, dtype=np.float32)

    pbar = tqdm(total=total_pairs, desc="  Computing features", mininterval=5.0)
    
    idx = 0
    for s1_id, cand_ids in candidate_dict.items():
        if not cand_ids:
            continue

        s1_data = s1_lookup.get(s1_id)
        if s1_data is None:
            pbar.update(len(cand_ids))
            continue

        s1_aliases = s1_data["aliases"]

        for cand_id in cand_ids:
            cand_data = cand_lookup.get(cand_id)
            if cand_data is None:
                pbar.update(1)
                continue

            feats = _compute_pair(s1_data, cand_data)
            alias_score = _best_alias_score(s1_aliases, cand_data["name_clean"])

            s1_id_arr[idx] = s1_id
            cand_id_arr[idx] = cand_id
            feature_matrix[idx, :] = feats
            alias_scores_arr[idx] = alias_score
            
            idx += 1
            pbar.update(1)

    pbar.close()
    
    # Slice arrays in case we skipped some missing cand_ids
    if idx < total_pairs:
        s1_id_arr = s1_id_arr[:idx]
        cand_id_arr = cand_id_arr[:idx]
        feature_matrix = feature_matrix[:idx]
        alias_scores_arr = alias_scores_arr[:idx]

    # Build DataFrame from pre-allocated arrays
    features_df = pd.DataFrame(feature_matrix, columns=FEATURE_COLUMNS)
    features_df.insert(0, "s1_id", s1_id_arr)
    features_df.insert(1, "cand_id", cand_id_arr)
    features_df["best_alias_score"] = alias_scores_arr

    print(f"  Feature matrix shape: {features_df.shape}")

    return features_df

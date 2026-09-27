"""
Stage C — Blocking / Candidate Generation (Optimized for scale)

Multiple independent blocking strategies, union-ed for maximum recall:
  1. Exact/near-exact keys: name prefix, postal code, phonetic codes
  2. Character n-gram TF-IDF with sparse cosine similarity (batched)

Outputs the final candidate set that gets written to candidate_pairs.tsv
and fed into the matching model.
"""

import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .config import TFIDF_NGRAM_RANGE, TFIDF_TOP_K, NAME_PREFIX_LEN


# ──────────────────────────────────────────────────────────────
# Phonetic coding (simple, dependency-free)
# ──────────────────────────────────────────────────────────────

def _simple_phonetic(word: str) -> str:
    """A simple phonetic hash: keeps first letter, strips vowels, dedupes
    consecutive consonants. Zero-dependency approximation.
    """
    if not word:
        return ""
    word = word.lower().strip()
    if not word:
        return ""
    first = word[0]
    rest = re.sub(r"[aeiouyhw]", "", word[1:])
    deduped = first
    for ch in rest:
        if ch != deduped[-1]:
            deduped += ch
    return deduped[:6]


# ──────────────────────────────────────────────────────────────
# Blocking Strategy 1: Inverted-Index Keys (vectorized)
# ──────────────────────────────────────────────────────────────

def build_inverted_index_blocks(
    s1_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> Dict[str, Set[str]]:
    """Build candidate sets using exact/near-exact key matching.

    Uses vectorized operations on pandas columns for speed.
    """
    print("  [Blocking] Building inverted index blocks...")

    cand_ids = candidates_df["entity_id"].values
    cand_name_ns = candidates_df["name_no_suffix"].values
    cand_postal = candidates_df["postal_code"].values
    cand_name_clean = candidates_df["name_clean"].values

    s1_ids = s1_df["entity_id"].values
    s1_name_ns = s1_df["name_no_suffix"].values
    s1_postal = s1_df["postal_code"].values
    s1_name_clean = s1_df["name_clean"].values

    # Build indexes from candidates (dict-of-lists for fast lookup)
    prefix_index: Dict[str, List[int]] = defaultdict(list)
    postal_index: Dict[str, List[int]] = defaultdict(list)
    phonetic_index: Dict[str, List[int]] = defaultdict(list)
    token_pair_index: Dict[str, List[int]] = defaultdict(list)

    print("    Indexing candidates...")
    for i in range(len(cand_ids)):
        # Name prefix
        name_ns = cand_name_ns[i] if cand_name_ns[i] else ""
        prefix = name_ns[:NAME_PREFIX_LEN]
        if prefix and len(prefix) >= 3:
            prefix_index[prefix].append(i)

        # Postal code
        postal = cand_postal[i] if cand_postal[i] else ""
        if postal and len(postal) >= 5:
            postal_index[postal].append(i)

        # Phonetic code of first token
        name_c = cand_name_clean[i] if cand_name_clean[i] else ""
        if name_c:
            first_space = name_c.find(" ")
            first_token = name_c[:first_space] if first_space > 0 else name_c
            phon = _simple_phonetic(first_token)
            if phon:
                phonetic_index[phon].append(i)

            # First two tokens as a pair
            if first_space > 0:
                rest = name_c[first_space + 1:]
                second_space = rest.find(" ")
                second_token = rest[:second_space] if second_space > 0 else rest
                if second_token:
                    pair_key = first_token + "_" + second_token
                    token_pair_index[pair_key].append(i)

    print(f"    Index sizes: prefix={len(prefix_index)}, postal={len(postal_index)}, "
          f"phonetic={len(phonetic_index)}, token_pair={len(token_pair_index)}")

    # Prune overly generic keys to prevent memory explosion (block size limit)
    MAX_CANDIDATES_PER_KEY = 5000
    for idx_dict in (prefix_index, postal_index, phonetic_index, token_pair_index):
        keys_to_delete = [k for k, v in idx_dict.items() if len(v) > MAX_CANDIDATES_PER_KEY]
        for k in keys_to_delete:
            del idx_dict[k]

    # Query for each S1 entity
    result: Dict[str, Set[str]] = {}

    print("    Querying inverted index...")
    for i in range(len(s1_ids)):
        candidates_indices: Set[int] = set()

        # Name prefix
        name_ns = s1_name_ns[i] if s1_name_ns[i] else ""
        prefix = name_ns[:NAME_PREFIX_LEN]
        if prefix and len(prefix) >= 3:
            indices = prefix_index.get(prefix)
            if indices:
                candidates_indices.update(indices)

        # Postal code
        postal = s1_postal[i] if s1_postal[i] else ""
        if postal and len(postal) >= 5:
            indices = postal_index.get(postal)
            if indices:
                candidates_indices.update(indices)

        # Phonetic
        name_c = s1_name_clean[i] if s1_name_clean[i] else ""
        if name_c:
            first_space = name_c.find(" ")
            first_token = name_c[:first_space] if first_space > 0 else name_c
            phon = _simple_phonetic(first_token)
            if phon:
                indices = phonetic_index.get(phon)
                if indices:
                    candidates_indices.update(indices)

            # Token pair
            if first_space > 0:
                rest = name_c[first_space + 1:]
                second_space = rest.find(" ")
                second_token = rest[:second_space] if second_space > 0 else rest
                if second_token:
                    pair_key = first_token + "_" + second_token
                    indices = token_pair_index.get(pair_key)
                    if indices:
                        candidates_indices.update(indices)

        # Convert indices to entity IDs
        result[s1_ids[i]] = {cand_ids[idx] for idx in candidates_indices}

    return result


# ──────────────────────────────────────────────────────────────
# Blocking Strategy 2: TF-IDF character n-gram cosine (batched)
# ──────────────────────────────────────────────────────────────

def build_tfidf_blocks(
    s1_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    top_k: int = TFIDF_TOP_K,
    ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
    batch_size: int = 100,
) -> Dict[str, Set[str]]:
    """TF-IDF blocking on combined name+address text with character n-grams.

    Fits TF-IDF on the candidate pool, then queries S1 entities in batches
    to find top-K nearest neighbors by cosine similarity.
    """
    print("  [Blocking] Building TF-IDF character n-gram blocks...")

    cand_texts = candidates_df["combined_text"].fillna("").values.tolist()
    cand_ids = candidates_df["entity_id"].values

    # Fit TF-IDF on candidates
    print("    Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=150000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    cand_matrix = vectorizer.fit_transform(cand_texts)
    print(f"    TF-IDF matrix: {cand_matrix.shape}")

    # Transpose for efficient batch dot product
    cand_matrix_T = cand_matrix.T.tocsc()

    s1_texts = s1_df["combined_text"].fillna("").values.tolist()
    s1_ids = s1_df["entity_id"].values

    result: Dict[str, Set[str]] = {}

    # Process in batches
    n_s1 = len(s1_texts)
    n_batches = (n_s1 + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="  TF-IDF blocking", mininterval=5.0):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_s1)
        batch_texts = s1_texts[start:end]
        batch_ids = s1_ids[start:end]

        # Transform S1 batch
        s1_batch_matrix = vectorizer.transform(batch_texts)

        # Cosine similarity via sparse dot product
        sim_matrix = s1_batch_matrix.dot(cand_matrix_T)

        # For each S1 in batch, get top-K
        for i in range(len(batch_ids)):
            row = sim_matrix.getrow(i)
            if row.nnz == 0:
                result[batch_ids[i]] = set()
                continue

            data = row.data
            indices = row.indices

            if len(data) <= top_k:
                # Take all non-trivial
                mask = data > 0.01
                selected_indices = indices[mask]
            else:
                # Partial sort for top-K
                if top_k < len(data):
                    top_k_positions = np.argpartition(data, -top_k)[-top_k:]
                    selected_indices = indices[top_k_positions]
                    selected_data = data[top_k_positions]
                    # Filter very low scores
                    mask = selected_data > 0.01
                    selected_indices = selected_indices[mask]
                else:
                    selected_indices = indices[data > 0.01]

            result[batch_ids[i]] = {cand_ids[idx] for idx in selected_indices}

    return result


# ──────────────────────────────────────────────────────────────
# Union blocking
# ──────────────────────────────────────────────────────────────

def union_blocks(*block_dicts: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Merge multiple blocking results by union of candidate sets per S1 entity."""
    merged: Dict[str, Set[str]] = {}
    for bd in block_dicts:
        for s1_id, cands in bd.items():
            if s1_id in merged:
                merged[s1_id].update(cands)
            else:
                merged[s1_id] = set(cands)
    return merged


# ──────────────────────────────────────────────────────────────
# Main blocking function
# ──────────────────────────────────────────────────────────────

def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    tfidf_top_k: int = TFIDF_TOP_K,
) -> Dict[str, Set[str]]:
    """Run all blocking strategies and return the union of candidates.

    Processes S2 and S3 together as one candidate pool.
    """
    print("[Stage C] Generating candidates via blocking...")

    # Combine S2 and S3 into a single candidate pool
    candidates_df = pd.concat([s2_df, s3_df], ignore_index=True)
    print(f"  Total candidate pool: {len(candidates_df)} records (S2+S3)")

    # Strategy 1: Inverted index keys
    inv_blocks = build_inverted_index_blocks(s1_df, candidates_df)

    # Strategy 2: TF-IDF character n-gram blocking
    tfidf_blocks = build_tfidf_blocks(
        s1_df, candidates_df, top_k=tfidf_top_k
    )

    # Union all strategies
    final_candidates = union_blocks(inv_blocks, tfidf_blocks)

    # Ensure every S1 entity has an entry (possibly empty)
    for s1_id in s1_df["entity_id"].values:
        if s1_id not in final_candidates:
            final_candidates[s1_id] = set()

    # Stats
    total_cands = sum(len(v) for v in final_candidates.values())
    non_empty = sum(1 for v in final_candidates.values() if len(v) > 0)
    avg_cands = total_cands / max(len(final_candidates), 1)
    print(f"  Blocking complete: {len(final_candidates)} S1 entities")
    print(f"  Total candidate pairs: {total_cands}")
    print(f"  S1 entities with candidates: {non_empty}")
    print(f"  Average candidates per S1: {avg_cands:.1f}")

    return final_candidates

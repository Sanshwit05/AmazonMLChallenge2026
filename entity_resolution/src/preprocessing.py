"""
Stage B — Preprocessing / Normalization

Applies identical transformations to all three sources:
  - Unicode NFKC normalization + ASCII folding
  - Lowercase, &→and, punctuation stripping
  - Legal suffix normalization
  - Address abbreviation expansion
  - Postal code extraction (US 5-digit, India 6-digit, France 5-digit)
  - DBA/alias splitting
"""

import re
import unicodedata
from typing import List, Tuple

import pandas as pd

from .config import LEGAL_SUFFIX_MAP, ADDRESS_ABBREV_MAP


# ──────────────────────────────────────────────────────────────
# Unicode helpers
# ──────────────────────────────────────────────────────────────

def unicode_normalize(text: str) -> str:
    """NFKC normalize and strip control characters."""
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Remove control chars but keep whitespace
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\t", "\n", " "))
    return text


def ascii_fold(text: str) -> str:
    """Strip accents/diacritics to produce an ASCII-safe variant.
    e.g. 'Société Générale' → 'Societe Generale'
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


# ──────────────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────────────

_AMP_RE = re.compile(r"\s*&\s*")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, replace & with 'and', strip punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = _AMP_RE.sub(" and ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────
# Legal suffix handling
# ──────────────────────────────────────────────────────────────

# Pattern to detect legal suffixes at the end of a name
_LEGAL_SUFFIXES_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(LEGAL_SUFFIX_MAP.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def normalize_legal_suffix(name: str) -> Tuple[str, str]:
    """Return (normalized_name_with_suffix_expanded, name_without_suffix).

    The suffix-stripped name is useful for blocking; the expanded one for features.
    """
    if not name:
        return ("", "")

    clean = clean_text(name)
    tokens = clean.split()
    stripped_tokens = []
    suffix_found = ""

    for t in tokens:
        mapped = LEGAL_SUFFIX_MAP.get(t, None)
        if mapped is not None:
            suffix_found = mapped
            if mapped:  # non-empty replacement
                stripped_tokens.append(mapped)
            # else: drop it (like LLC)
        else:
            stripped_tokens.append(t)

    normalized = " ".join(stripped_tokens)
    # suffix-stripped: remove the suffix entirely
    no_suffix = " ".join(t for t in tokens if LEGAL_SUFFIX_MAP.get(t) is None)

    return (normalized, no_suffix)


# ──────────────────────────────────────────────────────────────
# Address normalization
# ──────────────────────────────────────────────────────────────

def expand_address_abbrevs(address: str) -> str:
    """Expand common address abbreviations."""
    if not address:
        return ""
    tokens = address.split()
    expanded = []
    for t in tokens:
        mapped = ADDRESS_ABBREV_MAP.get(t.lower().rstrip("."), None)
        if mapped:
            expanded.append(mapped)
        else:
            expanded.append(t.lower())
    return " ".join(expanded)


# Postal code patterns (broad, country-agnostic digit patterns)
_POSTAL_CODE_RE = re.compile(
    r"\b(\d{5,6})\b"  # 5-digit (US/France) or 6-digit (India PIN)
)

_STREET_NUMBER_RE = re.compile(
    r"^(\d+[a-zA-Z]?(?:/\d+)?)\b"  # leading street number like "123", "14A", "19 1/2"
)


def extract_address_components(address: str) -> dict:
    """Extract postal code, street number, and residual text from an address."""
    if not address:
        return {"postal_code": "", "street_number": "", "address_residual": ""}

    clean = clean_text(address)

    # Extract postal code
    postal_codes = _POSTAL_CODE_RE.findall(clean)
    postal_code = postal_codes[-1] if postal_codes else ""  # take last (usually at end)

    # Extract leading street number
    street_match = _STREET_NUMBER_RE.match(clean)
    street_number = street_match.group(1) if street_match else ""

    # Residual: remove postal code and street number
    residual = clean
    if postal_code:
        residual = residual.replace(postal_code, "", 1)
    if street_number:
        residual = residual.replace(street_number, "", 1)
    residual = _MULTI_SPACE_RE.sub(" ", residual).strip()

    return {
        "postal_code": postal_code,
        "street_number": street_number,
        "address_residual": residual,
    }


def normalize_address(address: str) -> str:
    """Full address normalization pipeline."""
    if not isinstance(address, str) or not address:
        return ""
    text = unicode_normalize(address)
    text = ascii_fold(text)
    text = clean_text(text)
    text = expand_address_abbrevs(text)
    return text


# ──────────────────────────────────────────────────────────────
# DBA / alias splitting
# ──────────────────────────────────────────────────────────────

_DBA_SPLIT_RE = re.compile(
    r"\s*(?:/|\\|\bdba\b|\bt/a\b|\bformerly\b|\baka\b|\ba\.?k\.?a\.?\b|\bd/b/a\b)\s*",
    re.IGNORECASE,
)


def split_aliases(name: str) -> List[str]:
    """Split a business name on DBA/trade-name separators into alias list."""
    if not name:
        return [""]
    parts = _DBA_SPLIT_RE.split(name)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [""]


# ──────────────────────────────────────────────────────────────
# Full preprocessing for a source DataFrame
# ──────────────────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to a source dataframe.

    Input columns: entity_id, business_name, business_address, country
    Output: original columns + normalized columns.
    """
    df = df.copy()

    # Fill NaNs
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")

    # Unicode normalize raw fields
    df["name_raw"] = df["business_name"].apply(unicode_normalize)
    df["addr_raw"] = df["business_address"].apply(unicode_normalize)

    # ASCII-folded variants (for French/transliteration matching)
    df["name_ascii"] = df["name_raw"].apply(ascii_fold)
    df["addr_ascii"] = df["addr_raw"].apply(ascii_fold)

    # Cleaned (lowercase, no punct)
    df["name_clean"] = df["name_ascii"].apply(clean_text)
    df["addr_clean"] = df["addr_ascii"].apply(normalize_address)

    # Legal suffix normalization
    suffix_results = df["name_ascii"].apply(normalize_legal_suffix)
    df["name_normalized"] = suffix_results.apply(lambda x: x[0])
    df["name_no_suffix"] = suffix_results.apply(lambda x: x[1])

    # Address components
    addr_components = df["addr_clean"].apply(extract_address_components)
    df["postal_code"] = addr_components.apply(lambda x: x["postal_code"])
    df["street_number"] = addr_components.apply(lambda x: x["street_number"])
    df["addr_residual"] = addr_components.apply(lambda x: x["address_residual"])

    # Aliases (from original name)
    df["aliases"] = df["name_ascii"].apply(split_aliases)

    # Name prefix for blocking (first N chars of cleaned name, no suffix)
    df["name_prefix"] = df["name_no_suffix"].str[:5]

    # Combined text for embedding: name + address
    df["combined_text"] = df["name_clean"] + " " + df["addr_clean"]

    # Country lowered
    df["country_clean"] = df["country"].str.lower().str.strip()

    return df

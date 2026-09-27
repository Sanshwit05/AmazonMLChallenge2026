# Business Entity Resolution — Implementation Plan

This file is the technical plan for a coding assistant to implement. The full problem
statement PDF will be provided separately — read it first. This document translates it
into concrete build steps, design decisions, and things to never get wrong.

---

## 0. Non-negotiable constraints (violating any of these = rejected/disqualified)

- Output two TSV files in `output/`: `matching_results.tsv` (scored) and
  `candidate_pairs.tsv` (not scored, used to audit blocking).
- `matching_results.tsv`: exactly one row per Source-1 **test** entity. Empty
  `matched_entity_ids` string for singletons. Comma-separated `S2-`/`S3-` ids only.
  No duplicate ids within a row's list. No duplicate `source1_entity_id` rows. Every id
  must exist in the test set (no self-matches to S1, no invented ids).
- `candidate_pairs.tsv`: same row/format rules, but holds the **final** candidate set —
  i.e. exactly what gets fed into the matching model at inference time, after all
  blocking/filtering stages, not an earlier raw blocking dump. Every id that ends up in
  `matching_results.tsv` must also appear in the matching S1 row of `candidate_pairs.tsv`.
- Run `utils/validate_submission.py` before every submission; it must print `PASS`.
- **Fair play**: no external APIs, no geocoding services, no business-registry lookups,
  no internet calls at all during inference. Static abbreviation/suffix dictionaries built
  from general knowledge (e.g. `Rd`→`Road`, `Ltd`→`Limited`) are fine — looking up an
  actual business's real address/identity anywhere online is not.
- **Model constraint**: the final model must be MIT or Apache-2.0 licensed and ≤8B
  parameters. This permits small open pretrained embedding/cross-encoder models
  (e.g. MiniLM-class, multilingual sentence-transformers) but not proprietary APIs or
  large frontier LLMs.
- **Country is an open set.** Train covers US and India only. Test adds **France**,
  never seen in training. Do not hardcode, filter, or one-hot-restrict anything to
  `{US, India}`. Every test S1 entity — French ones included — must get a row in the
  output. Design all country-related logic as *agreement/disagreement between two
  records*, not as memorized category values, so it generalizes to the unseen label.

## 1. Data shape reminders

- Three source files per split: `*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`.
  Columns: `entity_id` (prefixed `S1-`/`S2-`/`S3-`), `business_name`,
  `business_address`, `country`. No explicit source column — infer it from the id
  prefix / which file the row came from.
- Source 1 is the **deduplicated reference** — no duplicates within S1 itself. Task is:
  for every S1 entity, find all matching records in S2 and S3 (zero, one, or many).
- `train_ground_truth.tsv`: `source1_entity_id`, `matched_entity_ids` (comma-separated,
  empty when no matches — these are the singletons).
- Files are `.tsv` — **always** read with `sep="\t"` explicitly. Addresses and id lists
  contain commas, which is exactly why tabs are used as the delimiter; reading without
  `sep="\t"` silently collapses everything into one column.
- Expected noise (confirm empirically via EDA, don't just assume from this list):
  - Names: abbreviation variants (`Corp`/`Corporation`, `Pvt`/`Private`, `Ltd`/`Limited`),
    legal-suffix inconsistency, DBA/trade names, `&` vs `and`, word-order transpositions,
    typos.
  - Addresses: abbreviations (`Rd`/`Road`, `St`/`Street`), transliteration variants,
    missing components (no PIN/zip, no state), landmark references (`Near SBI ATM`),
    municipal numbering quirks, component reordering.

## 2. Evaluation — design everything around this

- Metric: **F₀.5** per Source-1 entity, then macro-averaged across all S1 entities.
  `F0.5 = (1.25 * P * R) / (0.25 * P + R)` — precision weighted 2x over recall.
- Singletons matter a lot: correctly predicting an empty list for a true singleton scores
  **1.0**; predicting any match for it scores **0.0**. With a nontrivial singleton rate
  in this data, correctly saying "no match" is one of the highest-leverage things the
  model does — don't let blocking recall optimization make you forget precision here.
- Implication: tune every threshold to be precision-leaning, not the default F1-style
  midpoint. A missed match costs less than a false merge.

## 3. Pipeline stages (build and validate in this order)

### Stage A — EDA (do this before writing pipeline code)
- Confirm tab-separated parsing works; check schema/nulls per column per source.
- Distribution of match count per S1 entity (0 / 1 / many) from ground truth — gives the
  real singleton rate and multi-match rate.
- Check whether any S2/S3 id appears as a match for more than one S1 entity in ground
  truth. If never, treat "each S2/S3 record belongs to at most one true entity" as a
  hard constraint used later in conflict resolution.
- Field completeness by source/country; string length & token-count distributions.
- Manually inspect ~30-50 true-positive pairs from ground truth to catalogue the *actual*
  noise patterns present (don't just trust the bullet list above).

### Stage B — Preprocessing / normalization
Apply identically to all three sources:
- Unicode normalize (NFKC); also keep an accent-stripped ("ascii-folded") variant of
  every string — this helps transliteration matching and helps unseen French accented
  text match non-accented variants without ever training on French data.
- Lowercase; map `&` → `and` before stripping punctuation.
- Legal-suffix normalization dictionary (built once, general knowledge, not a lookup
  service): `Pvt`/`Private`, `Ltd`/`Limited`, `Corp`/`Corporation`, `Inc`, `LLC`,
  `SARL`/`Sàrl`, `GmbH`, etc. Keep both raw and suffix-stripped name; suffix agreement
  becomes its own feature.
- Address abbreviation dictionary: `Rd`/`Road`, `St`/`Street`, `Ave`/`Avenue`, `Nr`/`Near`,
  `Apt`, etc.
- Regex-only (no external parser library, to stay unambiguously within fair-play) address
  component extraction: leading street number, trailing postal code (detect by digit
  pattern — 5-digit US zip, 6-digit India PIN, 5-digit France code — not hardcoded per
  country), residual free-text middle segment.
- Split DBA/trade names on separators like `/`, `dba`, `t/a`, `formerly`, `aka` into an
  alias list; match candidates against the best-scoring alias, not just the primary name.

### Stage C — Blocking / candidate generation
This sets the recall ceiling for the whole pipeline — over-invest here. Use the **union**
of multiple independent strategies, then dedupe:
1. Exact/near-exact keys: normalized-name prefix, exact postal code match, phonetic code
   (Soundex/Double Metaphone) on the first significant name token.
2. Character n-gram inverted index + MinHash/LSH on normalized name — catches typos and
   transpositions cheaply at scale.
3. Dense embedding retrieval: embed `name + address` with a small open pretrained
   sentence-embedding model (MIT/Apache, ideally multilingual — this is what covers
   French for free at test time). Build a FAISS (or similar) ANN index per candidate
   pool (S2, S3) and pull top-K neighbors for every S1 entity.
4. Union all of the above per S1 entity as the working candidate pool.
5. **Do not hard-filter by country** — use country agreement as a feature later, not a
   blocking gate, so mislabeled country or the unseen France label never silently kills
   recall.
- Validate on a held-out split: **recall ceiling** (fraction of true matches surviving
  into the candidate set — push this above ~98% before tuning the matching model) and
  **reduction ratio** (candidates considered vs. all possible pairs).
- `candidate_pairs.tsv` is generated directly from this stage's final output, after all
  filtering — this is literally what gets written to that file.

### Stage D — Feature engineering (per S1–candidate pair)
- Name similarity: exact match (post-normalization), Levenshtein ratio, Jaro-Winkler,
  token-set/token-sort ratio, Jaccard on token sets, char-n-gram TF-IDF cosine, dense
  embedding cosine, acronym match, best-alias match score, length/token-count diff.
- Address similarity: same metric family on full normalized address, plus component-level
  features where parsing succeeded (street-number match, postal-code exact/partial match,
  city-token overlap).
- Country: encode as a match/mismatch boolean (generalizes automatically to France); if
  also using a categorical encoding, add an explicit "unseen" bucket.
- Meta features: which blocking strategy(ies) surfaced this candidate; rank of this
  candidate among all candidates for the same S1 by embedding similarity; margin between
  top-1 and top-2 candidate score for the same S1 (small margin = ambiguous → lean
  toward caution given the precision penalty).

### Stage E — Matching model
- Frame as pairwise binary classification (per S1–candidate pair), not clustering.
- Baseline: gradient-boosted trees (LightGBM/XGBoost) on the engineered features —
  fast, strong, interpretable, no license/size concerns.
- Stronger option: fine-tune a small pretrained cross-encoder (MiniLM/DistilBERT-class,
  MIT/Apache, well under 8B params, multilingual variant preferred) on
  `name1+address1 [SEP] name2+address2` → match probability.
- Combine via stacking (cross-encoder score as an extra GBM feature) or a validated
  weighted blend — pick whichever wins on held-out F₀.5.
- Calibrate final probabilities (Platt scaling / isotonic regression) so a fixed
  threshold behaves consistently even if the score distribution shifts on the unseen
  French subset.

### Stage F — Threshold tuning & conflict resolution
- Sweep decision threshold on the validation split, optimize directly for macro F₀.5 —
  never default to 0.5.
- Lean the threshold higher than an F1-optimal point, since false merges cost 2x.
- Use the top-1/top-2 margin feature as a gating rule: a thin margin can justify
  suppressing the match rather than risking a false merge.
- If EDA confirmed the injective constraint (Stage A): post-process so a single S2/S3 id
  is only kept as a match for the highest-scoring S1 entity among any it exceeded
  threshold for; drop it from the rest.
- Anything below threshold, or with an empty candidate list, becomes a singleton (empty
  `matched_entity_ids`) — don't force a match just because candidates exist.

### Stage G — Output & validation
- Write `candidate_pairs.tsv` from Stage C's final candidate set (post-filtering) and
  `matching_results.tsv` from Stage F's decisions.
- Run `python3 utils/validate_submission.py --matching output/matching_results.tsv
  --candidate output/candidate_pairs.tsv --test-dir dataset/test` — fix everything until
  it prints `PASS` before submitting.
- Package per the required zip structure: `output/`, `code/business_entity_resolution/
  {src, README.md, requirements.txt}`, filled-in `Documentation_template.md`.

## 4. Validation strategy (important given the unseen-country test set)

- Group-split by S1 entity id so no entity leaks between train/val.
- Run a **cross-country generalization check**: train on US-only pairs, validate on
  India-only pairs (and vice versa). Consistent F₀.5 across this swap is the best
  available proxy for how the pipeline will behave on French records it has never seen —
  much stronger evidence than a random in-distribution split.

## 5. Things to explicitly avoid

- Don't call any external API, geocoding service, or business database at any point —
  automatic disqualification if detected.
- Don't hardcode logic to `{US, India}` anywhere (filters, one-hot encodings, if/else
  branches) — France must flow through the same code paths.
- Don't force a match for every S1 entity just because it has candidates — singletons are
  worth real points.
- Don't treat this as a clustering problem needing global 1-to-1 assignment across all
  three sources at once — it's S1-anchored: for each S1 entity, find its S2/S3 matches
  independently (with the one exception of the injective post-processing rule in Stage F,
  if confirmed by EDA).
- Don't let `candidate_pairs.tsv` be an earlier/looser blocking dump than what the model
  actually scores — it must be the exact final candidate set, or the validator will flag
  a mismatch (a matched id that never appeared as a candidate is a pipeline bug signal).
- Don't use a model that isn't MIT/Apache-2.0 licensed or that exceeds 8B parameters as
  the final scoring model.

## 6. Documentation deliverable

The methodology write-up (`Documentation_template.md`) must cover: overall methodology,
blocking/candidate-generation strategy (with measured recall ceiling and reduction
ratio), model architecture and feature engineering, threshold-selection process tied to
F₀.5, and generalization evidence (the cross-country validation results). No page limit —
prioritize clarity and depth.

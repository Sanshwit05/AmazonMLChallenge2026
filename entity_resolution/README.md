# Business Entity Resolution Pipeline

A machine learning pipeline for resolving business entities across three independent
data sources with noisy, inconsistent records.

## Architecture

The pipeline follows these stages:

1. **Stage A — EDA**: Exploratory data analysis to understand data characteristics
2. **Stage B — Preprocessing**: Unicode normalization, ASCII folding, legal suffix expansion, address abbreviation handling
3. **Stage C — Blocking**: Multi-strategy candidate generation (inverted index + TF-IDF n-gram)
4. **Stage D — Feature Engineering**: ~42 pairwise similarity features (name, address, country)
5. **Stage E — Matching Model**: LightGBM binary classifier with class-imbalance handling
6. **Stage F — Threshold Tuning**: F₀.₅-optimized threshold sweep with injective conflict resolution
7. **Stage G — Output**: Generates `matching_results.tsv` and `candidate_pairs.tsv`

## Setup

```bash
# Install uv if not already installed
# Then from the entity_resolution directory:
uv sync
```

## Usage

```bash
# Run EDA (optional, recommended first)
uv run python run.py eda

# Train the model on training data
uv run python run.py train --max-s1 50000

# Generate predictions for test data (requires trained model)
uv run python run.py predict

# Full end-to-end pipeline (train + predict)
uv run python run.py full --max-s1 50000

# Validate submission format
uv run python run.py validate
```

## Project Structure

```
entity_resolution/
├── src/
│   ├── __init__.py          # Package init
│   ├── __main__.py          # Package entry point
│   ├── config.py            # Configuration, paths, hyperparameters
│   ├── preprocessing.py     # Stage B: Text normalization
│   ├── blocking.py          # Stage C: Candidate generation
│   ├── features.py          # Stage D: Pairwise feature engineering
│   ├── model.py             # Stage E+F: LightGBM training, threshold tuning
│   ├── io_utils.py          # Stage G: File I/O and output writing
│   ├── eda.py               # Stage A: Exploratory data analysis
│   └── pipeline.py          # Main orchestrator
├── run.py                   # CLI entry point
├── pyproject.toml           # Dependencies (managed by uv)
└── models/                  # Saved model artifacts (created at runtime)
```

## Key Design Decisions

- **Country-agnostic**: No hardcoding to US/India. Country is encoded as match/mismatch boolean.
- **Precision-heavy**: F₀.₅ metric rewards precision 2× over recall. Thresholds lean conservative.
- **Singleton-aware**: Correctly predicting "no match" scores 1.0 per the evaluation.
- **Scale-aware**: Sampling for training, chunked processing for prediction.
- **Fair-play compliant**: No external APIs, no geocoding, no business lookups.
- **License compliant**: LightGBM (MIT), all dependencies are open source.

## Dependencies

- pandas, numpy, scipy — data manipulation
- scikit-learn — TF-IDF vectorization
- lightgbm — gradient boosted tree classifier
- rapidfuzz — fast string similarity metrics
- tqdm — progress bars

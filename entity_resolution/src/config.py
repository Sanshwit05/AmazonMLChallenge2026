"""
Configuration constants and paths for the entity resolution pipeline.
"""
import os

# ──────────────────────────────────────────────────────────────
# Paths (relative to the project root = d:\hackathon)
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STUDENT_RESOURCE = os.path.join(PROJECT_ROOT, "student_resource")

TRAIN_DIR = os.path.join(STUDENT_RESOURCE, "dataset", "train")
TEST_DIR = os.path.join(STUDENT_RESOURCE, "dataset", "test")

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# ──────────────────────────────────────────────────────────────
# Pipeline parameters
# ──────────────────────────────────────────────────────────────
# Blocking
TFIDF_NGRAM_RANGE = (2, 4)  # character n-gram range for TF-IDF
TFIDF_TOP_K = 300            # top-K candidates per S1 from TF-IDF blocking
NAME_PREFIX_LEN = 4         # prefix length for exact-prefix blocking

# Matching model
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 2500,
    "learning_rate": 0.02,
    "num_leaves": 255,
    "max_depth": 15,
    "min_child_samples": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# Threshold (will be tuned on validation; this is initial)
MATCH_THRESHOLD = 0.5

# Validation
VAL_FRACTION = 0.15
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────
# Legal suffix normalization dictionary
# ──────────────────────────────────────────────────────────────
LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "pvt.": "private",
    "ltd": "limited",
    "ltd.": "limited",
    "ltda": "limited",
    "corp": "corporation",
    "corp.": "corporation",
    "inc": "incorporated",
    "inc.": "incorporated",
    "llc": "",
    "l.l.c.": "",
    "l.l.c": "",
    "llp": "",
    "l.l.p.": "",
    "l.l.p": "",
    "plc": "",
    "co": "company",
    "co.": "company",
    "assoc": "associates",
    "assoc.": "associates",
    "bros": "brothers",
    "bros.": "brothers",
    "intl": "international",
    "intl.": "international",
    "natl": "national",
    "natl.": "national",
    "govt": "government",
    "govt.": "government",
    "svcs": "services",
    "svc": "service",
    "mfg": "manufacturing",
    "mgmt": "management",
    "grp": "group",
    "dept": "department",
    "dept.": "department",
    "univ": "university",
    "univ.": "university",
    "tech": "technology",
    "engr": "engineering",
    "engr.": "engineering",
    "engg": "engineering",
    "engg.": "engineering",
    # French
    "sarl": "sarl",
    "sàrl": "sarl",
    "sa": "sa",
    "s.a.": "sa",
    "sas": "sas",
    "s.a.s.": "sas",
    "eurl": "eurl",
    # German
    "gmbh": "gmbh",
    "ag": "ag",
}

# ──────────────────────────────────────────────────────────────
# Address abbreviation dictionary
# ──────────────────────────────────────────────────────────────
ADDRESS_ABBREV_MAP = {
    "rd": "road",
    "rd.": "road",
    "st": "street",
    "st.": "street",
    "ave": "avenue",
    "ave.": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "ln": "lane",
    "ln.": "lane",
    "ct": "court",
    "ct.": "court",
    "pl": "place",
    "pl.": "place",
    "cir": "circle",
    "cir.": "circle",
    "hwy": "highway",
    "hwy.": "highway",
    "pkwy": "parkway",
    "pkwy.": "parkway",
    "sq": "square",
    "sq.": "square",
    "apt": "apartment",
    "apt.": "apartment",
    "ste": "suite",
    "ste.": "suite",
    "fl": "floor",
    "fl.": "floor",
    "bldg": "building",
    "bldg.": "building",
    "nr": "near",
    "nr.": "near",
    "opp": "opposite",
    "opp.": "opposite",
    "ext": "extension",
    "ext.": "extension",
    "mt": "mount",
    "mt.": "mount",
    "n": "north",
    "n.": "north",
    "s": "south",
    "s.": "south",
    "e": "east",
    "e.": "east",
    "w": "west",
    "w.": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

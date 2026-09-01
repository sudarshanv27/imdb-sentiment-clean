from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

EXTERNAL_DATA_DIR = DATA_DIR / "external"


# ============================================================
# IMDB DATASET
# ============================================================

IMDB_DATASET_DIR = RAW_DATA_DIR / "aclImdb"

TRAIN_RAW_DIR = IMDB_DATASET_DIR / "train"

TEST_RAW_DIR = IMDB_DATASET_DIR / "test"


# Training folders
TRAIN_POS_DIR = TRAIN_RAW_DIR / "pos"

TRAIN_NEG_DIR = TRAIN_RAW_DIR / "neg"

TRAIN_UNSUP_DIR = TRAIN_RAW_DIR / "unsup"


# Testing folders
TEST_POS_DIR = TEST_RAW_DIR / "pos"

TEST_NEG_DIR = TEST_RAW_DIR / "neg"


# ============================================================
# PROCESSED DATA
# ============================================================

TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"

VALIDATION_DATA_FILE = PROCESSED_DATA_DIR / "validation.csv"

TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

# ============================================================
# FEATURE DATA
# ============================================================

FEATURES_DIR = DATA_DIR / "features"

X_TRAIN_FILE = FEATURES_DIR / "X_train.pkl"
X_VALIDATION_FILE = FEATURES_DIR / "X_validation.pkl"
X_TEST_FILE = FEATURES_DIR / "X_test.pkl"

Y_TRAIN_FILE = FEATURES_DIR / "y_train.pkl"
Y_VALIDATION_FILE = FEATURES_DIR / "y_validation.pkl"
Y_TEST_FILE = FEATURES_DIR / "y_test.pkl"

# ============================================================
# MODEL DIRECTORY
# ============================================================

MODELS_DIR = PROJECT_ROOT / "models"

TFIDF_MODEL_FILE = MODELS_DIR / "tfidf_vectorizer.pkl"

SENTIMENT_MODEL_FILE = MODELS_DIR / "sentiment_model.pkl"


# ============================================================
# REPORT DIRECTORY
# ============================================================

REPORTS_DIR = PROJECT_ROOT / "reports"

CONFUSION_MATRIX_FILE = REPORTS_DIR / "confusion_matrix.png"

CLASSIFICATION_REPORT_FILE = (
    REPORTS_DIR / "classification_report.txt"
)


# ============================================================
# MACHINE LEARNING SETTINGS
# ============================================================

RANDOM_STATE = 42

VALIDATION_SIZE = 0.20

MAX_FEATURES = 10000

MIN_DF = 2

MAX_DF = 0.95


# ============================================================
# DATA COLUMN NAMES
# ============================================================

TEXT_COLUMN = "review"

TARGET_COLUMN = "sentiment"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

DIRECTORIES = [
    PROCESSED_DATA_DIR,
    FEATURES_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )
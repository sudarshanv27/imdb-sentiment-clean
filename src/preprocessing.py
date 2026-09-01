"""
IMDB Sentiment Analysis
Data Preprocessing Pipeline

This script:
1. Reads IMDB .txt reviews
2. Assigns positive/negative labels
3. Cleans the review text
4. Creates train/validation/test datasets
5. Saves processed CSV files
"""

import re
import html
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_POS_DIR,
    TRAIN_NEG_DIR,
    TEST_POS_DIR,
    TEST_NEG_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    VALIDATION_SIZE,
    TEXT_COLUMN,
    TARGET_COLUMN,
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean an IMDB review.

    Steps:
    - Convert HTML entities
    - Remove HTML tags
    - Convert text to lowercase
    - Remove URLs
    - Keep letters and basic punctuation
    - Remove extra whitespace
    """

    # Convert HTML entities such as &amp;
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep letters, numbers and basic punctuation
    text = re.sub(r"[^a-z0-9\s!?.,']", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# READ REVIEWS
# ============================================================

def read_reviews(folder: Path, sentiment: int) -> list:
    """
    Read all .txt reviews from a folder.

    sentiment:
        1 = positive
        0 = negative
    """

    records = []

    files = sorted(folder.glob("*.txt"))

    print(f"Reading {len(files):,} files from: {folder}")

    for file_path in files:

        try:
            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            text = clean_text(text)

            if text:
                records.append(
                    {
                        TEXT_COLUMN: text,
                        TARGET_COLUMN: sentiment
                    }
                )

        except Exception as error:

            print(
                f"Warning: Could not read {file_path.name}: {error}"
            )

    return records


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data() -> pd.DataFrame:

    print("\nLoading training data...")

    positive_reviews = read_reviews(
        TRAIN_POS_DIR,
        sentiment=1
    )

    negative_reviews = read_reviews(
        TRAIN_NEG_DIR,
        sentiment=0
    )

    records = positive_reviews + negative_reviews

    df = pd.DataFrame(records)

    return df


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data() -> pd.DataFrame:

    print("\nLoading test data...")

    positive_reviews = read_reviews(
        TEST_POS_DIR,
        sentiment=1
    )

    negative_reviews = read_reviews(
        TEST_NEG_DIR,
        sentiment=0
    )

    records = positive_reviews + negative_reviews

    df = pd.DataFrame(records)

    return df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
    dataset_name: str
) -> None:

    print(f"\n{'=' * 60}")
    print(f"{dataset_name.upper()} DATASET")
    print(f"{'=' * 60}")

    print(f"Total reviews: {len(df):,}")

    print("\nSentiment distribution:")

    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nMissing values:")

    print(
        df.isnull().sum()
    )

    print("\nDuplicate reviews:")

    print(
        df[TEXT_COLUMN].duplicated().sum()
    )


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("IMDB SENTIMENT ANALYSIS")
    print("DATA PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load original training data
    # --------------------------------------------------------

    train_df = load_training_data()

    # --------------------------------------------------------
    # 2. Load original test data
    # --------------------------------------------------------

    test_df = load_test_data()

    # --------------------------------------------------------
    # 3. Validate raw loaded data
    # --------------------------------------------------------

    validate_dataset(
        train_df,
        "Training"
    )

    validate_dataset(
        test_df,
        "Testing"
    )

    # --------------------------------------------------------
    # 4. Split training into train + validation
    # --------------------------------------------------------

    train_split, validation_split = train_test_split(
        train_df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df[TARGET_COLUMN]
    )

    # --------------------------------------------------------
    # 5. Reset indexes
    # --------------------------------------------------------

    train_split = train_split.reset_index(drop=True)

    validation_split = validation_split.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)

    # --------------------------------------------------------
    # 6. Save processed datasets
    # --------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_file = PROCESSED_DATA_DIR / "train.csv"

    validation_file = (
        PROCESSED_DATA_DIR / "validation.csv"
    )

    test_file = PROCESSED_DATA_DIR / "test.csv"

    train_split.to_csv(
        train_file,
        index=False
    )

    validation_split.to_csv(
        validation_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    # --------------------------------------------------------
    # 7. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"Training reviews   : {len(train_split):,}")
    print(f"Validation reviews : {len(validation_split):,}")
    print(f"Testing reviews    : {len(test_df):,}")

    print(
        f"\nTotal processed reviews: "
        f"{len(train_split) + len(validation_split) + len(test_df):,}"
    )

    print("\nSaved files:")

    print(f"Train      : {train_file}")

    print(f"Validation : {validation_file}")

    print(f"Test       : {test_file}")


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
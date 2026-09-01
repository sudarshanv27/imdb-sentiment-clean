"""
TF-IDF Feature Extraction
IMDB Sentiment Analysis Project
"""

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TRAIN_DATA_FILE,
    VALIDATION_DATA_FILE,
    TEST_DATA_FILE,
    FEATURES_DIR,
    X_TRAIN_FILE,
    X_VALIDATION_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_VALIDATION_FILE,
    Y_TEST_FILE,
    TFIDF_MODEL_FILE,
    TEXT_COLUMN,
    TARGET_COLUMN,
    MAX_FEATURES,
    MIN_DF,
    MAX_DF,
)


def load_data():
    """Load processed train, validation and test datasets."""

    print("=" * 60)
    print("TF-IDF FEATURE EXTRACTION")
    print("=" * 60)

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_DATA_FILE)
    validation_df = pd.read_csv(VALIDATION_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    print(f"Training reviews   : {len(train_df):,}")
    print(f"Validation reviews : {len(validation_df):,}")
    print(f"Testing reviews    : {len(test_df):,}")

    return train_df, validation_df, test_df


def create_vectorizer():
    """Create the TF-IDF vectorizer."""

    print("\nCreating TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        ngram_range=(1, 2),
    )

    return vectorizer


def extract_features():
    """Convert text into TF-IDF numerical features."""

    train_df, validation_df, test_df = load_data()

    # --------------------------------------------------------
    # Separate text and labels
    # --------------------------------------------------------

    X_train_text = train_df[TEXT_COLUMN]
    y_train = train_df[TARGET_COLUMN]

    X_validation_text = validation_df[TEXT_COLUMN]
    y_validation = validation_df[TARGET_COLUMN]

    X_test_text = test_df[TEXT_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    # --------------------------------------------------------
    # Create vectorizer
    # --------------------------------------------------------

    vectorizer = create_vectorizer()

    # --------------------------------------------------------
    # IMPORTANT:
    # Fit ONLY on training data
    # --------------------------------------------------------

    print("\nFitting TF-IDF on training data...")

    X_train = vectorizer.fit_transform(X_train_text)

    print("Transforming validation data...")

    X_validation = vectorizer.transform(X_validation_text)

    print("Transforming test data...")

    X_test = vectorizer.transform(X_test_text)

    # --------------------------------------------------------
    # Display feature information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE INFORMATION")
    print("=" * 60)

    print(f"Training matrix   : {X_train.shape}")
    print(f"Validation matrix : {X_validation.shape}")
    print(f"Testing matrix    : {X_test.shape}")

    print(f"\nNumber of features: {len(vectorizer.get_feature_names_out()):,}")

    # --------------------------------------------------------
    # Save feature matrices
    # --------------------------------------------------------

    print("\nSaving feature matrices...")

    joblib.dump(X_train, X_TRAIN_FILE)
    joblib.dump(X_validation, X_VALIDATION_FILE)
    joblib.dump(X_test, X_TEST_FILE)

    # --------------------------------------------------------
    # Save target labels
    # --------------------------------------------------------

    print("Saving target labels...")

    joblib.dump(y_train, Y_TRAIN_FILE)
    joblib.dump(y_validation, Y_VALIDATION_FILE)
    joblib.dump(y_test, Y_TEST_FILE)

    # --------------------------------------------------------
    # Save TF-IDF vectorizer
    # --------------------------------------------------------

    print("Saving TF-IDF vectorizer...")

    joblib.dump(vectorizer, TFIDF_MODEL_FILE)

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TF-IDF FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print("\nFiles created:")

    print(f"X_train       : {X_TRAIN_FILE}")
    print(f"X_validation  : {X_VALIDATION_FILE}")
    print(f"X_test        : {X_TEST_FILE}")

    print(f"y_train       : {Y_TRAIN_FILE}")
    print(f"y_validation  : {Y_VALIDATION_FILE}")
    print(f"y_test        : {Y_TEST_FILE}")

    print(f"Vectorizer    : {TFIDF_MODEL_FILE}")

    return X_train, X_validation, X_test


if __name__ == "__main__":
    extract_features()
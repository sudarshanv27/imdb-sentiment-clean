"""
IMDB Sentiment Analysis
Model Training

Models:
1. Logistic Regression
2. Linear SVM
3. Multinomial Naive Bayes
"""

import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_DIR = PROJECT_ROOT / "data" / "features"

MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURE FILES
# ============================================================

X_TRAIN_FILE = FEATURES_DIR / "X_train.pkl"
X_VALIDATION_FILE = FEATURES_DIR / "X_validation.pkl"
X_TEST_FILE = FEATURES_DIR / "X_test.pkl"

Y_TRAIN_FILE = FEATURES_DIR / "y_train.pkl"
Y_VALIDATION_FILE = FEATURES_DIR / "y_validation.pkl"
Y_TEST_FILE = FEATURES_DIR / "y_test.pkl"


# ============================================================
# MODEL FILES
# ============================================================

LOGISTIC_REGRESSION_FILE = (
    MODELS_DIR / "logistic_regression.pkl"
)

LINEAR_SVM_FILE = (
    MODELS_DIR / "linear_svm.pkl"
)

NAIVE_BAYES_FILE = (
    MODELS_DIR / "naive_bayes.pkl"
)

BEST_MODEL_FILE = (
    MODELS_DIR / "sentiment_model.pkl"
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 60)
print("IMDB SENTIMENT ANALYSIS")
print("MODEL TRAINING")
print("=" * 60)

print("\nLoading feature matrices...")

X_train = joblib.load(X_TRAIN_FILE)
X_validation = joblib.load(X_VALIDATION_FILE)

y_train = joblib.load(Y_TRAIN_FILE)
y_validation = joblib.load(Y_VALIDATION_FILE)


print("\nFeature shapes:")
print("X_train      :", X_train.shape)
print("X_validation :", X_validation.shape)

print("\nLabel shapes:")
print("y_train      :", y_train.shape)
print("y_validation :", y_validation.shape)


# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, model_name):
    """
    Train and evaluate a model on the validation dataset.
    """

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision = precision_score(
        y_validation,
        predictions
    )

    recall = recall_score(
        y_validation,
        predictions
    )

    f1 = f1_score(
        y_validation,
        predictions
    )

    print(f"\n{model_name} Results")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return {
        "model": model,
        "name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# MODEL 1 — LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    solver="liblinear"
)

logistic_result = evaluate_model(
    logistic_model,
    "Logistic Regression"
)


# ============================================================
# MODEL 2 — LINEAR SVM
# ============================================================

svm_model = LinearSVC(
    C=1.0,
    max_iter=5000,
    random_state=42
)

svm_result = evaluate_model(
    svm_model,
    "Linear SVM"
)


# ============================================================
# MODEL 3 — NAIVE BAYES
# ============================================================

naive_bayes_model = MultinomialNB(
    alpha=1.0
)

naive_bayes_result = evaluate_model(
    naive_bayes_model,
    "Multinomial Naive Bayes"
)


# ============================================================
# COLLECT RESULTS
# ============================================================

results = [
    logistic_result,
    svm_result,
    naive_bayes_result,
]


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"{'Model':<25}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 60)

for result in results:

    print(
        f"{result['name']:<25}"
        f"{result['accuracy']:<12.4f}"
        f"{result['precision']:<12.4f}"
        f"{result['recall']:<12.4f}"
        f"{result['f1']:<12.4f}"
    )


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_result = max(
    results,
    key=lambda result: result["f1"]
)

best_model = best_result["model"]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_result["name"])
print(f"Validation Accuracy: {best_result['accuracy']:.4f}")
print(f"Validation F1      : {best_result['f1']:.4f}")


# ============================================================
# SAVE INDIVIDUAL MODELS
# ============================================================

print("\nSaving individual models...")

joblib.dump(
    logistic_model,
    LOGISTIC_REGRESSION_FILE
)

joblib.dump(
    svm_model,
    LINEAR_SVM_FILE
)

joblib.dump(
    naive_bayes_model,
    NAIVE_BAYES_FILE
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    BEST_MODEL_FILE
)


# ============================================================
# COMPLETION STATUS
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

print("\nModels saved:")
print("Logistic Regression:", LOGISTIC_REGRESSION_FILE)
print("Linear SVM         :", LINEAR_SVM_FILE)
print("Naive Bayes        :", NAIVE_BAYES_FILE)
print("Best Model         :", BEST_MODEL_FILE)

print("\nSTATUS: MODEL TRAINING READY FOR FINAL EVALUATION")
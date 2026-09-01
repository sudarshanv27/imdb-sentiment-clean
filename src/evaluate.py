"""
IMDB Sentiment Analysis
Final Model Evaluation

Evaluates the selected model on the untouched test dataset.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.config import (
    X_TEST_FILE,
    Y_TEST_FILE,
    SENTIMENT_MODEL_FILE,
    REPORTS_DIR,
    CONFUSION_MATRIX_FILE,
    CLASSIFICATION_REPORT_FILE,
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("IMDB SENTIMENT ANALYSIS")
print("FINAL MODEL EVALUATION")
print("=" * 60)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

required_files = {
    "Test features": X_TEST_FILE,
    "Test labels": Y_TEST_FILE,
    "Sentiment model": SENTIMENT_MODEL_FILE,
}

for name, file_path in required_files.items():

    print(
        f"{name:<20}: "
        f"{'FOUND' if file_path.exists() else 'MISSING'}"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = joblib.load(X_TEST_FILE)
y_test = joblib.load(Y_TEST_FILE)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nTest sentiment distribution:")
print(y_test.value_counts())


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = joblib.load(SENTIMENT_MODEL_FILE)

print("Model:", type(model).__name__)


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

print("Predictions generated:", len(y_pred))


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Negative",
        "Positive"
    ]
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

print("\nMatrix interpretation:")

print(
    f"True Negative : {cm[0, 0]}"
)

print(
    f"False Positive: {cm[0, 1]}"
)

print(
    f"False Negative: {cm[1, 0]}"
)

print(
    f"True Positive  : {cm[1, 1]}"
)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

print("\nCreating confusion matrix image...")

fig, ax = plt.subplots(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Negative",
        "Positive"
    ]
)

display.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "IMDB Sentiment Analysis - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=150
)

plt.close()

print(
    "Saved:",
    CONFUSION_MATRIX_FILE
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    CLASSIFICATION_REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "IMDB SENTIMENT ANALYSIS\n"
    )

    file.write(
        "FINAL TEST CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Model: {type(model).__name__}\n\n"
    )

    file.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall   : {recall:.4f}\n"
    )

    file.write(
        f"F1 Score : {f1:.4f}\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(report)

    file.write(
        "\n\nConfusion Matrix\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        str(cm)
    )


# ============================================================
# SAVE ACCURACY REPORT
# ============================================================

accuracy_report_file = (
    REPORTS_DIR / "accuracy_report.txt"
)

with open(
    accuracy_report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "IMDB SENTIMENT ANALYSIS\n"
    )

    file.write(
        "FINAL MODEL ACCURACY REPORT\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Model: {type(model).__name__}\n\n"
    )

    file.write(
        f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)\n"
    )

    file.write(
        f"Precision: {precision:.4f} ({precision * 100:.2f}%)\n"
    )

    file.write(
        f"Recall   : {recall:.4f} ({recall * 100:.2f}%)\n"
    )

    file.write(
        f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)\n"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("FINAL EVALUATION COMPLETE")
print("=" * 60)

print("\nReports created:")

print(
    "Confusion Matrix:",
    CONFUSION_MATRIX_FILE
)

print(
    "Classification Report:",
    CLASSIFICATION_REPORT_FILE
)

print(
    "Accuracy Report:",
    accuracy_report_file
)

print("\nSTATUS: FINAL MODEL EVALUATED SUCCESSFULLY")
"""
IMDB Sentiment Analysis
Prediction Module

This module loads the trained TF-IDF vectorizer and
Logistic Regression model and predicts sentiment for
new movie reviews.
"""

import joblib

from src.config import (
    TFIDF_MODEL_FILE,
    SENTIMENT_MODEL_FILE,
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

def load_models():
    """
    Load the trained TF-IDF vectorizer and sentiment model.
    """

    print("Loading TF-IDF vectorizer...")
    vectorizer = joblib.load(TFIDF_MODEL_FILE)

    print("Loading sentiment model...")
    model = joblib.load(SENTIMENT_MODEL_FILE)

    print("Models loaded successfully.")

    return vectorizer, model


# ============================================================
# PREDICT SENTIMENT
# ============================================================

def predict_sentiment(review, vectorizer, model):
    """
    Predict sentiment for a single review.

    Parameters
    ----------
    review : str
        Movie review entered by the user.

    vectorizer :
        Trained TF-IDF vectorizer.

    model :
        Trained Logistic Regression model.

    Returns
    -------
    dict
        Prediction result containing sentiment,
        confidence and probability.
    """

    if not isinstance(review, str):
        raise TypeError("Review must be a string.")

    review = review.strip()

    if not review:
        raise ValueError("Review cannot be empty.")

    # Convert review into TF-IDF features
    review_vector = vectorizer.transform([review])

    # Predict class
    prediction = model.predict(review_vector)[0]

    # Get probabilities
    probabilities = model.predict_proba(review_vector)[0]

    # Convert numeric prediction to sentiment
    if prediction == 1:
        sentiment = "POSITIVE"
    else:
        sentiment = "NEGATIVE"

    confidence = probabilities[prediction] * 100

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "prediction": int(prediction),
        "positive_probability": probabilities[1] * 100,
        "negative_probability": probabilities[0] * 100,
    }


# ============================================================
# TEST PREDICTIONS
# ============================================================

def main():

    print("=" * 60)
    print("IMDB SENTIMENT ANALYSIS")
    print("PREDICTION SYSTEM")
    print("=" * 60)

    # Load models
    vectorizer, model = load_models()

    print()
    print("=" * 60)
    print("TESTING PREDICTIONS")
    print("=" * 60)

    test_reviews = [
        "This movie was absolutely fantastic. I loved every minute of it.",
        "This was one of the worst movies I have ever watched.",
        "The acting was excellent and the story was very interesting.",
        "Terrible movie. The story was boring and the acting was awful.",
    ]

    for review in test_reviews:

        result = predict_sentiment(
            review,
            vectorizer,
            model
        )

        print()
        print("Review:")
        print(review)

        print()
        print("Prediction :", result["sentiment"])
        print(
            "Confidence : "
            f"{result['confidence']:.2f}%"
        )

        print(
            "Positive probability : "
            f"{result['positive_probability']:.2f}%"
        )

        print(
            "Negative probability : "
            f"{result['negative_probability']:.2f}%"
        )

        print("-" * 60)


if __name__ == "__main__":
    main()
import pytest

from src.predict import load_models, predict_sentiment


@pytest.fixture(scope="module")
def models():
    """Load the trained models once for all tests."""
    return load_models()


def test_models_load(models):
    """Verify that both trained components load successfully."""

    vectorizer, model = models

    assert vectorizer is not None
    assert model is not None


def test_positive_review(models):
    """A clearly positive review should be classified as positive."""

    vectorizer, model = models

    review = (
        "This movie was absolutely fantastic. "
        "The acting was excellent and the story was amazing."
    )

    result = predict_sentiment(
        review,
        vectorizer,
        model,
    )

    assert result["sentiment"] == "POSITIVE"
    assert result["prediction"] == 1
    assert result["positive_probability"] > 50


def test_negative_review(models):
    """A clearly negative review should be classified as negative."""

    vectorizer, model = models

    review = (
        "This movie was terrible. "
        "The acting was awful and the story was boring."
    )

    result = predict_sentiment(
        review,
        vectorizer,
        model,
    )

    assert result["sentiment"] == "NEGATIVE"
    assert result["prediction"] == 0
    assert result["negative_probability"] > 50


def test_empty_review(models):
    """Empty reviews should be rejected."""

    vectorizer, model = models

    with pytest.raises(ValueError):
        predict_sentiment(
            "",
            vectorizer,
            model,
        )


def test_whitespace_review(models):
    """Whitespace-only reviews should be rejected."""

    vectorizer, model = models

    with pytest.raises(ValueError):
        predict_sentiment(
            "     ",
            vectorizer,
            model,
        )


def test_invalid_input(models):
    """Non-string input should be rejected."""

    vectorizer, model = models

    with pytest.raises(TypeError):
        predict_sentiment(
            12345,
            vectorizer,
            model,
        )


def test_probability_range(models):
    """Prediction probabilities must be between 0 and 100."""

    vectorizer, model = models

    review = "I really enjoyed this movie."

    result = predict_sentiment(
        review,
        vectorizer,
        model,
    )

    assert 0 <= result["positive_probability"] <= 100
    assert 0 <= result["negative_probability"] <= 100

    assert (
        abs(
            result["positive_probability"]
            + result["negative_probability"]
            - 100
        )
        < 0.001
    )
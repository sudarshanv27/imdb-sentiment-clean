from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import load_models, predict_sentiment


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description="REST API for IMDb movie review sentiment prediction.",
    version="1.0.0",
)


# ============================================================
# LOAD MODEL
# ============================================================

vectorizer, model = load_models()


# ============================================================
# REQUEST MODEL
# ============================================================

class ReviewRequest(BaseModel):
    review: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "IMDb Sentiment Analysis API",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": type(model).__name__,
        "vectorizer": type(vectorizer).__name__,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: ReviewRequest):

    if not request.review.strip():
        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty.",
        )

    try:

        result = predict_sentiment(
            request.review,
            vectorizer,
            model,
        )

        return {
            "review": request.review,
            "sentiment": result["sentiment"],
            "prediction": int(result["prediction"]),
            "confidence": float(result["confidence"]),
            "positive_probability": float(
                result["positive_probability"]
            ),
            "negative_probability": float(
                result["negative_probability"]
            ),
        }

    except (ValueError, TypeError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
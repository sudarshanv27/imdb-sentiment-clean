from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_positive_prediction():
    response = client.post(
        "/predict",
        json={
            "review": "This movie was absolutely fantastic. I loved every minute of it."
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] == "POSITIVE"
    assert data["prediction"] == 1


def test_negative_prediction():
    response = client.post(
        "/predict",
        json={
            "review": "This movie was terrible, boring and a complete waste of time."
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] == "NEGATIVE"
    assert data["prediction"] == 0


def test_empty_review():
    response = client.post(
        "/predict",
        json={
            "review": ""
        },
    )

    assert response.status_code in [400, 422]


def test_invalid_request():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


def test_probability_values():
    response = client.post(
        "/predict",
        json={
            "review": "The movie was excellent and very enjoyable."
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["confidence"] <= 100
    assert 0 <= data["positive_probability"] <= 100
    assert 0 <= data["negative_probability"] <= 100

    assert abs(
        data["positive_probability"]
        + data["negative_probability"]
        - 100
    ) < 0.01
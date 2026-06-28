"""Integration tests for the Housing Price Prediction API."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import reset_model

client = TestClient(app)


class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True


class TestPredict:
    VALID_FEATURES = {
        "square_footage": 1850,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 1998,
        "lot_size": 7500,
        "distance_to_city_center": 5.6,
        "school_rating": 8.2,
    }

    def test_single_prediction(self):
        resp = client.post("/predict", json={"features": self.VALID_FEATURES})
        assert resp.status_code == 200
        body = resp.json()
        assert "predicted_price" in body
        assert isinstance(body["predicted_price"], float)
        assert 100_000 < body["predicted_price"] < 500_000  # sanity check

    def test_batch_prediction(self):
        resp = client.post(
            "/predict/batch",
            json={"features": [self.VALID_FEATURES, self.VALID_FEATURES]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2

    def test_invalid_features_returns_422(self):
        resp = client.post("/predict", json={"features": {"bedrooms": -1}})
        assert resp.status_code == 422


class TestModelInfo:
    def test_model_info(self):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_type"] == "LinearRegression"
        assert len(body["coefficients"]) == 7
        assert body["training_samples"] > 0
        assert any(m["name"] == "r2_score" for m in body["metrics"])

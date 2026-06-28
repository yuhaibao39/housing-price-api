"""API route definitions for the Housing Price Prediction service."""

import logging

import pandas as pd

from fastapi import APIRouter

from app.model import (
    FEATURE_NAMES,
    get_model,
    get_model_info,
    predict,
)
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CoefficientInfo,
    HealthResponse,
    ModelInfoResponse,
    ModelMetric,
    PredictionRequest,
    SinglePredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _features_to_df(features_list):
    """Convert a list of HousingFeatures Pydantic models to a DataFrame."""
    records = [f.model_dump() for f in features_list]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current service status.",
)
async def health():
    get_model()  # ensures model is loaded
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Predict – single
# ---------------------------------------------------------------------------
@router.post(
    "/predict",
    response_model=SinglePredictionResponse,
    summary="Predict housing price",
    description=(
        "Accepts a single set of housing features and returns the predicted "
        "price in USD."
    ),
)
async def predict_single(request: PredictionRequest):
    model, scaler = get_model()
    df = _features_to_df([request.features])
    pred = predict(model, scaler, df)
    price = round(float(pred[0]), 2)
    return SinglePredictionResponse(
        predicted_price=price,
        input_features=request.features,
    )


# ---------------------------------------------------------------------------
# Predict – batch
# ---------------------------------------------------------------------------
@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch predict housing prices",
    description=(
        "Accepts up to 10,000 sets of housing features and returns "
        "predictions for all of them."
    ),
)
async def predict_batch(request: BatchPredictionRequest):
    model, scaler = get_model()
    df = _features_to_df(request.features)
    preds = predict(model, scaler, df)
    prices = [round(float(p), 2) for p in preds]
    return BatchPredictionResponse(
        predictions=prices,
        count=len(prices),
    )


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------
@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model information",
    description=(
        "Returns the trained model type, feature coefficients, intercept, "
        "and performance metrics (R², MAE, RMSE)."
    ),
)
async def model_info():
    model, _ = get_model()
    info = get_model_info(model)
    return ModelInfoResponse(
        model_type=info["model_type"],
        coefficients=[CoefficientInfo(**c) for c in info["coefficients"]],
        intercept=info["intercept"],
        metrics=[ModelMetric(**m) for m in info["metrics"]],
        training_samples=info["training_samples"],
        feature_names=info["feature_names"],
    )

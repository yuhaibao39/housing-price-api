from pydantic import BaseModel, Field
from typing import List


class HousingFeatures(BaseModel):
    """Single house feature set for prediction."""

    square_footage: float = Field(
        ..., ge=0, description="Total square footage of the house", example=1850
    )
    bedrooms: int = Field(
        ..., ge=0, description="Number of bedrooms", example=3
    )
    bathrooms: float = Field(
        ..., ge=0, description="Number of bathrooms", example=2
    )
    year_built: int = Field(
        ..., ge=1900, le=2030, description="Year the house was built", example=1998
    )
    lot_size: float = Field(
        ..., ge=0, description="Lot size in square feet", example=7500
    )
    distance_to_city_center: float = Field(
        ..., ge=0, description="Distance to city center (miles)", example=5.6
    )
    school_rating: float = Field(
        ..., ge=0, le=10, description="Nearby school rating (0-10)", example=8.2
    )

    class Config:
        json_schema_extra = {
            "example": {
                "square_footage": 1850,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 1998,
                "lot_size": 7500,
                "distance_to_city_center": 5.6,
                "school_rating": 8.2,
            }
        }


class PredictionRequest(BaseModel):
    """Single prediction request."""
    features: HousingFeatures


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    features: List[HousingFeatures] = Field(..., min_length=1, max_length=10_000)


class SinglePredictionResponse(BaseModel):
    """Response for a single prediction."""
    predicted_price: float = Field(..., description="Predicted house price (USD)")
    input_features: HousingFeatures


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[float] = Field(..., description="List of predicted prices (USD)")
    count: int


class ModelMetric(BaseModel):
    """Individual model metric."""
    name: str
    value: float


class CoefficientInfo(BaseModel):
    """Model coefficient information."""
    feature: str
    coefficient: float


class ModelInfoResponse(BaseModel):
    """Response for model-info endpoint."""
    model_type: str
    coefficients: List[CoefficientInfo]
    intercept: float
    metrics: List[ModelMetric]
    training_samples: int
    feature_names: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str

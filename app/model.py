"""
Model training and persistence module.

Trains a LinearRegression model on the provided House Price Dataset CSV,
then persists it via joblib for serving.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

TRAIN_CSV = DATA_DIR / "House Price Dataset.csv"
TEST_CSV = DATA_DIR / "Test Data For Prediction.csv"
MODEL_PATH = MODEL_DIR / "housing_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"

# Feature names used by the model (excluding target 'price' and 'id')
FEATURE_NAMES = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating",
]

# Training metadata filled during train_model()
_training_metadata = {
    "n_samples": 0,
    "r2": 0.0,
    "mae": 0.0,
    "rmse": 0.0,
}

# Shared in-memory globals – populated by load_model() / train_model()
_model_instance = None
_scaler_instance = None


def get_model() -> tuple[LinearRegression, StandardScaler]:
    """Return the cached (model, scaler) tuple, loading if necessary."""
    global _model_instance, _scaler_instance
    if _model_instance is None or _scaler_instance is None:
        _model_instance, _scaler_instance = load_model()
    return _model_instance, _scaler_instance


def reset_model():
    """Clear cached model state (useful for testing)."""
    global _model_instance, _scaler_instance
    _model_instance = _scaler_instance = None


def train_model() -> tuple[LinearRegression, StandardScaler]:
    """
    Train a LinearRegression model on the CSV dataset.

    Returns
    -------
    tuple[LinearRegression, StandardScaler]
        The trained model and fitted scaler.
    """
    logger.info("Loading dataset from %s", TRAIN_CSV)
    df = pd.read_csv(TRAIN_CSV)

    X = df[FEATURE_NAMES]
    y = df["price"]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    logger.info("Training LinearRegression model on %d samples...", len(X_train))
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info("Model trained — R²=%.4f, MAE=%.2f, RMSE=%.2f", r2, mae, rmse)

    # Store metadata (attach to model so it survives joblib persistence)
    _training_metadata["n_samples"] = len(X_train)
    _training_metadata["r2"] = round(r2, 4)
    _training_metadata["mae"] = round(mae, 2)
    _training_metadata["rmse"] = round(rmse, 2)
    model._training_metadata = dict(_training_metadata)

    # Persist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    return model, scaler


def load_model() -> tuple[LinearRegression, StandardScaler]:
    """
    Load a trained model and scaler from disk, or train if not found.

    Populates the module-level cached globals.

    Returns
    -------
    tuple[LinearRegression, StandardScaler]
    """
    global _model_instance, _scaler_instance

    if MODEL_PATH.exists() and SCALER_PATH.exists():
        logger.info("Loading model from %s", MODEL_PATH)
        _model_instance = joblib.load(MODEL_PATH)
        _scaler_instance = joblib.load(SCALER_PATH)
        # Restore metadata from model if available
        if hasattr(_model_instance, "_training_metadata"):
            _training_metadata.update(_model_instance._training_metadata)
        return _model_instance, _scaler_instance

    logger.info("No saved model found — training new model...")
    return train_model()


def predict(
    model: LinearRegression,
    scaler: StandardScaler,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Run inference.

    Parameters
    ----------
    model : LinearRegression
    scaler : StandardScaler
    features : pd.DataFrame
        Must contain columns matching FEATURE_NAMES.

    Returns
    -------
    np.ndarray
        Predicted prices in USD.
    """
    features_scaled = scaler.transform(features[FEATURE_NAMES])
    return model.predict(features_scaled)


def get_model_info(model: LinearRegression) -> dict:
    """
    Return model coefficients, intercept, and performance metrics.

    Parameters
    ----------
    model : LinearRegression

    Returns
    -------
    dict with keys: model_type, coefficients, intercept, metrics, training_samples, feature_names
    """
    coefficients = [
        {"feature": name, "coefficient": round(float(coef), 2)}
        for name, coef in zip(FEATURE_NAMES, model.coef_)
    ]

    metrics = [
        {"name": "r2_score", "value": _training_metadata["r2"]},
        {"name": "mean_absolute_error", "value": _training_metadata["mae"]},
        {"name": "root_mean_squared_error", "value": _training_metadata["rmse"]},
    ]

    return {
        "model_type": type(model).__name__,
        "coefficients": coefficients,
        "intercept": round(float(model.intercept_), 2),
        "metrics": metrics,
        "training_samples": _training_metadata["n_samples"],
        "feature_names": FEATURE_NAMES,
    }

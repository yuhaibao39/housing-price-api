"""
Housing Price Prediction API

A FastAPI-based service that serves a LinearRegression model trained on
the provided House Price Dataset (CSV).

Endpoints
---------
GET  /health       – Health check
POST /predict      – Single house price prediction
POST /predict/batch – Batch house price prediction (up to 10,000)
GET  /model-info   – Model coefficients and performance metrics
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model import get_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan – warm the model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — pre-warming model...")
    try:
        get_model()
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.warning("Model pre-warm failed (will lazy-load on first request): %s", exc)
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Housing Price Prediction API",
    description=(
        "Predicts house prices based on property features using a\n"
        "LinearRegression model trained on the provided dataset.\n\n"
        "## Features\n"
        "- **square_footage** – Total square footage of the house\n"
        "- **bedrooms** – Number of bedrooms\n"
        "- **bathrooms** – Number of bathrooms\n"
        "- **year_built** – Year the house was built\n"
        "- **lot_size** – Lot size in square feet\n"
        "- **distance_to_city_center** – Distance to city center (miles)\n"
        "- **school_rating** – Nearby school rating (0-10)\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Developer",
        "url": "https://github.com/yourusername/housing-price-api",
    },
    license_info={
        "name": "MIT",
    },
)

# Allow CORS for Swagger UI and other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
from app.router import router as api_router  # noqa: E402

app.include_router(api_router)

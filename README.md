# 🏠 Housing Price Prediction API

A containerised FastAPI service that serves a **LinearRegression** model
trained on the provided **House Price Dataset** (50 houses) to predict
house prices in USD.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Health check |
| `/predict` | `POST` | Predict price for a single house |
| `/predict/batch` | `POST` | Batch predict (up to 10,000 houses) |
| `/model-info` | `GET` | Model coefficients & performance metrics |

## Quick Start

### Local (Python)

```bash
cd housing-price-api

# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model & run server
uvicorn app.main:app --reload
```

### Docker

```bash
docker build -t housing-price-api .
docker run -d -p 8000:8000 --name housing-api housing-price-api
```

### Docker Compose

```bash
docker compose up -d
```

Once running, open **http://localhost:8000/docs** for the Swagger UI.

## API Examples

### Predict (single)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "square_footage": 1850,
      "bedrooms": 3,
      "bathrooms": 2,
      "year_built": 1998,
      "lot_size": 7500,
      "distance_to_city_center": 5.6,
      "school_rating": 8.2
    }
  }'
```

**Response:**
```json
{
  "predicted_price": 281761.16,
  "input_features": { ... }
}
```

### Predict (batch)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {"square_footage": 1550, "bedrooms": 3, "bathrooms": 2, "year_built": 1997, "lot_size": 6800, "distance_to_city_center": 4.1, "school_rating": 7.6},
      {"square_footage": 2200, "bedrooms": 4, "bathrooms": 2.5, "year_built": 2008, "lot_size": 9600, "distance_to_city_center": 7.0, "school_rating": 8.8}
    ]
  }'
```

### Model Info

```bash
curl http://localhost:8000/model-info
```

## Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9811 |
| **MAE** | $7,916 |
| **RMSE** | $10,277 |
| **Training samples** | 40 |

## Input Features

| Feature | Description | Example |
|---------|-------------|---------|
| `square_footage` | Total square footage | 1850 |
| `bedrooms` | Number of bedrooms | 3 |
| `bathrooms` | Number of bathrooms | 2 |
| `year_built` | Year built | 1998 |
| `lot_size` | Lot size (sq ft) | 7500 |
| `distance_to_city_center` | Distance to city center (miles) | 5.6 |
| `school_rating` | Nearby school rating (0-10) | 8.2 |

## Data Files

- `data/House Price Dataset.csv` — 50 training records with target `price`
- `data/Test Data For Prediction.csv` — 10 records for inference testing

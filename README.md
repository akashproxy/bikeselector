# BikeSelect — Indian Bikes Price Sensitivity

A machine learning project that classifies how sensitive Indian motorcycle buyers are to price changes, plus a **bike finder** that recommends the top 10 models for your budget and specs. It trains multiple classifiers on a 1,000-bike dataset and exposes everything through a **Flask web app** and a **CLI analysis script**.

## Features

- **Find Bike** — set minimum mileage, minimum CC, and maximum price; get the top 10 matching models ranked by fit
- **Add Bike** — submit a new bike with full CSV fields, save to the dataset, and compare with ML prediction and similar bikes
- **Manage Bikes** — search, edit, or remove bikes from the CSV dataset
- **Dashboard** — dataset stats, sensitivity distribution, and model leaderboard
- **Predict** — enter bike specs and get a price sensitivity label with probability breakdown
- **Browse Bikes** — search and filter the dataset by brand, segment, and sensitivity
- **REST API** — JSON endpoints for stats, bike search, selection, predictions, and leaderboard

## Project Structure

```
bikeselect/
├── app.py                          # Flask web application
├── bike_selector.py                # ML pipeline, preprocessing, training
├── indian_bikes_dataset_1000.csv   # Primary dataset (1000 bikes)
├── bikes.csv                         # Legacy small dataset (not used by default)
├── requirements.txt
├── templates/                      # HTML pages
│   ├── base.html
│   ├── index.html
│   ├── select.html
│   ├── add_bike.html
│   ├── predict.html
│   └── bikes.html
└── static/css/style.css
```

## Setup

```bash
pip install -r requirements.txt
```

**Note:** `matplotlib` and `seaborn` are only needed for CLI charts. The web app runs without them.

## Run the Web App

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

The first startup trains seven models (Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, Extra Trees, Gradient Boosting). This usually takes about a minute. The best model by accuracy is used for predictions.

To use a different CSV file:

```bash
set BIKE_CSV_PATH=your_data.csv
python app.py
```

## Run the CLI Analysis

For charts and a full terminal report:

```bash
python bike_selector.py
```

This loads `indian_bikes_dataset_1000.csv`, prints dataset summary, shows sensitivity and correlation plots, trains all models, and prints the leaderboard plus classification report.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stats` | Dataset summary and sensitivity counts |
| GET | `/api/leaderboard` | Model comparison metrics |
| GET | `/api/bikes` | Filter bikes (`q`, `brand`, `segment`, `sensitivity`, `limit`, `offset`) |
| POST | `/api/select` | Find top 10 bikes by mileage, CC, and max price (JSON body) |
| POST | `/api/bikes/add` | Add bike to CSV, predict sensitivity, compare with dataset (JSON body) |
| GET | `/api/bikes/<row_id>` | Get full bike details for editing |
| PUT | `/api/bikes/<row_id>` | Update bike in CSV (JSON body) |
| POST | `/api/bikes/remove` | Remove bike(s) from CSV by `row_ids` array (JSON body) |
| POST | `/api/predict` | Predict sensitivity from bike features (JSON body) |

### Example bike selection request

```bash
curl -X POST http://127.0.0.1:5000/api/select \
  -H "Content-Type: application/json" \
  -d "{\"min_mileage\":35,\"min_cc\":150,\"max_price\":200000}"
```

Response:

```json
{
  "count": 10,
  "filters": {
    "min_mileage": 35,
    "min_cc": 150,
    "max_price": 200000
  },
  "bikes": [
    {
      "brand": "Bajaj",
      "model": "CT100",
      "cc": 102,
      "mileage_kmpl": 78.1,
      "on_road_price_inr": 65925,
      "overall_score": 60,
      "match_score": 0.802,
      "price_sensitivity": "Very High"
    }
  ]
}
```

### Example prediction request

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"brand\":\"Honda\",\"segment\":\"mid\",\"speedometer_type\":\"Digital\",\"buyer_behaviour\":\"Might reconsider\",\"cc\":184,\"year\":2020,\"top_speed_kmh\":137,\"mileage_kmpl\":43.8,\"fuel_tank_liters\":10.1,\"factory_price_inr\":100781,\"gst_rate_pct\":28,\"on_road_price_inr\":147604,\"overall_score\":70,\"price_increase_scenario_pct\":9.3}"
```

Response:

```json
{
  "model": "Logistic Regression",
  "prediction": "Medium",
  "probabilities": {
    "High": 0.05,
    "Low": 0.00,
    "Medium": 0.81,
    "Very High": 0.13
  }
}
```

## How It Works

### 1. Preprocessing

Raw bike rows are enriched with engineered features: log prices, price per cc, speed per cc, segment flags, GST flag, and speedometer type flag.

### 2. Model Training

Features are split into categorical (brand, segment, speedometer type, buyer behaviour) and numerical columns. A `ColumnTransformer` imputes missing values, scales numerics, and one-hot encodes categoricals. Seven classifiers are trained with stratified 5-fold cross-validation.

### 3. Target

The target label is **price sensitivity**, one of:

- Low
- Medium
- High
- Very High

### 4. Prediction

The highest-accuracy model from the leaderboard is used to classify new bikes via the web form or API.

### 5. Bike Selection (Find Bike)

Hard filters: mileage ≥ minimum, engine CC ≥ minimum, on-road price ≤ maximum budget.

Each unique **brand + model** is scored on mileage (higher is better), price (lower is better), CC (closer to your minimum ranks higher), and overall score. The top 10 models are returned.

## Dataset

**File:** `indian_bikes_dataset_1000.csv` (1,000 rows)

**Columns:** `brand`, `model`, `cc`, `segment`, `year`, `speedometer_type`, `top_speed_kmh`, `mileage_kmpl`, `fuel_tank_liters`, `factory_price_inr`, `gst_rate_pct`, `gst_amount_inr`, `ex_showroom_inr`, `on_road_price_inr`, `overall_score`, `price_increase_scenario_pct`, `buyer_behaviour`, `price_sensitivity`

Segments: `budget`, `mid`, `premium`

## Dependencies

- pandas, numpy, scikit-learn — data and ML
- flask — web app
- matplotlib, seaborn — optional, for CLI visualizations only

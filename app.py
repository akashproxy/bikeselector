import os

from flask import Flask, jsonify, render_template, request

from bike_selector import (
    SENS_COLORS,
    SENS_ORDER,
    append_bike_to_csv,
    compare_bike_with_dataset,
    engineer_features,
    get_dataset_stats,
    get_form_options,
    load_and_train,
    load_dataset_only,
    parse_bike_input,
    get_bike_from_csv,
    remove_bikes_from_csv,
    select_bikes_to_records,
    update_bike_in_csv,
)

app = Flask(__name__)


@app.context_processor
def inject_globals():
    return {"stats": stats or {"rows": 0}}


CSV_PATH = os.environ.get(
    "BIKE_CSV_PATH",
    "indian_bikes_dataset_1000.csv",
)

df = None
ml = None
stats = None
form_options = None


def init_app():
    global df, ml, stats, form_options
    df, ml = load_and_train(CSV_PATH)
    refresh_stats()


def refresh_stats():
    global stats, form_options
    stats = get_dataset_stats(df)
    form_options = get_form_options(df)


def reload_data():
    """Reload CSV into memory after dataset changes."""
    global df
    df = load_dataset_only(CSV_PATH)
    refresh_stats()


def filter_bikes_dataframe(
    source_df,
    brand="",
    segment="",
    sensitivity="",
    search="",
):
    filtered = source_df.copy()
    if brand:
        filtered = filtered[filtered["brand"] == brand]
    if segment:
        filtered = filtered[filtered["segment"] == segment]
    if sensitivity:
        filtered = filtered[filtered["price_sensitivity"] == sensitivity]
    if search:
        mask = (
            filtered["brand"].str.lower().str.contains(search, na=False)
            | filtered["model"].str.lower().str.contains(search, na=False)
        )
        filtered = filtered[mask]
    return filtered


@app.route("/")
def index():
    leaderboard = ml.leaderboard().to_dict(orient="records")
    best_name, best = ml.best_model()
    return render_template(
        "index.html",
        stats=stats,
        leaderboard=leaderboard,
        best_model=best_name,
        best_accuracy=round(best["accuracy"] * 100, 1),
        sens_order=SENS_ORDER,
        sens_colors=SENS_COLORS,
    )


@app.route("/predict")
def predict_page():
    return render_template(
        "predict.html",
        options=form_options,
        sens_colors=SENS_COLORS,
    )


@app.route("/select")
def select_page():
    return render_template(
        "select.html",
        defaults={
            "min_mileage": 35,
            "min_cc": 150,
            "max_price": 200000,
        },
        sens_colors=SENS_COLORS,
        price_max=int(df["on_road_price_inr"].max()),
        mileage_max=float(df["mileage_kmpl"].max()),
    )


@app.route("/api/select", methods=["POST"])
def api_select():
    data = request.get_json(silent=True) or request.form
    try:
        min_mileage = float(data["min_mileage"])
        min_cc = float(data.get("min_cc", data.get("preferred_cc")))
        max_price = float(data["max_price"])
        top_n = min(int(data.get("top_n", 10)), 10)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    if min_mileage < 0 or min_cc <= 0 or max_price <= 0:
        return jsonify({"error": "Values must be positive"}), 400

    results = select_bikes_to_records(
        df,
        min_mileage=min_mileage,
        min_cc=min_cc,
        max_price=max_price,
        top_n=top_n,
    )
    return jsonify({
        "count": len(results),
        "filters": {
            "min_mileage": min_mileage,
            "min_cc": min_cc,
            "max_price": max_price,
        },
        "bikes": results,
    })


@app.route("/add-bike")
def add_bike_page():
    return render_template(
        "add_bike.html",
        options=form_options,
        sens_colors=SENS_COLORS,
        sens_order=SENS_ORDER,
    )


@app.route("/api/bikes/add", methods=["POST"])
def api_add_bike():
    data = request.get_json(silent=True) or request.form
    try:
        row = parse_bike_input(data)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    if not row["brand"] or not row["model"]:
        return jsonify({"error": "Brand and model are required"}), 400

    if row["segment"] not in ("budget", "mid", "premium"):
        return jsonify({"error": "Segment must be budget, mid, or premium"}), 400

    comparison = compare_bike_with_dataset(df, row, ml)
    row["price_sensitivity"] = comparison["bike"]["price_sensitivity"]

    try:
        append_bike_to_csv(CSV_PATH, row)
    except OSError as exc:
        return jsonify({"error": f"Could not save to CSV: {exc}"}), 500

    reload_data()
    comparison["saved"] = True
    comparison["message"] = (
        f"{row['brand']} {row['model']} added to dataset "
        f"({stats['rows']} bikes total)."
    )
    return jsonify(comparison)


@app.route("/manage-bikes")
@app.route("/remove-bike")
def manage_bikes_page():
    return render_template(
        "manage_bikes.html",
        total=len(df),
        options=form_options,
        sens_colors=SENS_COLORS,
        sens_order=SENS_ORDER,
        brands=sorted(df["brand"].unique().tolist()),
    )


@app.route("/api/bikes/<int:row_id>", methods=["GET"])
def api_get_bike(row_id):
    bike = get_bike_from_csv(CSV_PATH, row_id)
    if bike is None:
        return jsonify({"error": "Bike not found"}), 404
    return jsonify(bike)


@app.route("/api/bikes/<int:row_id>", methods=["PUT"])
def api_update_bike(row_id):
    data = request.get_json(silent=True) or request.form
    if get_bike_from_csv(CSV_PATH, row_id) is None:
        return jsonify({"error": "Bike not found"}), 404

    try:
        row = parse_bike_input(data)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    if not row["brand"] or not row["model"]:
        return jsonify({"error": "Brand and model are required"}), 400

    if row["segment"] not in ("budget", "mid", "premium"):
        return jsonify({"error": "Segment must be budget, mid, or premium"}), 400

    if not row.get("price_sensitivity"):
        features = engineer_features(row)
        row["price_sensitivity"] = ml.predict_row(features)["prediction"]

    try:
        update_bike_in_csv(CSV_PATH, row_id, row)
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 500

    reload_data()
    return jsonify({
        "row_id": row_id,
        "bike": row,
        "message": f"Updated {row['brand']} {row['model']} successfully.",
    })


@app.route("/api/bikes/remove", methods=["POST"])
def api_remove_bikes():
    data = request.get_json(silent=True) or request.form
    row_ids = data.get("row_ids") or data.get("row_id")

    if row_ids is None:
        return jsonify({"error": "row_ids required"}), 400

    if not isinstance(row_ids, list):
        row_ids = [row_ids]

    try:
        row_ids = [int(i) for i in row_ids]
    except (TypeError, ValueError):
        return jsonify({"error": "row_ids must be integers"}), 400

    if not row_ids:
        return jsonify({"error": "No valid rows selected"}), 400

    try:
        removed = remove_bikes_from_csv(CSV_PATH, row_ids)
    except OSError as exc:
        return jsonify({"error": f"Could not update CSV: {exc}"}), 500

    if not removed:
        return jsonify({"error": "No matching bikes found to remove"}), 404

    reload_data()

    names = [
        f"{r['brand']} {r['model']} ({int(r['year'])})"
        for r in removed
    ]
    return jsonify({
        "removed_count": len(removed),
        "removed": removed,
        "total_remaining": stats["rows"],
        "message": (
            f"Removed {len(removed)} bike(s): {', '.join(names)}. "
            f"{stats['rows']} bikes remaining."
        ),
    })


@app.route("/bikes")
def bikes_page():
    return render_template(
        "bikes.html",
        bikes=df.head(100).to_dict(orient="records"),
        total=len(df),
        sens_colors=SENS_COLORS,
        brands=sorted(df["brand"].unique().tolist()),
    )


@app.route("/api/bikes")
def api_bikes():
    brand = request.args.get("brand", "").strip()
    segment = request.args.get("segment", "").strip()
    sensitivity = request.args.get("sensitivity", "").strip()
    search = request.args.get("q", "").strip().lower()
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))

    filtered = filter_bikes_dataframe(
        df, brand, segment, sensitivity, search
    )

    total = len(filtered)
    page = filtered.iloc[offset : offset + limit]
    bikes = []
    for idx, row in page.iterrows():
        bikes.append({
            "row_id": int(idx),
            "brand": row["brand"],
            "model": row["model"],
            "cc": float(row["cc"]),
            "year": int(row["year"]),
            "segment": row["segment"],
            "mileage_kmpl": float(row["mileage_kmpl"]),
            "on_road_price_inr": int(row["on_road_price_inr"]),
            "overall_score": float(row["overall_score"]),
            "price_sensitivity": row["price_sensitivity"],
        })
    return jsonify({
        "total": total,
        "offset": offset,
        "limit": limit,
        "bikes": bikes,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or request.form

    try:
        row = {
            "brand": data["brand"],
            "segment": data["segment"],
            "speedometer_type": data["speedometer_type"],
            "buyer_behaviour": data["buyer_behaviour"],
            "cc": float(data["cc"]),
            "year": int(data["year"]),
            "top_speed_kmh": float(data["top_speed_kmh"]),
            "mileage_kmpl": float(data["mileage_kmpl"]),
            "fuel_tank_liters": float(data["fuel_tank_liters"]),
            "factory_price_inr": float(data["factory_price_inr"]),
            "gst_rate_pct": float(data["gst_rate_pct"]),
            "on_road_price_inr": float(data["on_road_price_inr"]),
            "overall_score": float(data["overall_score"]),
            "price_increase_scenario_pct": float(
                data["price_increase_scenario_pct"]
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    features = engineer_features(row)
    result = ml.predict_row(features)
    return jsonify(result)


@app.route("/api/stats")
def api_stats():
    return jsonify(stats)


@app.route("/api/leaderboard")
def api_leaderboard():
    records = ml.leaderboard().to_dict(orient="records")
    for row in records:
        for key in ("Accuracy", "F1 Macro", "ROC-AUC", "CV Mean", "CV Std"):
            if key in row and row[key] is not None:
                row[key] = round(float(row[key]), 4)
    return jsonify(records)


_initialized = False


def _startup():
    global _initialized
    if _initialized:
        return
    print("Training models (first startup may take a minute)...")
    init_app()
    _initialized = True
    print(f"Ready. Best model: {ml.best_model()[0]}")


_startup()


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)

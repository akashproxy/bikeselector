# =========================================================
# Indian Bikes - Price Sensitivity Classification System
# Clean End-to-End ML Pipeline
# =========================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB


C1 = "#2980b9"
C2 = "#e74c3c"
C3 = "#f39c12"

SEG_COLORS = {
    "budget": "#27ae60",
    "mid": "#f39c12",
    "premium": "#e74c3c"
}

SENS_COLORS = {
    "Low": "#10B981",
    "Medium": "#3B82F6",
    "High": "#F59E0B",
    "Very High": "#EF4444",
}

SENS_ORDER = [
    "Low",
    "Medium",
    "High",
    "Very High"
]


# =========================================================
# Dataset Manager
# =========================================================

class BikeDataset:

    def __init__(self, csv_path: str):

        self.df = pd.read_csv(csv_path)

    def preprocess(self):

        df = self.df.copy()

        # -----------------------------------------
        # Feature Engineering
        # -----------------------------------------

        df["log_price"] = np.log1p(
            df["on_road_price_inr"]
        )

        df["log_factory"] = np.log1p(
            df["factory_price_inr"]
        )

        df["price_per_cc"] = (
            df["on_road_price_inr"] /
            df["cc"]
        )

        df["speed_per_cc"] = (
            df["top_speed_kmh"] /
            df["cc"]
        )

        df["is_premium"] = (
            df["segment"] == "premium"
        ).astype(int)

        df["is_budget"] = (
            df["segment"] == "budget"
        ).astype(int)

        df["is_high_gst"] = (
            df["gst_rate_pct"] == 31
        ).astype(int)

        df["is_digital_speedo"] = (
            df["speedometer_type"] == "Digital"
        ).astype(int)

        self.df = df

        return df

    def dataset_summary(self):

        df = self.df

        print("\nDataset Summary\n")

        print(f"Rows          : {df.shape[0]:,}")
        print(f"Columns       : {df.shape[1]}")
        print(f"Brands        : {df['brand'].nunique()}")
        print(f"Models        : {df['model'].nunique()}")

        print(
            f"Price Range   : "
            f"INR {df['on_road_price_inr'].min():,}"
            f" - "
            f"INR {df['on_road_price_inr'].max():,}"
        )

        print(
            f"CC Range      : "
            f"{df['cc'].min()} - {df['cc'].max()}"
        )

        print(
            f"Year Range    : "
            f"{df['year'].min()} - {df['year'].max()}"
        )

    def check_missing_values(self):

        missing = self.df.isnull().sum()

        print("\nMissing Values\n")

        if missing.sum() == 0:
            print("No missing values found.")
        else:
            print(missing[missing > 0])


# =========================================================
# Visualization Layer
# =========================================================

class BikeVisualizer:

    def __init__(self, df):

        self.df = df

    def _ensure_plt(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.rcParams.update({
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        })
        return plt, sns

    def plot_price_sensitivity(self):

        plt, _ = self._ensure_plt()
        df = self.df

        counts = (
            df["price_sensitivity"]
            .value_counts()
            .reindex(SENS_ORDER)
        )

        colors = [
            SENS_COLORS[s]
            for s in SENS_ORDER
        ]

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 5)
        )

        # Pie Chart

        axes[0].pie(
            counts.values,
            labels=SENS_ORDER,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.6)
        )

        axes[0].set_title(
            "Price Sensitivity Distribution",
            fontweight="bold"
        )

        # Bar Chart

        bars = axes[1].bar(
            SENS_ORDER,
            counts.values,
            color=colors,
            edgecolor="white"
        )

        for bar, val in zip(bars, counts.values):

            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{val}",
                ha="center",
                fontsize=9,
                fontweight="bold"
            )

        axes[1].set_title(
            "Count by Price Sensitivity",
            fontweight="bold"
        )

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):

        plt, sns = self._ensure_plt()
        numeric_df = self.df.select_dtypes(
            include=np.number
        )

        corr = numeric_df.corr()

        plt.figure(figsize=(12, 10))

        mask = np.triu(
            np.ones_like(corr, dtype=bool)
        )

        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            annot=False,
            linewidths=0.5
        )

        plt.title(
            "Feature Correlation Matrix",
            fontweight="bold"
        )

        plt.tight_layout()
        plt.show()


# =========================================================
# ML Pipeline
# =========================================================

@dataclass
class MLConfig:

    target_col: str = "price_sensitivity"

    test_size: float = 0.2

    random_state: int = 42


class BikeMLPipeline:

    def __init__(
        self,
        df,
        config: MLConfig
    ):

        self.df = df

        self.config = config

        self.results = {}

    def prepare_data(self):

        df = self.df.copy()

        target = self.config.target_col

        # -----------------------------------------
        # Feature Columns
        # -----------------------------------------

        categorical_features = [
            "brand",
            "segment",
            "speedometer_type",
            "buyer_behaviour"
        ]

        numerical_features = [
            "cc",
            "year",
            "top_speed_kmh",
            "mileage_kmpl",
            "fuel_tank_liters",
            "factory_price_inr",
            "gst_rate_pct",
            "on_road_price_inr",
            "overall_score",
            "price_increase_scenario_pct",
            "log_price",
            "log_factory",
            "price_per_cc",
            "speed_per_cc",
            "is_premium",
            "is_budget",
            "is_high_gst",
            "is_digital_speedo"
        ]

        X = df[
            categorical_features +
            numerical_features
        ]

        y = df[target]

        # -----------------------------------------
        # Train/Test Split
        # -----------------------------------------

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        # -----------------------------------------
        # Preprocessor
        # -----------------------------------------

        numeric_transformer = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ])

        categorical_transformer = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ])

        self.preprocessor = ColumnTransformer([
            (
                "num",
                numeric_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ])

    def build_models(self):

        self.models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ),

            "KNN":
                KNeighborsClassifier(
                    n_neighbors=7
                ),

            "Naive Bayes":
                GaussianNB(),

            "Decision Tree":
                DecisionTreeClassifier(
                    max_depth=10,
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                ),

            "Extra Trees":
                ExtraTreesClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                ),

            "Gradient Boosting":
                GradientBoostingClassifier(
                    n_estimators=200,
                    random_state=42
                )
        }

    def train_and_evaluate(self):

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        for name, model in self.models.items():

            if name == "Naive Bayes":

                pipeline = Pipeline([
                    (
                        "preprocessor",
                        self.preprocessor
                    ),
                    (
                        "to_dense",
                        FunctionTransformer(
                            lambda x: (
                                x.toarray()
                                if hasattr(x, "toarray")
                                else x
                            ),
                            accept_sparse=True,
                        )
                    ),
                    (
                        "model",
                        model
                    )
                ])

            else:

                pipeline = Pipeline([
                    (
                        "preprocessor",
                        self.preprocessor
                    ),
                    (
                        "model",
                        model
                    )
                ])

            pipeline.fit(
                self.X_train,
                self.y_train
            )

            preds = pipeline.predict(
                self.X_test
            )

            acc = accuracy_score(
                self.y_test,
                preds
            )

            f1 = f1_score(
                self.y_test,
                preds,
                average="macro"
            )

            try:

                probs = pipeline.predict_proba(
                    self.X_test
                )

                roc = roc_auc_score(
                    self.y_test,
                    probs,
                    multi_class="ovr",
                    average="weighted"
                )

            except Exception:

                roc = np.nan

            cv_scores = cross_val_score(
                pipeline,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1
            )

            self.results[name] = {
                "model": pipeline,
                "accuracy": acc,
                "f1": f1,
                "roc_auc": roc,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "predictions": preds
            }

    def leaderboard(self):

        leaderboard_df = pd.DataFrame([

            {
                "Model": name,
                "Accuracy": res["accuracy"],
                "F1 Macro": res["f1"],
                "ROC-AUC": res["roc_auc"],
                "CV Mean": res["cv_mean"],
                "CV Std": res["cv_std"]
            }

            for name, res in self.results.items()

        ])

        leaderboard_df = leaderboard_df.sort_values(
            "Accuracy",
            ascending=False
        )

        return leaderboard_df

    def best_model(self):

        best_name = max(
            self.results,
            key=lambda k: self.results[k]["accuracy"]
        )

        return (
            best_name,
            self.results[best_name]
        )

    def plot_dashboard(self):

        plt, _ = self._ensure_plt()
        model_names = list(self.results.keys())

        accuracies = [
            self.results[m]["accuracy"]
            for m in model_names
        ]

        f1_scores = [
            self.results[m]["f1"]
            for m in model_names
        ]

        roc_aucs = [
            self.results[m]["roc_auc"]
            for m in model_names
        ]

        cv_means = [
            self.results[m]["cv_mean"]
            for m in model_names
        ]

        cv_stds = [
            self.results[m]["cv_std"]
            for m in model_names
        ]

        x = np.arange(len(model_names))

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(16, 10)
        )

        # Accuracy

        bars = axes[0, 0].bar(
            x,
            accuracies,
            color=C1
        )

        axes[0, 0].set_title(
            "Accuracy",
            fontweight="bold"
        )

        axes[0, 0].set_xticks(x)

        axes[0, 0].set_xticklabels(
            model_names,
            rotation=15
        )

        # F1

        bars2 = axes[0, 1].bar(
            x,
            f1_scores,
            color=C2
        )

        axes[0, 1].set_title(
            "F1 Macro",
            fontweight="bold"
        )

        axes[0, 1].set_xticks(x)

        axes[0, 1].set_xticklabels(
            model_names,
            rotation=15
        )

        # ROC-AUC

        roc_vals = [
            0 if np.isnan(r) else r
            for r in roc_aucs
        ]

        bars3 = axes[1, 0].bar(
            x,
            roc_vals,
            color=C3
        )

        axes[1, 0].set_title(
            "ROC-AUC",
            fontweight="bold"
        )

        axes[1, 0].set_xticks(x)

        axes[1, 0].set_xticklabels(
            model_names,
            rotation=15
        )

        # CV

        axes[1, 1].bar(
            x,
            cv_means,
            yerr=cv_stds,
            capsize=5,
            color="#2c3e50"
        )

        axes[1, 1].set_title(
            "Cross Validation Accuracy",
            fontweight="bold"
        )

        axes[1, 1].set_xticks(x)

        axes[1, 1].set_xticklabels(
            model_names,
            rotation=15
        )

        plt.suptitle(
            "Model Performance Dashboard",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout()

        plt.show()

    def plot_confusion_matrix(self):

        plt, _ = self._ensure_plt()
        best_name, best = self.best_model()

        cm = confusion_matrix(
            self.y_test,
            best["predictions"]
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ConfusionMatrixDisplay(
            cm
        ).plot(
            ax=ax,
            cmap="Blues"
        )

        ax.set_title(
            f"Confusion Matrix - {best_name}",
            fontweight="bold"
        )

        plt.tight_layout()

        plt.show()

    def classification_report(self):

        best_name, best = self.best_model()

        print("\nBest Model\n")

        print(best_name)

        print("\nClassification Report\n")

        print(
            classification_report(
                self.y_test,
                best["predictions"]
            )
        )

    def predict_row(self, row: dict) -> dict:
        """Predict price sensitivity for a single bike feature dict."""
        best_name, best = self.best_model()
        pipeline = best["model"]
        df = pd.DataFrame([row])
        pred = pipeline.predict(df)[0]
        proba = pipeline.predict_proba(df)[0]
        classes = pipeline.named_steps["model"].classes_
        return {
            "model": best_name,
            "prediction": pred,
            "probabilities": {
                cls: float(p) for cls, p in zip(classes, proba)
            },
        }


CATEGORICAL_FEATURES = [
    "brand",
    "segment",
    "speedometer_type",
    "buyer_behaviour",
]

NUMERICAL_FEATURES = [
    "cc",
    "year",
    "top_speed_kmh",
    "mileage_kmpl",
    "fuel_tank_liters",
    "factory_price_inr",
    "gst_rate_pct",
    "on_road_price_inr",
    "overall_score",
    "price_increase_scenario_pct",
    "log_price",
    "log_factory",
    "price_per_cc",
    "speed_per_cc",
    "is_premium",
    "is_budget",
    "is_high_gst",
    "is_digital_speedo",
]


def engineer_features(row: dict) -> dict:
    """Add engineered columns required by the ML pipeline."""
    out = dict(row)
    out["log_price"] = float(np.log1p(row["on_road_price_inr"]))
    out["log_factory"] = float(np.log1p(row["factory_price_inr"]))
    out["price_per_cc"] = row["on_road_price_inr"] / row["cc"]
    out["speed_per_cc"] = row["top_speed_kmh"] / row["cc"]
    out["is_premium"] = int(row["segment"] == "premium")
    out["is_budget"] = int(row["segment"] == "budget")
    out["is_high_gst"] = int(row["gst_rate_pct"] == 31)
    out["is_digital_speedo"] = int(row["speedometer_type"] == "Digital")
    return out


def get_dataset_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "brands": int(df["brand"].nunique()),
        "models": int(df["model"].nunique()),
        "price_min": int(df["on_road_price_inr"].min()),
        "price_max": int(df["on_road_price_inr"].max()),
        "cc_min": int(df["cc"].min()),
        "cc_max": int(df["cc"].max()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "sensitivity_counts": (
            df["price_sensitivity"]
            .value_counts()
            .reindex(SENS_ORDER)
            .fillna(0)
            .astype(int)
            .to_dict()
        ),
    }


CSV_COLUMNS = [
    "brand",
    "model",
    "cc",
    "segment",
    "year",
    "speedometer_type",
    "top_speed_kmh",
    "mileage_kmpl",
    "fuel_tank_liters",
    "factory_price_inr",
    "gst_rate_pct",
    "gst_amount_inr",
    "ex_showroom_inr",
    "on_road_price_inr",
    "overall_score",
    "price_increase_scenario_pct",
    "buyer_behaviour",
    "price_sensitivity",
]


def parse_bike_input(data: dict) -> dict:
    """Validate form/API input and build a complete CSV row."""
    factory = float(data["factory_price_inr"])
    gst_rate = float(data["gst_rate_pct"])
    gst_amount = round(
        float(data["gst_amount_inr"])
        if data.get("gst_amount_inr")
        else factory * gst_rate / 100
    )
    ex_showroom = round(
        float(data["ex_showroom_inr"])
        if data.get("ex_showroom_inr")
        else factory + gst_amount
    )

    return {
        "brand": str(data["brand"]).strip(),
        "model": str(data["model"]).strip(),
        "cc": float(data["cc"]),
        "segment": str(data["segment"]).strip().lower(),
        "year": int(data["year"]),
        "speedometer_type": str(data["speedometer_type"]).strip(),
        "top_speed_kmh": float(data["top_speed_kmh"]),
        "mileage_kmpl": float(data["mileage_kmpl"]),
        "fuel_tank_liters": float(data["fuel_tank_liters"]),
        "factory_price_inr": factory,
        "gst_rate_pct": gst_rate,
        "gst_amount_inr": gst_amount,
        "ex_showroom_inr": ex_showroom,
        "on_road_price_inr": float(data["on_road_price_inr"]),
        "overall_score": float(data["overall_score"]),
        "price_increase_scenario_pct": float(
            data["price_increase_scenario_pct"]
        ),
        "buyer_behaviour": str(data["buyer_behaviour"]).strip(),
        "price_sensitivity": str(data.get("price_sensitivity", "")).strip(),
    }


def append_bike_to_csv(csv_path: str, row: dict) -> None:
    """Append one bike row to the dataset CSV."""
    new_df = pd.DataFrame([row], columns=CSV_COLUMNS)
    header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    new_df.to_csv(csv_path, mode="a", header=header, index=False)


def get_bike_from_csv(csv_path: str, row_id: int) -> dict | None:
    """Get one bike row from raw CSV by index."""
    raw = pd.read_csv(csv_path)
    if row_id < 0 or row_id >= len(raw):
        return None
    row = raw.iloc[row_id]
    result = {"row_id": int(row_id)}
    for col in CSV_COLUMNS:
        val = row[col]
        if col in ("on_road_price_inr", "gst_amount_inr", "ex_showroom_inr",
                    "factory_price_inr", "year", "overall_score"):
            result[col] = int(val) if col != "overall_score" else float(val)
        elif col in ("cc", "top_speed_kmh", "mileage_kmpl", "fuel_tank_liters",
                      "gst_rate_pct", "price_increase_scenario_pct"):
            result[col] = float(val)
        else:
            result[col] = val if isinstance(val, str) else str(val)
    return result


def update_bike_in_csv(csv_path: str, row_id: int, row: dict) -> dict:
    """Update one bike row in the CSV by index."""
    raw = pd.read_csv(csv_path)
    if row_id < 0 or row_id >= len(raw):
        raise ValueError(f"Invalid row_id: {row_id}")
    for col in CSV_COLUMNS:
        raw.at[row_id, col] = row[col]
    raw.to_csv(csv_path, index=False)
    return row


def remove_bikes_from_csv(csv_path: str, row_ids: list[int]) -> list[dict]:
    """Remove bikes by CSV row index. Returns removed row dicts."""
    raw = pd.read_csv(csv_path)
    valid_ids = sorted(
        {int(i) for i in row_ids if 0 <= int(i) < len(raw)},
        reverse=True,
    )
    if not valid_ids:
        return []

    removed_df = raw.iloc[valid_ids].copy()
    raw = raw.drop(index=valid_ids).reset_index(drop=True)
    raw.to_csv(csv_path, index=False)

    removed = []
    for _, row in removed_df.iterrows():
        item = {col: row[col] for col in CSV_COLUMNS if col in row}
        if "on_road_price_inr" in item:
            item["on_road_price_inr"] = int(item["on_road_price_inr"])
        removed.append(item)
    return removed


def bike_records_with_ids(
    df: pd.DataFrame,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Return bike rows with row_id for list/delete UI."""
    page = df.iloc[offset : offset + limit]
    records = []
    for idx, row in page.iterrows():
        records.append({
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
    return records, len(df)


def load_dataset_only(csv_path: str) -> pd.DataFrame:
    """Load and preprocess CSV without retraining models."""
    dataset = BikeDataset(csv_path)
    dataset.preprocess()
    return dataset.df


def _percentile_rank(series: pd.Series, value: float) -> float:
    return round(float((series <= value).mean() * 100), 1)


def find_similar_bikes(
    df: pd.DataFrame,
    row: dict,
    top_n: int = 5,
) -> list[dict]:
    """Find closest bikes by normalized spec distance."""
    features = [
        "cc",
        "mileage_kmpl",
        "on_road_price_inr",
        "overall_score",
        "top_speed_kmh",
    ]
    work = df[features].astype(float)
    new_vec = np.array([[row[f] for f in features]], dtype=float)

    mins = work.min()
    maxs = work.max()
    span = (maxs - mins).replace(0, 1)
    norm = (work - mins) / span
    new_norm = (new_vec - mins.values) / span.values

    dist = np.sqrt(((norm - new_norm) ** 2).sum(axis=1))
    work_df = df.copy()
    work_df["_dist"] = dist
    best = (
        work_df.sort_values("_dist")
        .drop_duplicates(subset=["brand", "model"])
        .head(top_n)
    )

    cols = [
        "brand", "model", "cc", "segment", "mileage_kmpl",
        "on_road_price_inr", "overall_score", "price_sensitivity", "_dist",
    ]
    out = best[cols].copy()
    max_dist = out["_dist"].max() or 1.0
    out["similarity_pct"] = (100 * (1 - out["_dist"] / max_dist)).round(1)
    out = out.drop(columns=["_dist"])
    out["on_road_price_inr"] = out["on_road_price_inr"].astype(int)
    return out.to_dict(orient="records")


def compare_bike_with_dataset(
    df: pd.DataFrame,
    row: dict,
    ml: "BikeMLPipeline",
    top_similar: int = 5,
) -> dict:
    """Predict sensitivity and compare new bike against the dataset."""
    features = engineer_features(row)
    prediction = ml.predict_row(features)

    if not row.get("price_sensitivity"):
        row["price_sensitivity"] = prediction["prediction"]

    comparison = {
        "price_inr": {
            "value": int(row["on_road_price_inr"]),
            "percentile": _percentile_rank(
                df["on_road_price_inr"], row["on_road_price_inr"]
            ),
            "dataset_avg": int(df["on_road_price_inr"].mean()),
        },
        "mileage_kmpl": {
            "value": row["mileage_kmpl"],
            "percentile": _percentile_rank(
                df["mileage_kmpl"], row["mileage_kmpl"]
            ),
            "dataset_avg": round(float(df["mileage_kmpl"].mean()), 1),
        },
        "cc": {
            "value": row["cc"],
            "percentile": _percentile_rank(df["cc"], row["cc"]),
            "dataset_avg": round(float(df["cc"].mean()), 1),
        },
        "overall_score": {
            "value": row["overall_score"],
            "percentile": _percentile_rank(
                df["overall_score"], row["overall_score"]
            ),
            "dataset_avg": round(float(df["overall_score"].mean()), 1),
        },
    }

    segment_count = int((df["segment"] == row["segment"]).sum())
    same_segment = df[df["segment"] == row["segment"]]
    segment_avg_price = (
        int(same_segment["on_road_price_inr"].mean())
        if len(same_segment)
        else None
    )

    return {
        "bike": row,
        "prediction": prediction,
        "comparison": comparison,
        "segment": {
            "name": row["segment"],
            "count_in_dataset": segment_count,
            "avg_price_inr": segment_avg_price,
        },
        "similar_bikes": find_similar_bikes(df, row, top_similar),
        "dataset_size": len(df),
    }


def get_form_options(df: pd.DataFrame) -> dict:
    return {
        "brands": sorted(df["brand"].unique().tolist()),
        "segments": sorted(df["segment"].unique().tolist()),
        "speedometer_types": sorted(df["speedometer_type"].unique().tolist()),
        "buyer_behaviours": sorted(df["buyer_behaviour"].unique().tolist()),
        "gst_rates": sorted(df["gst_rate_pct"].unique().tolist()),
    }


def select_bikes(
    df: pd.DataFrame,
    min_mileage: float,
    min_cc: float,
    max_price: float,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Filter bikes by minimum mileage, minimum CC, and maximum price;
    return top N unique models (one row per brand+model).
    """
    filtered = df[
        (df["mileage_kmpl"] >= min_mileage)
        & (df["cc"] >= min_cc)
        & (df["on_road_price_inr"] <= max_price)
    ].copy()

    if filtered.empty:
        return filtered

    # Among eligible bikes, prefer higher mileage, lower price, CC closer to minimum
    mileage_range = filtered["mileage_kmpl"].max() - filtered["mileage_kmpl"].min()
    price_range = filtered["on_road_price_inr"].max() - filtered["on_road_price_inr"].min()
    cc_range = filtered["cc"].max() - filtered["cc"].min()

    norm_mileage = (
        (filtered["mileage_kmpl"] - filtered["mileage_kmpl"].min()) / mileage_range
        if mileage_range > 0
        else 1.0
    )
    norm_price = (
        1 - (filtered["on_road_price_inr"] - filtered["on_road_price_inr"].min()) / price_range
        if price_range > 0
        else 1.0
    )
    norm_cc = (
        1 - (filtered["cc"] - filtered["cc"].min()) / cc_range
        if cc_range > 0
        else 1.0
    )
    norm_score = filtered["overall_score"] / 100.0

    filtered["match_score"] = (
        0.35 * norm_mileage
        + 0.30 * norm_price
        + 0.20 * norm_cc
        + 0.15 * norm_score
    )

    best_idx = filtered.groupby(["brand", "model"])["match_score"].idxmax()
    results = (
        filtered.loc[best_idx]
        .sort_values("match_score", ascending=False)
        .head(top_n)
    )
    return results


def select_bikes_to_records(
    df: pd.DataFrame,
    min_mileage: float,
    min_cc: float,
    max_price: float,
    top_n: int = 10,
) -> list[dict]:
    """Return top bike matches as JSON-serializable dicts."""
    results = select_bikes(
        df, min_mileage, min_cc, max_price, top_n
    )
    if results.empty:
        return []

    cols = [
        "brand", "model", "cc", "year", "segment",
        "mileage_kmpl", "on_road_price_inr", "overall_score",
        "top_speed_kmh", "price_sensitivity", "match_score",
    ]
    out = results[cols].copy()
    out["match_score"] = out["match_score"].round(3)
    out["on_road_price_inr"] = out["on_road_price_inr"].astype(int)
    return out.to_dict(orient="records")


def load_and_train(csv_path: str = "indian_bikes_dataset_1000.csv") -> tuple:
    """Load data, preprocess, train models, return (df, ml_pipeline)."""
    dataset = BikeDataset(csv_path)
    dataset.preprocess()
    df = dataset.df
    config = MLConfig()
    ml = BikeMLPipeline(df, config)
    ml.prepare_data()
    ml.build_models()
    ml.train_and_evaluate()
    return df, ml


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Load Dataset
    # -----------------------------------------------------

    dataset = BikeDataset(
        "indian_bikes_dataset_1000.csv"
    )

    dataset.preprocess()

    dataset.dataset_summary()

    dataset.check_missing_values()

    df = dataset.df

    # -----------------------------------------------------
    # Visualizations
    # -----------------------------------------------------

    visualizer = BikeVisualizer(df)

    visualizer.plot_price_sensitivity()

    visualizer.plot_correlation_heatmap()

    # -----------------------------------------------------
    # ML Pipeline
    # -----------------------------------------------------

    config = MLConfig()

    ml = BikeMLPipeline(
        df,
        config
    )

    ml.prepare_data()

    ml.build_models()

    ml.train_and_evaluate()

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\nLeaderboard\n")

    print(
        ml.leaderboard()
    )

    ml.plot_dashboard()

    ml.plot_confusion_matrix()

    ml.classification_report()
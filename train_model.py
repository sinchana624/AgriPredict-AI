
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Paths
DATA_PATH = Path(
    "data/processed/exploratory_model_dataset_for_handoff.csv"
)
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

TARGET = "yield_tonnes_per_hectare"

# Initial exploratory features.
# Production and harvested area are deliberately excluded.
FEATURES = [
    "year",
    "crop",
    "rainfall_ann",
    "tmax_ann",
    "tmean_ann",
    "tmin_ann",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [column for column in required
               if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()
    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="coerce"
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET])
    df = df.sort_values("year").reset_index(drop=True)

    # Chronological split: older years train; newer years test.
    train_df = df[df["year"] <= 2018].copy()
    test_df = df[df["year"] >= 2019].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or test set is empty.")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    numeric_features = [
        "year",
        "rainfall_ann",
        "tmax_ann",
        "tmean_ann",
        "tmin_ann",
    ]
    categorical_features = ["crop"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    )
                ]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                    ),
                ]),
                categorical_features,
            ),
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    results = []
    fitted_models = {}

    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print(
        "Training years:",
        train_df["year"].min(),
        "to",
        train_df["year"].max(),
    )
    print(
        "Test years:",
        test_df["year"].min(),
        "to",
        test_df["year"].max(),
    )

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        fitted_models[name] = pipeline

        print(f"\n{name}")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")

    results_df = pd.DataFrame(results).sort_values("RMSE")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        REPORT_DIR / "model_metrics.csv",
        index=False,
    )

    # Select by lowest RMSE on the held-out years.
    best_name = results_df.iloc[0]["Model"]
    best_pipeline = fitted_models[best_name]

    joblib.dump(
        best_pipeline,
        MODEL_DIR / "yield_model.joblib",
    )

    print("\n===== MODEL COMPARISON =====")
    print(results_df.to_string(index=False))
    print("\nSelected model:", best_name)
    print("Saved model:", MODEL_DIR / "yield_model.joblib")
    print("Saved metrics:", REPORT_DIR / "model_metrics.csv")

    print(
        "\nIMPORTANT: This is an exploratory retrospective "
        "evaluation, not validation for nationwide or "
        "pre-season crop-yield predictions."
    )


if __name__ == "__main__":
    main()

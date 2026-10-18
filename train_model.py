
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
ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    ROOT / "data" / "processed"
    / "exploratory_model_dataset_for_handoff.csv"
)
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "yield_tonnes_per_hectare"

# Six model input features.
# Production and harvested area are deliberately excluded.
FEATURES = [
    "year",
    "crop",
    "rainfall_ann",
    "tmax_ann",
    "tmean_ann",
    "tmin_ann",
]

NUMERIC_FEATURES = [
    "year",
    "rainfall_ann",
    "tmax_ann",
    "tmean_ann",
    "tmin_ann",
]
CATEGORICAL_FEATURES = ["crop"]


def make_pipeline(estimator):
    """Create a preprocessing and regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def calculate_metrics(y_true, predictions):
    """Calculate regression evaluation metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, predictions))
        ),
        "R2": float(r2_score(y_true, predictions)),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()

    # Validate and clean input columns.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["crop"] = df["crop"].astype("string").str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["year", TARGET, "crop"])
    df = df[df["crop"] != ""].copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable dataset rows remain after cleaning.")

    # Chronological split: validation chooses the model;
    # the test set is reserved for final evaluation.
    train_df = df[df["year"] <= 2010].copy()
    validation_df = df[
        (df["year"] >= 2011) & (df["year"] <= 2017)
    ].copy()
    test_df = df[df["year"] >= 2018].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError(
            "Training, validation, or test set is empty. "
            "Check the dataset year range."
        )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[FEATURES]
    y_validation = validation_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    validation_results = []
    fitted_candidates = {}

    print("Dataset rows:", len(df))
    print(
        f"Training years: {train_df['year'].min()}-"
        f"{train_df['year'].max()} ({len(train_df)} rows)"
    )
    print(
        f"Validation years: {validation_df['year'].min()}-"
        f"{validation_df['year'].max()} ({len(validation_df)} rows)"
    )
    print(
        f"Test years: {test_df['year'].min()}-"
        f"{test_df['year'].max()} ({len(test_df)} rows)"
    )

    # Fit candidates on training data and select using validation RMSE.
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        validation_predictions = pipeline.predict(X_validation)
        metrics = calculate_metrics(y_validation, validation_predictions)

        validation_results.append({
            "Model": name,
            "Validation_MAE": metrics["MAE"],
            "Validation_RMSE": metrics["RMSE"],
            "Validation_R2": metrics["R2"],
        })
        fitted_candidates[name] = estimator

        print(
            f"{name}: validation "
            f"MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, "
            f"R2={metrics['R2']:.4f}"
        )

    validation_df_results = pd.DataFrame(validation_results)
    validation_df_results = validation_df_results.sort_values(
        "Validation_RMSE"
    )

    best_name = validation_df_results.iloc[0]["Model"]

    # Refit the selected model using all development data.
    development_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    ).sort_values("year")

    final_model = make_pipeline(fitted_candidates[best_name])
    final_model.fit(
        development_df[FEATURES],
        development_df[TARGET],
    )

    # Evaluate the selected model on the untouched test set.
    test_predictions = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)

    test_result = {
        "Model": best_name,
        "Test_MAE": test_metrics["MAE"],
        "Test_RMSE": test_metrics["RMSE"],
        "Test_R2": test_metrics["R2"],
        "Test_Start_Year": int(test_df["year"].min()),
        "Test_End_Year": int(test_df["year"].max()),
        "Test_Rows": len(test_df),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the final pipeline and separate validation/test reports.
    joblib.dump(final_model, MODEL_DIR / "yield_model.joblib")

    validation_df_results.to_csv(
        REPORT_DIR / "validation_metrics.csv",
        index=False,
    )
    pd.DataFrame([test_result]).to_csv(
        REPORT_DIR / "model_metrics.csv",
        index=False,
    )

    pd.DataFrame({
        "year": test_df["year"].to_numpy(),
        "crop": test_df["crop"].to_numpy(),
        "actual_yield_tonnes_per_hectare": y_test.to_numpy(),
        "predicted_yield_tonnes_per_hectare": test_predictions,
    }).to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False,
    )

    print("\n===== VALIDATION COMPARISON =====")
    print(validation_df_results.to_string(index=False))
    print("\nSelected model:", best_name)

    print("\n===== FINAL TEST RESULTS =====")
    for metric, value in test_result.items():
        print(f"{metric}: {value}")

    print("\nSaved model:", MODEL_DIR / "yield_model.joblib")
    print("Saved validation metrics:", REPORT_DIR / "validation_metrics.csv")
    print("Saved test metrics:", REPORT_DIR / "model_metrics.csv")
    print("Saved test predictions:", REPORT_DIR / "test_predictions.csv")

    print(
        "\nLIMITATION: This is an exploratory retrospective model. "
        "The dataset has geographic scope limitations and is not "
        "validated for nationwide or pre-season crop-yield predictions."
    )


if __name__ == "__main__":
    main()

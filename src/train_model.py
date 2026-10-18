from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "combined_crop_yield.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "yield_tonnes_per_hectare"
FEATURES = ["year", "crop"]


def make_pipeline(estimator):
    preprocessor = ColumnTransformer(
        transformers=[
            ("crop", OneHotEncoder(handle_unknown="ignore"), ["crop"]),
            ("year", "passthrough", ["year"]),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["crop"] = df["crop"].astype("string").str.strip()

    df = df.dropna(subset=required)
    df = df[df["crop"] != ""]

    if (df[TARGET] < 0).any():
        raise ValueError("Negative yield values found. Review the dataset.")

    if df.duplicated(["year", "crop"]).any():
        raise ValueError("Duplicate crop-year records found.")

    years = sorted(df["year"].unique())

    if len(years) < 20:
        raise ValueError("Not enough distinct years for this time-based split.")

    # Fixed chronological split for the current 1991–2024 dataset.
    train = df[df["year"] <= 2010]
    validation = df[(df["year"] >= 2011) & (df["year"] <= 2017)]
    test = df[df["year"] >= 2018]

    if train.empty or validation.empty or test.empty:
        raise ValueError("A training, validation, or test split is empty.")

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val = validation[FEATURES]
    y_val = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }

    validation_results = []
    fitted_candidates = {}

    print(f"Dataset rows: {len(df)}")
    print(f"Training years: {int(train.year.min())}-{int(train.year.max())}")
    print(
        f"Validation years: "
        f"{int(validation.year.min())}-{int(validation.year.max())}"
    )
    print(f"Test years: {int(test.year.min())}-{int(test.year.max())}")
    print("\nValidation results:")

    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)

        result = {
            "model": name,
            "validation_MAE": mean_absolute_error(y_val, predictions),
            "validation_RMSE": rmse(y_val, predictions),
            "validation_R2": r2_score(y_val, predictions),
        }

        validation_results.append(result)
        fitted_candidates[name] = estimator

        print(
            f"{name}: "
            f"MAE={result['validation_MAE']:.4f}, "
            f"RMSE={result['validation_RMSE']:.4f}, "
            f"R2={result['validation_R2']:.4f}"
        )

    best_result = min(validation_results, key=lambda row: row["validation_RMSE"])
    best_name = best_result["model"]

    # Refit the selected model on all pre-test observations.
    development = pd.concat([train, validation]).sort_values("year")
    final_model = make_pipeline(fitted_candidates[best_name])
    final_model.fit(development[FEATURES], development[TARGET])

    test_predictions = final_model.predict(X_test)

    test_result = {
        "model": best_name,
        "test_MAE": mean_absolute_error(y_test, test_predictions),
        "test_RMSE": rmse(y_test, test_predictions),
        "test_R2": r2_score(y_test, test_predictions),
        "test_start_year": int(test["year"].min()),
        "test_end_year": int(test["year"].max()),
        "test_rows": len(test),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODEL_DIR / "yield_model.joblib")
    pd.DataFrame(validation_results).to_csv(
        REPORT_DIR / "validation_metrics.csv", index=False
    )
    pd.DataFrame([test_result]).to_csv(
        REPORT_DIR / "model_metrics.csv", index=False
    )

    pd.DataFrame({
        "year": test["year"].astype(int).to_numpy(),
        "crop": test["crop"].to_numpy(),
        "actual_yield_tonnes_per_hectare": y_test.to_numpy(),
        "predicted_yield_tonnes_per_hectare": test_predictions,
    }).to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    print(f"\nSelected model: {best_name}")
    print("Final test results (2018–2024):")
    for metric, value in test_result.items():
        print(f"{metric}: {value}")

    print("\nCreated:")
    print(MODEL_DIR / "yield_model.joblib")
    print(REPORT_DIR / "validation_metrics.csv")
    print(REPORT_DIR / "model_metrics.csv")
    print(REPORT_DIR / "test_predictions.csv")

    print(
        "\nLIMITATION: This is a historical trend baseline using year and crop "
        "only. It does not use weather or establish causal agricultural "
        "relationships. Future-year predictions are extrapolations."
    )


if __name__ == "__main__":
    main()

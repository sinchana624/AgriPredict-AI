from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "combined_crop_yield.csv"
MODEL_PATH = ROOT / "models" / "yield_model.joblib"
REPORT_DIR = ROOT / "reports"

TARGET = "yield_tonnes_per_hectare"
FEATURES = ["year", "crop"]


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def main():
    for path in (DATA_PATH, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = df[required].copy()
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["crop"] = df["crop"].astype(str).str.strip()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
    df = df.dropna(subset=required)

    development = df[df["year"] <= 2017].copy()
    test = df[df["year"] >= 2018].copy()

    if development.empty or test.empty:
        raise ValueError("Development or test split is empty.")

    # Each crop's latest observed yield through 2017 is the naive baseline.
    last_yield_by_crop = (
        development.sort_values("year")
        .groupby("crop")
        .tail(1)
        .set_index("crop")[TARGET]
        .to_dict()
    )

    if not set(test["crop"]).issubset(last_yield_by_crop):
        raise ValueError("A test crop has no historical baseline value.")

    X_test = test[FEATURES]
    y_test = test[TARGET]
    model = joblib.load(MODEL_PATH)

    model_predictions = model.predict(X_test)
    baseline_predictions = test["crop"].map(last_yield_by_crop).to_numpy()

    comparison_rows = []
    for name, predictions in [
        ("Saved ML model", model_predictions),
        ("Last observed yield baseline", baseline_predictions),
    ]:
        row = {"model": name, **calculate_metrics(y_test, predictions)}
        comparison_rows.append(row)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = REPORT_DIR / "baseline_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)

    # Descriptive test-set permutation importance. Small test sets make
    # these estimates uncertain, so they should not be treated as definitive.
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=30,
        random_state=42,
    )

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "mae_increase_mean": importance.importances_mean,
        "mae_increase_std": importance.importances_std,
    }).sort_values("mae_increase_mean", ascending=False)

    importance_path = REPORT_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    print("Evaluation years:", int(test["year"].min()), "-", int(test["year"].max()))
    print("Test rows:", len(test))
    print("\nBaseline comparison (lower MAE/RMSE is better):")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    print("\nPermutation importance (larger positive MAE increase means more useful):")
    print(importance_df.to_string(index=False))
    print("\nCreated:")
    print(comparison_path)
    print(importance_path)
    print(
        "\nCaution: The test set contains few observations. Importance values "
        "are exploratory, can be noisy, and do not imply causation."
    )


if __name__ == "__main__":
    main()

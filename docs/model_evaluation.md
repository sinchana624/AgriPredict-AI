# AgriPredict AI — ML Model Evaluation

## 1. Objective

Estimate crop yield using historical year and crop information. This is a baseline machine-learning experiment, not a fully validated operational forecasting system.

## 2. Dataset

- Source file: `data/processed/combined_crop_yield.csv`
- Observations: 102
- Crops: Maize, Rice, Wheat
- Geography: India-level annual records
- Years: 1991–2024
- Target: `yield_tonnes_per_hectare`
- Input features: `year`, `crop`

Weather and pesticide files are not inputs to this model. Regional NASA POWER weather data must not be treated as nationally representative without a justified geographic matching and validation process.

## 3. Evaluation design

- Training period: 1991–2010
- Validation period: 2011–2017
- Held-out test period: 2018–2024
- Test observations: 21

Candidate models were compared on validation data. Linear Regression was selected based on validation RMSE and evaluated on the later test period.

The saved model was fitted using training and validation observations through 2017. The held-out test period was not used to select the model.

## 4. Held-out test metrics

| Metric | Saved ML model |
|---|---:|
| MAE | 0.276931 |
| RMSE | 0.333521 |
| R-squared | 0.335229 |

MAE and RMSE are measured in tonnes per hectare. Lower MAE/RMSE is better. R-squared describes the proportion of test-set variation explained relative to the test-set mean; it can be negative for a poor model.

## 5. Historical baseline comparison

The baseline predicts each crop's last observed yield in the development period (through 2017).

| Model | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Saved ML model | 0.276931 | 0.333521 | 0.335229 |
| Last observed yield baseline | 0.321181 | 0.348864 | 0.272659 |

The saved model performed better than this baseline on the 2018–2024 test set. This result alone does not establish that the model will generalize to future years or other geographic scales.

Detailed results: `reports/baseline_comparison.csv`.

## 6. Exploratory feature importance

Permutation importance was calculated on the held-out test set using the increase in MAE after shuffling each feature.

| Feature | Mean MAE increase | Standard deviation |
|---|---:|---:|
| crop | 0.265362 | 0.043573 |
| year | 0.015909 | 0.009845 |

Crop had the larger estimated impact in this evaluation. The test set is small, so these estimates can be noisy. Feature importance does not establish causation.

Detailed results: `reports/feature_importance.csv`.

## 7. Prediction module and tests

- Prediction function: `src/predict.py`
- Evaluation script: `src/evaluate_model.py`
- Automated tests: `tests/test_predict.py`

The prediction function validates year and crop inputs and loads the saved model. The initial prediction-module test run passed all 9 tests.

## 8. Limitations and next steps

1. The dataset has only 102 annual crop observations across three crops.
2. The model uses only year and crop; it does not use rainfall, temperature, soil, irrigation, or pesticide data.
3. National annual records do not support district-level predictions.
4. The test period is short, and reported metrics have uncertainty.
5. Future work should validate on more observations, compare against additional baselines, and incorporate weather only after matching compatible geography, years, and crop records.
6. Model outputs should be presented as estimates, not guaranteed yields or agricultural advice.

## 9. Reproducibility

Run from the repository root with the project virtual environment active:

```powershell
py src/evaluate_model.py
py -m pytest -v
```

The evaluation script writes `reports/baseline_comparison.csv` and `reports/feature_importance.csv`.

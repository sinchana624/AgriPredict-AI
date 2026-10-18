# AgriPredict AI — Machine Learning Module

## Overview
This module trains and evaluates machine learning models
for crop yield prediction.

## Dataset
- File: `data/processed/exploratory_model_dataset_for_handoff.csv`
- Target: `yield_tonnes_per_hectare`
- Inputs: `year`, `crop`, `rainfall_ann`, `tmax_ann`,
  `tmean_ann`, `tmin_ann`

## Setup
Activate the virtual environment and install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Train the model

```powershell
python train_model.py
```

The trained model is saved to `models/yield_model.joblib`.
Evaluation metrics are saved to `reports/model_metrics.csv`.

## Run tests

```powershell
python -m pytest tests/test_predict.py -v
```

## Prediction
Import `predict_yield` from `src.predict`.
Provide all six required input features.
The returned value is in tonnes per hectare.

## Model limitations
The current baseline uses only `year` and `crop` to predict
crop yield. It does not use rainfall or temperature features.
Predictions for future years are extrapolations of historical
trends and are not validated nationwide agricultural forecasts.


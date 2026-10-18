# Exploratory Model Dataset Validation

## Status
Exploratory candidate validated for structural consistency. Not yet approved
for reliable nationwide crop-yield prediction.

## Files
- `data/processed/regional_weather_annual_candidate.csv`
- `data/processed/exploratory_model_dataset.csv`

## Validation results
- Coverage: 1991?2024 (34 years).
- Annual regional weather summary: 34 rows, one per year.
- Joined exploratory dataset: 102 rows.
- Crops: Maize, Rice, and Wheat, with 34 rows each.
- Missing values in the joined dataset: 0.
- Duplicate year/crop/location keys: 0.
- Pesticide series matched by year; it is national-level data.
- Weather values are averaged across 336 grid points in the 6?16?N,
  78?88?E region.

## Important limitations
The crop-yield and pesticide datasets describe India nationally, whereas
the weather summary describes only a regional grid. Matching on year makes
the join structurally consistent, but does not make the regional weather
representative of all India.

Do not present this dataset as a validated nationwide prediction dataset.
For a national model, obtain suitable India-wide weather data and verify its
definitions and units. For a state-level model, use geographically aligned
crop, weather, and pesticide data at state-year level; national pesticide
values cannot be treated as state-specific measurements.

Pesticide-use data includes official, imputed, and estimated values. Preserve
its source quality flags when selecting features for modelling.

## Recommended next work
1. Source and validate weather data representative of India nationally, or
   explicitly redefine the project as a regional model.
2. Confirm the intended prediction target and geographic scope.
3. Review pesticide data quality flags and feature suitability.
4. Perform chronological train/test separation and baseline evaluation.
5. Document limitations and final handover decisions.

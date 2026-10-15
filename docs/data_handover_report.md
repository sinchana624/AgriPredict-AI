# AgriPredict AI — Data Handover Report

## 1. Purpose
This report summarizes the available crop-yield, weather, and pesticide datasets for the AgriPredict AI project.

## 2. Crop data

### National FAOSTAT crop data
- Raw file: `data/raw/combined_crop_yield.csv`
- Processed file: `data/processed/combined_crop_yield.csv`
- Geographic level: India national
- Crops: Maize, Rice, Wheat
- Years: 1991–2024
- Processed rows: 102
- Contains production, harvested area, and yield.
- Yield is expressed in tonnes per hectare in the processed file.

### State-level crop data
- Processed source: `data/processed/state_crop_production_yield.csv`
- Candidate file: `data/processed/state_crop_yield_3crops_candidate.csv`
- Positive-yield candidate: `data/processed/state_crop_yield_3crops_positive_candidate.csv`
- Selected crops: Rice, Maize, Wheat
- Coverage: 1998–99 to 2011–12
- Positive-yield candidate: 1,356 rows
- Duplicate state/year/crop/season keys: 0
- Missing values in candidate: 0
- One zero-production record was excluded from the positive-yield candidate and retained in the original candidate for review.
- Most observations are seasonal, not annual. Do not combine seasons or treat them as annual totals without validating the source definitions.
- The 2011–12 data appears incomplete and requires review.

## 3. Weather data
- Raw source: NASA POWER monthly data.
- Files are in `data/raw/weather/`.
- Combined file: `data/processed/weather_combined_6_16N_78_88E_1991_2024.csv`
- Coverage: 1991–2024.
- Grid: 336 coordinate points across the 6–16°N, 78–88°E regional bounding box.
- Variables: precipitation, mean temperature, maximum temperature, and minimum temperature, represented by monthly columns.
- Combined rows: 11,424.
- Missing values and duplicate coordinate/year records: 0 in the validated combined file.
- Important: this is a regional grid, not an India-wide dataset. Some grid cells may fall outside India or outside agricultural land.
- Verify precipitation units and monthly aggregation semantics against NASA POWER documentation before modeling.

## 4. Pesticide data
- Raw file: `data/raw/pesticide_use.csv`
- Processed file: `data/processed/pesticide_india_main_indicators.csv`
- Geographic level: India national.
- Coverage: 1991–2024.
- Rows: 136 (four indicators × 34 years).
- Missing values: 0.
- Units are preserved separately: tonnes, kg/ha, kg/cap, and g/Int$.
- Quality flags are preserved. Total pesticide use has imputed values and repeated values in recent years; do not treat all years as verified official observations.

## 5. Key limitations before modeling
1. Crop, weather, and pesticide data do not all share the same geographic level.
2. State-level crop observations are seasonal, while the weather data is a regional grid and pesticide indicators are India-wide.
3. Crop years and calendar weather years need a documented alignment rule.
4. The weather grid needs a geographic mask or spatial join to relevant Indian locations.
5. Precipitation units and aggregation semantics need verification.
6. Pesticide quality flags must be retained and considered when selecting features.
7. Avoid target leakage: do not use yield-derived information as an input feature for predicting yield.
8. Train/test splits should respect time to avoid leakage from future years.

## 6. Suggested next steps
1. Agree on the prediction target and whether it is seasonal or annual.
2. Validate crop-season definitions and zero-production records.
3. Align weather observations to crop locations and crop growing seasons.
4. Decide whether national pesticide indicators are appropriate as features.
5. Create a final documented modeling table only after the alignment rules are agreed.
6. Keep raw data unchanged and version processed outputs.

## 7. Reproducibility
Raw files should remain unchanged. Processed outputs should be reproducible from documented source files and preprocessing steps. Record any excluded observations and reasons.

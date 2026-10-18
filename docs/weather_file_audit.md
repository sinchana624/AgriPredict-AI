# Weather File Audit

## Decision summary

- `data/processed/regional_annual_precipitation.csv`:
  Regional summary for the 336-point grid over 6?16?N, 78?88?E.
  It is not a nationwide India rainfall estimate. Retain only with this limitation documented.

- `data/processed/annual_weather_by_state.csv`:
  Older state-level aggregation. Compared with
  `annual_weather_by_state_validated_candidate.csv`, 102 state-year records overlap.
  Mean absolute precipitation difference: 308.094 mm; maximum: 950.148 mm.
  Grid-point counts and geographic coverage differ. Do not use as a substitute
  for the validated candidate.

- `data/processed/weather.csv`:
  Older NASA POWER precipitation export covering a different longitude region
  (approximately 68?78?E). Keep local for reference; do not combine with the
  newer 78?88?E weather dataset.

- `data/processed/weather_grid_state_mapping.csv`:
  Older mapping with 269 of 336 state entries missing. Do not use for state-level
  modeling. Prefer the validated mapping and preserve unmapped points as unmapped.

## Modeling caution

These files have different geographic coverage and levels of spatial aggregation.
Do not join them to crop-yield data as though they represent the same locations.


## Final audit measurements

### Older annual state rainfall
- Older file: 170 rows; validated candidate: 136 rows.
- Overlapping state-year records: 102.
- Mean absolute precipitation difference: 308.094 mm.
- Conclusion: do not substitute the older aggregation for the validated candidate.

### Older NASA POWER export
- Records: 11,424, covering 1991?2024.
- Longitude range: 68.125??77.5?E.
- Current combined dataset longitude range: 78.125??87.5?E.
- Duplicate coordinate-year records in the older export: 0.
- Conclusion: different geographic coverage; retain locally for reference only.

### Older grid-to-state mapping
- Grid points: 336.
- Missing state labels: 269.
- Labeled points: 67.
- Validated mapping has 51 labeled points out of 336.
- Conclusion: prefer the validated mapping and do not force labels onto unmapped points.

These three older files remain local and untracked. They have not been deleted
or added to the modeling workflow.

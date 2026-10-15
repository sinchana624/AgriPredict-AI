# AgriPredict AI ? Weather Data Preprocessing Report

## Dataset
- Source: NASA POWER
- Temporal coverage: 1991?2024
- Temporal resolution: Monthly observations
- Spatial coverage: Regional grid, latitude 6??16? N and longitude 78??88? E
- Grid points: 336
- Records: 11,424 grid-point/year rows

## Variables
- Rainfall: PRECTOTCORR_SUM
- Mean temperature: T2M
- Maximum temperature: T2M_MAX
- Minimum temperature: T2M_MIN

## Output
`data/processed/weather_combined_6_16N_78_88E_1991_2024.csv`

The output contains year, latitude, longitude, and 12 monthly values for each of the four weather variables.

## Validation performed
- All four files contain 11,424 rows.
- All four files cover 1991?2024.
- All four files have matching year/latitude/longitude keys.
- No missing values or -999 markers were reported.
- No duplicate rows were reported in the individual files.
- Monthly rainfall sums match NASA's annual column in the downloaded file.
- No cases were found where minimum temperature exceeded mean temperature or mean temperature exceeded maximum temperature.

## Limitations and pending checks
- This is a regional subset, not an India-wide dataset.
- The grid must be checked against India's geographic boundary before making country-level claims.
- Precipitation units and monthly aggregation semantics must be confirmed against NASA POWER documentation before modeling.
- These checks establish structural consistency, not full scientific validity.

from pathlib import Path
import pandas as pd

root = Path.cwd()
weather_file = root / "data/processed/weather_combined_6_16N_78_88E_1991_2024.csv"
mapping_file = root / "data/processed/weather_grid_state_mapping_validated.csv"
output_file = root / "data/processed/annual_weather_by_state_validated_candidate.csv"

weather = pd.read_csv(weather_file)
mapping = pd.read_csv(mapping_file)

months = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]
rain_columns = [f"rainfall_{month}" for month in months]

required = ["YEAR", "LAT", "LON", *rain_columns]
missing = [column for column in required if column not in weather.columns]
if missing:
    raise ValueError(f"Missing expected weather columns: {missing}")

if mapping.duplicated(["LAT", "LON"]).any():
    raise ValueError("Mapping contains duplicate grid coordinates.")

# Sum all 12 monthly rainfall values for each grid point and year.
weather["annual_precipitation"] = weather[rain_columns].sum(axis=1, min_count=12)

# Attach only points assigned to a state by the validated polygon mapping.
joined = weather.merge(
    mapping[["LAT", "LON", "state", "state_code"]],
    on=["LAT", "LON"],
    how="left",
    validate="many_to_one"
)

mapped = joined[joined["state"].notna()].copy()
if mapped.empty:
    raise ValueError("No weather grid points matched the validated mapping.")

annual = (
    mapped.groupby(["YEAR", "state", "state_code"], as_index=False)
    .agg(
        grid_points=("annual_precipitation", "count"),
        mean_annual_precipitation=("annual_precipitation", "mean")
    )
    .rename(columns={"YEAR": "year"})
)

annual["location_scope"] = (
    "Regional grid-point average; not a complete state-wide estimate"
)
annual = annual[
    ["year", "state", "state_code", "location_scope",
     "grid_points", "mean_annual_precipitation"]
].sort_values(["year", "state"]).reset_index(drop=True)

annual.to_csv(output_file, index=False)

print("Candidate file created:", output_file)
print("Rows:", len(annual))
print("Years:", annual["year"].min(), "to", annual["year"].max())
print("States:", annual["state"].nunique())
print("\nGrid points represented by state:")
print(mapping.dropna(subset=["state"]).groupby("state").size().to_string())
print("\nRows per state:")
print(annual.groupby("state").size().to_string())
print("\nMissing values:")
print(annual.isna().sum().to_string())
print("\nFirst rows:")
print(annual.head(8).to_string(index=False))

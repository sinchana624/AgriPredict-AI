from pathlib import Path
import geopandas as gpd
import pandas as pd

root = Path.cwd()
boundary_file = root / "data/raw/boundaries/geoBoundaries-IND-ADM1_simplified.geojson"
weather_file = root / "data/processed/weather_combined_6_16N_78_88E_1991_2024.csv"
output_file = root / "data/processed/weather_grid_state_mapping_validated.csv"

# Load the actual state polygons.
states = gpd.read_file(boundary_file)
if states.crs is None:
    raise ValueError("Boundary file has no coordinate reference system.")
states = states.to_crs("EPSG:4326")

# Keep only the needed boundary attributes.
states = states[["shapeName", "shapeISO", "geometry"]].rename(
    columns={"shapeName": "state", "shapeISO": "state_code"}
)

# Extract unique weather grid points.
weather = pd.read_csv(weather_file, usecols=["LAT", "LON"])
points_df = weather.drop_duplicates().dropna().copy()
points = gpd.GeoDataFrame(
    points_df,
    geometry=gpd.points_from_xy(points_df["LON"], points_df["LAT"]),
    crs="EPSG:4326"
)

# Assign a state only when the point falls inside its polygon.
matched = gpd.sjoin(
    points,
    states,
    how="left",
    predicate="within"
)

result = pd.DataFrame({
    "LAT": matched["LAT"].to_numpy(),
    "LON": matched["LON"].to_numpy(),
    "state": matched["state"].to_numpy(),
    "state_code": matched["state_code"].to_numpy()
})

result = result.sort_values(["LAT", "LON"]).reset_index(drop=True)
result.to_csv(output_file, index=False)

print("Mapping validation completed.")
print("Output:", output_file)
print("Unique grid points:", len(result))
print("Mapped points:", result["state"].notna().sum())
print("Unmapped points:", result["state"].isna().sum())
print("Mapped states:", result["state"].nunique())
print("\nGrid points by state:")
print(result["state"].value_counts().to_string())
print("\nNote: unmapped points are intentionally left blank.")

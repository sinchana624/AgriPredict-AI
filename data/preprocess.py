
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed" / "combined_crop_yield.csv"

INPUT = RAW / "combined_crop_yield.csv"

# Read the original dataset without modifying it.
df = pd.read_csv(INPUT)
df.columns = df.columns.str.strip()

required = {"Area", "Item", "Element", "Year", "Unit", "Value"}
missing_columns = required - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

# Select the three required crops.
crop_names = {
    "rice": "Rice",
    "wheat": "Wheat",
    "maize (corn)": "Maize",
}

df["crop_key"] = df["Item"].astype(str).str.strip().str.lower()
df = df[df["crop_key"].isin(crop_names)].copy()

if df.empty:
    raise ValueError("No rice, wheat, or maize records were found.")

df["crop"] = df["crop_key"].map(crop_names)

# Check that the source measurements use the expected units.
expected_units = {
    "Area harvested": "ha",
    "Yield": "kg/ha",
    "Production": "t",
}

for element, expected_unit in expected_units.items():
    rows = df[df["Element"] == element]

    if rows.empty:
        raise ValueError(f"No records found for {element}.")

    actual_units = set(rows["Unit"].dropna().astype(str).str.strip())

    if actual_units != {expected_unit}:
        raise ValueError(
            f"Unexpected unit for {element}: {actual_units}. "
            f"Expected {expected_unit}. Check source units before proceeding."
        )

# Each crop-year must have one record for each measurement.
keys = ["Area", "crop", "Year", "Element"]

duplicates = df[df.duplicated(keys, keep=False)]

if not duplicates.empty:
    raise ValueError(
        "Duplicate crop-year-measurement records found. "
        "Inspect the source before combining."
    )

# Convert the source's long format into one row per crop-year.
wide = df.pivot(
    index=["Area", "crop", "Year"],
    columns="Element",
    values="Value",
).reset_index()

wide.columns.name = None

required_elements = ["Area harvested", "Yield", "Production"]

for element in required_elements:
    if element not in wide.columns:
        raise ValueError(f"Missing measurement after pivot: {element}")

if wide[required_elements].isna().any().any():
    raise ValueError(
        "Some crop-year records lack one or more measurements. "
        "Inspect and document them before deciding how to handle them."
    )

# Rename columns and convert yield from kg/ha to tonnes/ha.
wide = wide.rename(
    columns={
        "Area": "location",
        "Year": "year",
        "Area harvested": "harvested_area_ha",
        "Production": "production_tonnes",
    }
)

wide["yield_tonnes_per_hectare"] = wide["Yield"] / 1000
wide = wide.drop(columns=["Yield"])

wide["year"] = wide["year"].astype(int)

wide = wide[
    [
        "year",
        "crop",
        "location",
        "production_tonnes",
        "harvested_area_ha",
        "yield_tonnes_per_hectare",
    ]
].sort_values(["year", "crop"]).reset_index(drop=True)

# Final validation.
if wide.duplicated(["year", "crop", "location"]).any():
    raise ValueError("Duplicate output crop-year-location keys detected.")

if (wide["production_tonnes"] < 0).any():
    raise ValueError("Negative production values detected.")

if (wide["harvested_area_ha"] < 0).any():
    raise ValueError("Negative harvested-area values detected.")

if (wide["yield_tonnes_per_hectare"] < 0).any():
    raise ValueError("Negative yield values detected.")

# Create the output only after all validation checks pass.
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
wide.to_csv(OUTPUT, index=False)

print("Crop preprocessing completed successfully.")
print("Output:", OUTPUT)
print("Shape:", wide.shape)
print("Years:", wide["year"].min(), "-", wide["year"].max())
print("Crops:", sorted(wide["crop"].unique()))
print("\nRows per crop:")
print(wide.groupby("crop").size().to_string())
print("\nMissing values:")
print(wide.isna().sum().to_string())
print("\nFirst 10 rows:")
print(wide.head(10).to_string(index=False))

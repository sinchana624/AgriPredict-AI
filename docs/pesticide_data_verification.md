# Pesticide Data Verification

## Files
- `data/processed/pesticide_india_main_indicators.csv`
  - Primary processed dataset.
  - Contains four pesticide-use indicators and source quality flags.
- `data/processed/pesticide_use.csv`
  - Simplified annual total pesticide-use series.
  - Convenience copy of the Agricultural Use total, in tonnes.

## Verification
- Years covered: 1991 to 2024.
- Simplified file rows: 34.
- Detailed agricultural-use total rows: 34.
- Years matched: 34/34.
- Values matched: 34/34.
- Unmatched years or values: 0.

## Quality flags in the total-use series
- Official value: 16.
- Value imputed by a receiving agency: 15.
- Estimated value: 3.

## Modeling caution
The simplified file does not preserve quality flags. Use the detailed dataset
when preparing model features and retain the flag information. Do not assume
imputed or estimated values have the same reliability as official values.

Pesticide use is reported for India nationally, so it does not directly provide
state-level pesticide usage. Consider geographic and temporal compatibility
before joining it to crop-yield or weather data.

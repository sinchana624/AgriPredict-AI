# Official Rice Yield Dataset Verification

## Source
DES, Report 4.1.5: State-wise Yield of Rice.
Local source file inspected: `4.1.5.xls`, sheet `Rice U`.

## Processed candidate
`data/processed/official_state_rice_yield_1999_2011.csv`

- 204 rows
- 17 states/UTs
- 12 years: 1999-00 through 2010-11
- Season: Total
- Duplicate State-Year-Crop-Season keys: 0
- Missing rice-yield values: 13

## Missing values
The original report was checked for the missing state-year entries.

- Gujarat: 1999-00 to 2002-03 marked `-`
- Jharkhand: 1999-00 marked `-`
- Uttarakhand: 1999-00 to 2002-03 marked `-` or blank
- Meghalaya, Mizoram, Pondicherry and Tripura: 2010-11 entries blank

Missing values are retained as missing. They must not be replaced with zero or an arbitrary estimate.

## Limitations
- This candidate covers only 17 states/UTs and 12 years.
- The report heading marks the final year as `2010-11*`; confirm the footnote before interpreting that year's figures.
- The candidate is not a complete national panel and should not be treated as a complete state-level training dataset.
- Preserve the original source and document any later corrections.

## Handover recommendation
Use this file only as an official-source candidate with its missing values documented. Align years, state names, crop definitions and geographic coverage with the other AgriPredict AI datasets before model training.

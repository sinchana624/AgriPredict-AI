# Regional State-Mapped Rainfall Candidate

## Dataset summary
- File: `data/processed/annual_weather_by_state_validated_candidate.csv`
- Records: 136
- Years: 1991?2024
- States represented: 4
- Measure: mean annual precipitation across mapped NASA POWER grid points (mm).

## State coverage

| State | Grid points | Years | Minimum (mm) | Maximum (mm) |
|---|---:|---:|---:|---:|
| Andhra Pradesh | 23 | 34 | 483.86 | 1196.09 |
| Karnātaka | 2 | 34 | 597.66 | 1720.16 |
| Tamil Nādu | 25 | 34 | 571.87 | 1491.56 |
| Telangāna | 1 | 34 | 384.52 | 1147.22 |

## Validation
- No duplicate year-state rows.
- Annual rainfall was recomputed as the sum of the 12 monthly precipitation values.
- State-year averages were independently recomputed and matched within floating-point tolerance.

## Important limitations
- This is a regional grid, not an India-wide weather dataset.
- Grid-point counts vary greatly by state; one or two points do not represent statewide climate.
- The averages describe only the mapped points and should not be described as official state-wide rainfall.
- Do not use this file as a substitute for district-level or statewide weather data without further validation.

## Recommended use
Retain as a documented exploratory candidate. For modeling, align weather with the actual crop-production geography and acquire suitable district/state-level weather data where required.

# Findings Summary

## Core findings

- Negative hourly prices: **1608**
- Hourly price range: **-500.00 to 872.96 EUR/MWh**
- Target frequency break: hourly through **2025-01-01 00:00:00**, then quarter-hour from **2025-01-01 00:15:00**
- Strongest multi-feature exploratory OLS model: **Multiple OLS B: price ~ weather + production + calendar** with adjusted R² **0.1501**

## Top weather variables by pairwise R²

- Best for price: **wind_speed_10m** with R² **0.0368**
- Best for total load: **temperature_2m** with R² **0.0631**
- Interpretation: Weather appears more directly tied to load than to price in pairwise linear screening.

## Variable-level price regressions

- Best single-variable price model: **wind_speed_10m** with adjusted R² **0.0368**
- Best two-variable price model: **wind_speed_10m + generation_forecast** with adjusted R² **0.0796**
- The by-variable regression outputs now explicitly cover every single exogenous feature and every two-feature combination against price.
- `total_load -> price`: fitted direction is **falls**, model p-value is **9.25e-08**, but R² is only **0.0003**, so the effect is statistically detectable and still very weak in explanatory terms.

## Modeling implications

- Most promising current candidate features: **total_load, wind_speed_10m, shortwave_radiation, generation_forecast, temperature_2m**
- Weak direct signals in the current hourly panel: **cloud_cover, temperature_2m**
- Most visible missingness issues: **generation_forecast, total_load**
- Pairwise weather R² should be used for explanatory screening only, not as a substitute for time-aware forecast validation.

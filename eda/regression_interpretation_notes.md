# Regression Interpretation Notes

## Direct answers to common questions

### Does price rise when load rises?

No, not in the simple same-time linear screen. The fitted coefficient in `price ~ total_load` is **-0.000547**, so as load rises, the fitted same-time price **falls** slightly on average. The model p-value is **9.25e-08**, which is below 0.05, but the R² is only **0.0003**. That means the relationship is statistically detectable yet explains almost none of the price variation.

### Does price rise when temperature rises?

No in the simple same-time model. The fitted coefficient in `price ~ temperature_2m` is **-0.3491**, so as temperature rises, fitted price **falls** on average. The relationship is statistically significant, but the explanatory power is still very weak with R² **0.0009**.

### Does price rise when wind speed rises?

No in the simple same-time model. The fitted coefficient in `price ~ wind_speed_10m` is **-2.2597**, so as wind speed rises, fitted price **falls** on average. This is the strongest single-variable price model in the current screen, but its R² is still only **0.0368**.

### What about cloud cover and wind speed together?

For `price ~ cloud_cover + wind_speed_10m`, the overall model has adjusted R² **0.0385**. The feature effects are:

- cloud_cover: price rises when the variable rises (significant, coef=10.3051, p=3.28e-38)
- wind_speed_10m: price falls when the variable rises (significant, coef=-2.3639, p=0)

This pair improves on `wind_speed_10m` alone, but it is still an explanatory screen rather than a forecasting evaluation.

## How to interpret p-value and R² correctly

- `p < 0.05` means the fitted linear association is unlikely to be exactly zero under the OLS assumptions.
- It does **not** mean the variable should definitely be included.
- `R²` shows how much in-sample variance is explained by the model. A very small `R²` means the model can be statistically significant and still explain very little.
- For forecasting, a feature should be judged by availability at forecast time, leakage safety, stability over time, and out-of-sample performance, not by p-value alone.

FEATURE BEHAVIOR INTERPRETATION:

--------------------------------------------------

temperature_2m (°C)
- Low (< 5°C):
    Increased heating demand → electricity demand rises → upward pressure on prices.
- Moderate (5–20°C):
    Baseline demand conditions → limited impact on price.
- High (> 25°C):
    Increased cooling demand (air conditioning load) → demand rises → potential price increase.

--------------------------------------------------

cloud_cover (0–1)
- Low (0.0–0.2):
    Clear sky conditions → high solar irradiance → increased photovoltaic generation → downward pressure on prices.
- Medium (0.2–0.7):
    Partial cloud coverage → reduced solar output → moderate impact on supply.
- High (0.7–1.0):
    Limited solar generation → reduced renewable supply → upward pressure on prices.

--------------------------------------------------

wind_speed_10m (m/s)
- Low (< 3 m/s):
    Insufficient wind for turbine operation → low wind generation → reduced supply → higher prices.
- Moderate (3–10 m/s):
    Optimal wind conditions → high turbine efficiency → increased supply → lower prices.
- High (> 10–12 m/s):
    Potential curtailment or turbine shutdown (cut-out speeds) → reduced generation → possible price increase.

--------------------------------------------------

shortwave_radiation (W/m²)
- 0:
    Nighttime → no solar generation → no contribution to supply.
- 0–200:
    Low irradiance → minimal solar output.
- 200–600:
    Moderate solar generation → contributes to supply.
- 600–1000+:
    High irradiance → strong photovoltaic output → increased supply → lower prices.

--------------------------------------------------

total_load (MW)
- Low:
    Reduced electricity demand → excess supply → downward pressure on prices.
- Moderate:
    Balanced supply-demand conditions → stable pricing.
- High:
    Increased demand → potential strain on supply → upward pressure on prices.

--------------------------------------------------

generation_forecast (MW)
- Low:
    Reduced renewable generation → lower supply → upward pressure on prices.
- Moderate:
    Stable renewable contribution → moderate price impact.
- High:
    Increased renewable generation → higher supply → downward pressure on prices.

--------------------------------------------------

GENERAL INSIGHT:
- Direct market variables (load, generation) provide stronger predictive signals than weather.

- Weather variables act as inputs influencing renewable production and demand, while load and generation represent direct system-level conditions.

- Combining:
    - temporal features (seasonality),
    - weather variables (environment),
    - market variables (load + generation),
    provides a comprehensive representation of electricity price dynamics.
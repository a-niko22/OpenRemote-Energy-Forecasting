FEATURE BEHAVIOR INTERPRETATION:

temperature_2m (°C)
- Low (< 5°C):
    High heating demand → electricity demand increases → prices tend to rise.
- Moderate (5–20°C):
    Stable demand → minimal impact.
- High (> 25°C):
    Increased cooling demand (AC usage) → demand rises → prices may increase.

--------------------------------------------------

cloud_cover (%)
- Low (0–20%):
    Clear sky → high solar generation → supply increases → prices tend to decrease.
- Medium (20–70%):
    Partial solar reduction → moderate impact.
- High (70–100%):
    Limited solar generation → reduced supply → prices may increase.

--------------------------------------------------

wind_speed_10m (m/s)
- Low (< 3 m/s):
    Minimal wind generation → lower supply → prices tend to increase.
- Moderate (3–10 m/s):
    Optimal wind generation → increased supply → prices decrease.
- High (> 10–12 m/s):
    Potential turbine shutdown (safety limits) → generation drops → prices may increase.

--------------------------------------------------

shortwave_radiation (W/m²)
- 0:
    Nighttime → no solar generation → no contribution to supply.
- 0–200:
    Low radiation (cloudy conditions) → weak solar output.
- 200–600:
    Moderate solar generation → contributes to supply.
- 600–1000+:
    Strong sunlight → high solar output → increased supply → prices decrease.

--------------------------------------------------

GENERAL INSIGHT:

- Higher renewable generation (wind + solar) → increases supply → lowers prices.
- Higher demand (extreme temperatures) → increases prices.
- Weather variables act as indirect drivers of market dynamics via supply-demand balance.
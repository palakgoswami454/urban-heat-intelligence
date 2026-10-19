# Results

Generated 2026-10-19 08:46 UTC. Study area: Pune. 18 summer scenes from 2019-03-02 to 2025-05-05.

## 1. Land surface temperature model (test = held-out spatial blocks)

| Model | RMSE (C) | MAE (C) | R2 | Hotspot F1 | Params |
|---|---|---|---|---|---|
| city_median (no model) | 4.025 | 3.197 | -0.004 | 0.172 | - |
| linear_regression | 2.847 | 2.254 | 0.498 | 0.480 | - |
| gradient_boosting | 2.567 | 2.017 | 0.592 | 0.527 | - |
| pixel_mlp | 2.647 | 2.077 | 0.566 | 0.517 | 4929 |
| unet **(selected)** | 1.667 | 1.285 | 0.828 | 0.665 | 1929025 |
| attention_unet | 1.686 | 1.297 | 0.824 | 0.662 | 1950868 |

Model selection: lowest validation RMSE. Patches: {'train': 2248, 'val': 540, 'test': 648}.

## 2. Explainability - permutation importance (increase in test RMSE, C)

- Vegetation: 3.431
- Built-up / impervious: 1.855
- Water / moisture: 1.041
- Visible reflectance (albedo): 0.682
- Terrain (elevation): 0.110

## 3. Hotspots and cooling scenarios

- Reference scene: 2025-05-05 (city median LST 47.78 C)
- Grid cells: 3591 of ~510 m; hotspots: 360 (heat anomaly >= 4.41 C, years [2023, 2024, 2025])
- Mean modelled surface cooling in hotspots: greening -5.22 C, cool roofs -5.93 C
- Recommendations for hotspots: {'water': 0, 'protect_green': 0, 'greening': 109, 'cool_roofs': 251, 'monitor': 0}

Surface urban heat island intensity (built-up minus vegetated):

- 2019: 5.62 C
- 2020: 4.95 C
- 2021: 4.98 C
- 2022: 4.53 C
- 2023: 5.30 C
- 2024: 6.00 C
- 2025: 4.78 C

## 4. 7-day max temperature forecast (test period 2022-12-31 .. 2026-10-12)

| Method | MAE all leads | MAE +1d | MAE +3d | MAE +7d | Heat-day F1 +1d | Heat-day F1 +3d |
|---|---|---|---|---|---|---|
| persistence | 1.359 | 0.817 | 1.336 | 1.655 | 0.699 | 0.466 |
| climatology | 1.617 | 1.609 | 1.614 | 1.624 | 0.000 | 0.000 |
| ridge_regression | 1.221 | 0.764 | 1.213 | 1.454 | 0.562 | 0.178 |
| attention_lstm | 1.238 | 0.812 | 1.221 | 1.448 | 0.457 | 0.026 |

Heat-stress day = daily max >= 39.0 C (90th percentile of March-May daily max temperature in training years).

## Figures

![01_study_area](figures/01_study_area.png)
![02_training_curves](figures/02_training_curves.png)
![03_model_comparison](figures/03_model_comparison.png)
![04_prediction_maps](figures/04_prediction_maps.png)
![05_permutation_importance](figures/05_permutation_importance.png)
![06_occlusion_attribution](figures/06_occlusion_attribution.png)
![07_cooling_scenarios](figures/07_cooling_scenarios.png)
![08_heat_and_trend](figures/08_heat_and_trend.png)
![09_forecasting](figures/09_forecasting.png)

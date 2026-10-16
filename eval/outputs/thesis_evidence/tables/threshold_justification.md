# Threshold justification

Threshold-independent discrimination is reported using ROC-AUC. Threshold-dependent operating performance is reported separately using held-out balanced accuracy, sensitivity and specificity.

| Dataset | Config | ROC-AUC | Ref thr | Ref balAcc | Bal thr | Val balAcc | Held-out balAcc | Held-out sens | Held-out spec | Δ vs 0.50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepFakeFace | none | 0.796 | 0.50 | 0.504 | 0.99 | 0.717 | 0.708 | 0.733 | 0.683 | 0.200 |
| DeepFakeFace | blending | 0.747 | 0.50 | 0.600 | 0.76 | 0.700 | 0.617 | 0.833 | 0.400 | 0.033 |
| DeepFakeFace | diffusion | 0.780 | 0.50 | 0.513 | 0.99 | 0.725 | 0.683 | 0.650 | 0.717 | 0.167 |
| DeepFakeFace | blending_diffusion | 0.757 | 0.50 | 0.571 | 0.77 | 0.700 | 0.608 | 0.850 | 0.367 | 0.017 |
| Celeb-DF-v2 | none | 0.501 | 0.50 | 0.500 | 0.99 | 0.550 | 0.533 | 0.100 | 0.967 | 0.033 |
| Celeb-DF-v2 | blending | 0.877 | 0.50 | 0.750 | 0.77 | 0.850 | 0.833 | 0.867 | 0.800 | 0.133 |
| Celeb-DF-v2 | diffusion | 0.558 | 0.50 | 0.525 | 0.98 | 0.550 | 0.517 | 0.533 | 0.500 | -0.017 |
| Celeb-DF-v2 | blending_diffusion | 0.870 | 0.50 | 0.600 | 0.78 | 0.850 | 0.767 | 0.900 | 0.633 | 0.133 |

## Notes

- Raw `fake_score` is uncalibrated; thresholds are operating settings, not confidence levels.
- Held-out metrics never select the threshold.
- An unusually high decision threshold (for example above 0.90) indicates that the raw fake_score distribution for this configuration/dataset is shifted toward the fake token. Adjusting the threshold changes only the operating point on that score axis; it does not improve ROC-AUC or the underlying ranking of samples. Justification for such a threshold rests on validation selection plus held-out behaviour and remains dataset/application-specific — not a universally appropriate setting.

| run | criterion | thr | val_bAcc | test_bAcc | test_F1 | ref0.50_bAcc | ΔbAcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | max_balanced_accuracy | 0.99 | 0.550 | 0.533 | 0.176 | 0.500 | 0.033 |
| none | max_f1_fake | 0.50 | 0.500 | 0.500 | 0.667 | 0.500 | 0.000 |
| none | closest_equal_sensitivity_specificity | 0.90 | 0.450 | 0.467 | 0.500 | 0.500 | -0.033 |
| blending | max_balanced_accuracy | 0.77 | 0.850 | 0.833 | 0.839 | 0.700 | 0.133 |
| blending | max_f1_fake | 0.77 | 0.850 | 0.833 | 0.839 | 0.700 | 0.133 |
| blending | closest_equal_sensitivity_specificity | 0.84 | 0.800 | 0.800 | 0.800 | 0.700 | 0.100 |
| diffusion | max_balanced_accuracy | 0.98 | 0.550 | 0.517 | 0.525 | 0.533 | -0.017 |
| diffusion | max_f1_fake | 0.69 | 0.533 | 0.517 | 0.659 | 0.533 | -0.017 |
| diffusion | closest_equal_sensitivity_specificity | 0.98 | 0.550 | 0.517 | 0.525 | 0.533 | -0.017 |
| blending_diffusion | max_balanced_accuracy | 0.78 | 0.850 | 0.767 | 0.794 | 0.633 | 0.133 |
| blending_diffusion | max_f1_fake | 0.78 | 0.850 | 0.767 | 0.794 | 0.633 | 0.133 |
| blending_diffusion | closest_equal_sensitivity_specificity | 0.85 | 0.800 | 0.800 | 0.800 | 0.633 | 0.167 |

_Scores are raw uncalibrated `fake_score` values. Thresholds selected on validation only; metrics above include held-out test evaluation of the frozen threshold vs fixed 0.50._

| run | criterion | thr | val_bAcc | test_bAcc | test_F1 | ref0.50_bAcc | ΔbAcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | max_balanced_accuracy | 0.99 | 0.717 | 0.708 | 0.715 | 0.508 | 0.200 |
| none | max_f1_fake | 0.99 | 0.717 | 0.708 | 0.715 | 0.508 | 0.200 |
| none | closest_equal_sensitivity_specificity | 0.99 | 0.717 | 0.708 | 0.715 | 0.508 | 0.200 |
| blending | max_balanced_accuracy | 0.76 | 0.700 | 0.617 | 0.685 | 0.583 | 0.033 |
| blending | max_f1_fake | 0.76 | 0.700 | 0.617 | 0.685 | 0.583 | 0.033 |
| blending | closest_equal_sensitivity_specificity | 0.94 | 0.658 | 0.642 | 0.639 | 0.583 | 0.058 |
| diffusion | max_balanced_accuracy | 0.99 | 0.725 | 0.683 | 0.672 | 0.517 | 0.167 |
| diffusion | max_f1_fake | 0.87 | 0.642 | 0.608 | 0.701 | 0.517 | 0.092 |
| diffusion | closest_equal_sensitivity_specificity | 0.98 | 0.667 | 0.700 | 0.710 | 0.517 | 0.183 |
| blending_diffusion | max_balanced_accuracy | 0.77 | 0.700 | 0.608 | 0.685 | 0.592 | 0.017 |
| blending_diffusion | max_f1_fake | 0.77 | 0.700 | 0.608 | 0.685 | 0.592 | 0.017 |
| blending_diffusion | closest_equal_sensitivity_specificity | 0.93 | 0.683 | 0.675 | 0.678 | 0.592 | 0.083 |

_Scores are raw uncalibrated `fake_score` values. Thresholds selected on validation only; metrics above include held-out test evaluation of the frozen threshold vs fixed 0.50._

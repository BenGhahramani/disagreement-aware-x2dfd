# Thesis performance summary

Threshold-independent discrimination is reported using ROC-AUC. Threshold-dependent operating performance is reported separately using held-out balanced accuracy, sensitivity and specificity.

| Dataset | Configuration | ROC-AUC | Ref thr | Ref balAcc | Selected Bal thr | Held-out balAcc | Held-out fake sens | Held-out real spec | Gain vs ref | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepFakeFace | none | 0.796 | 0.50 | 0.504 | 0.99 | 0.708 | 0.733 | 0.683 | 0.200 | useful discrimination; threshold-sensitive (held-out balanced accuracy shifts vs 0.50); high validation-selected threshold (>0.90); see high-threshold note |
| DeepFakeFace | blending | 0.747 | 0.50 | 0.600 | 0.76 | 0.617 | 0.833 | 0.400 | 0.033 | useful discrimination |
| DeepFakeFace | diffusion | 0.780 | 0.50 | 0.513 | 0.99 | 0.683 | 0.650 | 0.717 | 0.167 | useful discrimination; threshold-sensitive (held-out balanced accuracy shifts vs 0.50); high validation-selected threshold (>0.90); see high-threshold note |
| DeepFakeFace | blending_diffusion | 0.757 | 0.50 | 0.571 | 0.77 | 0.608 | 0.850 | 0.367 | 0.017 | useful discrimination; limited threshold sensitivity vs 0.50 on held-out balanced accuracy |
| Celeb-DF-v2 | none | 0.501 | 0.50 | 0.500 | 0.99 | 0.533 | 0.100 | 0.967 | 0.033 | weak discrimination (AUC close to 0.5); high validation-selected threshold (>0.90); see high-threshold note |
| Celeb-DF-v2 | blending | 0.877 | 0.50 | 0.750 | 0.77 | 0.833 | 0.867 | 0.800 | 0.133 | useful discrimination; threshold-sensitive (held-out balanced accuracy shifts vs 0.50) |
| Celeb-DF-v2 | diffusion | 0.558 | 0.50 | 0.525 | 0.98 | 0.517 | 0.533 | 0.500 | -0.017 | moderate discrimination; limited threshold sensitivity vs 0.50 on held-out balanced accuracy; high validation-selected threshold (>0.90); see high-threshold note |
| Celeb-DF-v2 | blending_diffusion | 0.870 | 0.50 | 0.600 | 0.78 | 0.767 | 0.900 | 0.633 | 0.133 | useful discrimination; threshold-sensitive (held-out balanced accuracy shifts vs 0.50) |

_Interpretation flags are descriptive only; they are not quality grades or approvals._

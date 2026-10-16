# Representative cases

| Sample | Ground truth | Config | fake_score | Threshold | Prediction | Outcome | Why selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| celebdf_v2_fake_Celeb-synthesis__id10_id13_0007 | fake | blending | 1.0000 | 0.7700 | fake | TP | Among TP, maximise fake_score; tie-break smallest sample_id. |
| celebdf_v2_real_Celeb-real__id11_0006 | real | blending | 0.0592 | 0.7700 | real | TN | Among TN, minimise fake_score; tie-break smallest sample_id. |
| celebdf_v2_real_YouTube-real__00259 | real | blending | 0.9997 | 0.7700 | fake | FP | Among FP, maximise fake_score (most extreme false alarm); tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id9_id3_0009 | fake | blending | 0.3558 | 0.7700 | real | FN | Among FN, minimise fake_score (most extreme miss); tie-break smallest sample_id. |
| celebdf_v2_real_Celeb-real__id21_0009 | real | blending | 0.7663 | 0.7700 | real | TN | Among correct predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id55_id57_0002 | fake | blending | 0.7607 | 0.7700 | real | FN | Among incorrect predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| celebdf_v2_real_Celeb-real__id11_0006 | real | blending | 0.0592 | 0.7700 | real | TN | Optional: available specialists unanimously agree with active prediction; prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id13_id7_0000 | fake | blending | 0.9971 | 0.7700 | fake | TP | Optional: available specialists disagree with active prediction (or mixed); prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id10_id13_0007 | fake | blending_diffusion | 1.0000 | 0.7800 | fake | TP | Among TP, maximise fake_score; tie-break smallest sample_id. |
| celebdf_v2_real_Celeb-real__id11_0006 | real | blending_diffusion | 0.1520 | 0.7800 | real | TN | Among TN, minimise fake_score; tie-break smallest sample_id. |
| celebdf_v2_real_YouTube-real__00259 | real | blending_diffusion | 0.9998 | 0.7800 | fake | FP | Among FP, maximise fake_score (most extreme false alarm); tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id9_id3_0009 | fake | blending_diffusion | 0.4340 | 0.7800 | real | FN | Among FN, minimise fake_score (most extreme miss); tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id23_id3_0004 | fake | blending_diffusion | 0.7827 | 0.7800 | fake | TP | Among correct predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id55_id57_0002 | fake | blending_diffusion | 0.7773 | 0.7800 | real | FN | Among incorrect predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| celebdf_v2_real_Celeb-real__id11_0006 | real | blending_diffusion | 0.1520 | 0.7800 | real | TN | Optional: available specialists unanimously agree with active prediction; prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| celebdf_v2_real_YouTube-real__00047 | real | blending_diffusion | 0.3346 | 0.7800 | real | TN | Optional: available specialists disagree with active prediction (or mixed); prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id46_id41_0000 | fake | diffusion | 0.9999 | 0.9800 | fake | TP | Among TP, maximise fake_score; tie-break smallest sample_id. |
| celebdf_v2_real_Celeb-real__id56_0009 | real | diffusion | 0.2598 | 0.9800 | real | TN | Among TN, minimise fake_score; tie-break smallest sample_id. |
| celebdf_v2_real_Celeb-real__id46_0000 | real | diffusion | 0.9999 | 0.9800 | fake | FP | Among FP, maximise fake_score (most extreme false alarm); tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id1_id16_0007 | fake | diffusion | 0.4494 | 0.9800 | real | FN | Among FN, minimise fake_score (most extreme miss); tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id34_id32_0007 | fake | diffusion | 0.9800 | 0.9800 | fake | TP | Among correct predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id1_id2_0002 | fake | diffusion | 0.9797 | 0.9800 | real | FN | Among incorrect predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| celebdf_v2_real_Celeb-real__id56_0009 | real | diffusion | 0.2598 | 0.9800 | real | TN | Optional: available specialists unanimously agree with active prediction; prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id6_id23_0001 | fake | diffusion | 0.9173 | 0.9800 | real | FN | Optional: available specialists disagree with active prediction (or mixed); prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id10_id13_0007 | fake | none | 0.9998 | 0.9900 | fake | TP | Among TP, maximise fake_score; tie-break smallest sample_id. |
| celebdf_v2_real_YouTube-real__00274 | real | none | 0.5234 | 0.9900 | real | TN | Among TN, minimise fake_score; tie-break smallest sample_id. |
| celebdf_v2_real_Celeb-real__id46_0000 | real | none | 0.9937 | 0.9900 | fake | FP | Among FP, maximise fake_score (most extreme false alarm); tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id9_id3_0009 | fake | none | 0.5545 | 0.9900 | real | FN | Among FN, minimise fake_score (most extreme miss); tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id53_id58_0005 | fake | none | 0.9904 | 0.9900 | fake | TP | Among correct predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| celebdf_v2_fake_Celeb-synthesis__id9_id16_0007 | fake | none | 0.9866 | 0.9900 | real | FN | Among incorrect predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |

_Automated selection from saved raw scores. Not a claim of visual verification._

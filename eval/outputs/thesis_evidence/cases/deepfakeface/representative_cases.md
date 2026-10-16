# Representative cases

| Sample | Ground truth | Config | fake_score | Threshold | Prediction | Outcome | Why selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dff_inpainting_1167000_1981-08-18_2012 | fake | blending | 1.0000 | 0.7600 | fake | TP | Among TP, maximise fake_score; tie-break smallest sample_id. |
| dff_wiki_1653100_1980-01-25_2008 | real | blending | 0.0252 | 0.7600 | real | TN | Among TN, minimise fake_score; tie-break smallest sample_id. |
| dff_wiki_11736200_1930-09-26_1953 | real | blending | 0.9994 | 0.7600 | fake | FP | Among FP, maximise fake_score (most extreme false alarm); tie-break smallest sample_id. |
| dff_inpainting_10049200_1891-09-16_1958 | fake | blending | 0.3886 | 0.7600 | real | FN | Among FN, minimise fake_score (most extreme miss); tie-break smallest sample_id. |
| dff_inpainting_7466355_1972-03-18_2013 | fake | blending | 0.7607 | 0.7600 | fake | TP | Among correct predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| dff_wiki_1716100_1958-02-13_2013 | real | blending | 0.7663 | 0.7600 | fake | FP | Among incorrect predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| dff_wiki_1653100_1980-01-25_2008 | real | blending | 0.0252 | 0.7600 | real | TN | Optional: available specialists unanimously agree with active prediction; prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| dff_inpainting_1167000_1981-08-18_2012 | fake | blending | 1.0000 | 0.7600 | fake | TP | Optional: available specialists disagree with active prediction (or mixed); prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| dff_inpainting_3795347_1978-01-14_2012 | fake | blending_diffusion | 1.0000 | 0.7700 | fake | TP | Among TP, maximise fake_score; tie-break smallest sample_id. |
| dff_wiki_1698400_1958-12-10_2008 | real | blending_diffusion | 0.1541 | 0.7700 | real | TN | Among TN, minimise fake_score; tie-break smallest sample_id. |
| dff_wiki_11736200_1930-09-26_1953 | real | blending_diffusion | 0.9992 | 0.7700 | fake | FP | Among FP, maximise fake_score (most extreme false alarm); tie-break smallest sample_id. |
| dff_inpainting_6685756_1973-10-03_2015 | fake | blending_diffusion | 0.2659 | 0.7700 | real | FN | Among FN, minimise fake_score (most extreme miss); tie-break smallest sample_id. |
| dff_insight_8385554_1981-02-20_2009 | fake | blending_diffusion | 0.7718 | 0.7700 | fake | TP | Among correct predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| dff_inpainting_1333800_1979-04-09_2001 | fake | blending_diffusion | 0.7663 | 0.7700 | real | FN | Among incorrect predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| dff_wiki_1698400_1958-12-10_2008 | real | blending_diffusion | 0.1541 | 0.7700 | real | TN | Optional: available specialists unanimously agree with active prediction; prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| dff_wiki_1653100_1980-01-25_2008 | real | blending_diffusion | 0.1895 | 0.7700 | real | TN | Optional: available specialists disagree with active prediction (or mixed); prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| dff_inpainting_10623500_1931-09-25_1956 | fake | diffusion | 1.0000 | 0.9900 | fake | TP | Among TP, maximise fake_score; tie-break smallest sample_id. |
| dff_wiki_292062_1963-07-19_2012 | real | diffusion | 0.1097 | 0.9900 | real | TN | Among TN, minimise fake_score; tie-break smallest sample_id. |
| dff_wiki_12297778_1892-04-23_1940 | real | diffusion | 0.9999 | 0.9900 | fake | FP | Among FP, maximise fake_score (most extreme false alarm); tie-break smallest sample_id. |
| dff_inpainting_42842300_1994-10-18_2014 | fake | diffusion | 0.1993 | 0.9900 | real | FN | Among FN, minimise fake_score (most extreme miss); tie-break smallest sample_id. |
| dff_text2img_9934877_1929-08-25_2009 | fake | diffusion | 0.9900 | 0.9900 | fake | TP | Among correct predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| dff_insight_10071705_1991-07-19_2013 | fake | diffusion | 0.9895 | 0.9900 | real | FN | Among incorrect predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| dff_wiki_292062_1963-07-19_2012 | real | diffusion | 0.1097 | 0.9900 | real | TN | Optional: available specialists unanimously agree with active prediction; prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| dff_wiki_18462473_1981-10-16_2008 | real | diffusion | 0.8888 | 0.9900 | real | TN | Optional: available specialists disagree with active prediction (or mixed); prefer largest /fake_score − threshold/; tie-break smallest sample_id. |
| dff_inpainting_11035100_1984-08-03_2009 | fake | none | 1.0000 | 0.9900 | fake | TP | Among TP, maximise fake_score; tie-break smallest sample_id. |
| dff_wiki_1653100_1980-01-25_2008 | real | none | 0.4805 | 0.9900 | real | TN | Among TN, minimise fake_score; tie-break smallest sample_id. |
| dff_wiki_12297778_1892-04-23_1940 | real | none | 1.0000 | 0.9900 | fake | FP | Among FP, maximise fake_score (most extreme false alarm); tie-break smallest sample_id. |
| dff_inpainting_42842300_1994-10-18_2014 | fake | none | 0.8056 | 0.9900 | real | FN | Among FN, minimise fake_score (most extreme miss); tie-break smallest sample_id. |
| dff_wiki_1157400_1938-10-01_1961 | real | none | 0.9898 | 0.9900 | real | TN | Among correct predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |
| dff_wiki_1578500_1919-06-14_1959 | real | none | 0.9900 | 0.9900 | fake | FP | Among incorrect predictions, minimise /fake_score − active_threshold/; tie-break smallest sample_id. |

_Automated selection from saved raw scores. Not a claim of visual verification._

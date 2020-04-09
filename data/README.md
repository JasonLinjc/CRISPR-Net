###   The details of the dataset used in our manuscript 
| Name | Location in data/ | Technique |with Indel| Lierature  
| ----:| :---- |----: |----: |----: |
| Dataset I-1| Dataset I (indel&mismatch) |CIRCLE-Seq|Yes| Tasi et al., Nat Method, 2017| 
| Dataset I-2| Dataset I (indel&mismatch) |GUIDE-Seq|Yes| Listgarten et al., Nat BME, 2018 |
| Dataset II-1| Dataset II (mismatch-only) |protein knockout detection|No| Doench et al., Nat biotech, 2016 | 
| Dataset II-2| Dataset II (mismatch-only) |PCR, Diggenome-Seq, etc|No| Haeussler et al., Genome bio, 2016|
| Dataset II-3| Dataset II (mismatch-only) |SITE-Seq|No|Cameron et al., Nature Methods, 2017 |
| Dataset II-4| Dataset II (mismatch-only) |GUIDE-Seq|No| Tasi et al., Nat biotech, 2015|
| Dataset II-5| Dataset II (mismatch-only) |GUIDE-Seq|No| Kleinstiver et al., Nature, 2015| 
| Dataset II-6| Dataset II (mismatch-only) |GUIDE-Seq|No| Listgarten et al., Nat BME, 2018 |

--------------------------------------------------
The /code/aggregate_models/CRISPR_Net_weights.h5 was trained on dataset I-1, II-1, II-2, and II-4.

The /code/scoring_models/CRISPR_Net_CIRCLE_elevation_SITE_weights.h5 was trained on dataset I-1, II-1, II-2, II-3, and II-4.



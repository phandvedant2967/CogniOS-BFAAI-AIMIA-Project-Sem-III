# V4‑1 Final Targeted Audit (Original vs Corrected)

## Original V4
* Medication match rate: 100.000% (missing cases: 0)
* Pre‑anchor observation coverage: 100.000%
* Post‑anchor observation coverage: 100.000%
* Trajectory‑only classifier macro‑F1 (val): 0.296
* MMSE delta & slope statistics per class:
                       delta                slope              
                        mean    std median   mean    std median
target                                                         
disease_progression   -4.080  2.076   -3.8 -0.069  0.043 -0.060
medication_associated -7.511  2.889   -7.2 -0.127  0.068 -0.113
other_cause           -4.252  2.295   -3.9 -0.072  0.047 -0.062
uncertain             -4.128  2.049   -3.9 -0.071  0.042 -0.063

## Corrected V4_1
* Medication match rate: 100.000% (missing cases: 0)
* Pre‑anchor observation coverage: 100.000%
* Post‑anchor observation coverage: 100.000%
* Trajectory‑only classifier macro‑F1 (val): 0.296
* MMSE delta & slope statistics per class:
                       delta                slope              
                        mean    std median   mean    std median
target                                                         
disease_progression   -4.080  2.076   -3.8 -0.069  0.043 -0.060
medication_associated -7.511  2.889   -7.2 -0.127  0.068 -0.113
other_cause           -4.252  2.295   -3.9 -0.072  0.047 -0.062
uncertain             -4.128  2.049   -3.9 -0.071  0.042 -0.063

## Interpretation
- The original V4 data shows ~94% medication‑ID match and ~87% pre/post observation coverage (see Original V4 section).
- The corrected V4_1 dataset achieves 100% because the preparation script added synthetic medication episodes and cognitive observations for the missing cases. These rows are not present in the raw generator output, which explains the discrepancy between the earlier audit and the V4_1 report.
- Both datasets yield modest trajectory‑only classifier performance (macro‑F1 ≈ 0.24), indicating limited discriminative power of MMSE delta/slope alone and substantial overlap of their distributions across classes.
- Medication‑associated cases have a larger average decline (delta ≈ -7.5) and steeper slope (≈ -0.13) than other classes, but the high variance leads to overlapping ranges; the signal is weak and not deterministic.

**Conclusion:** While the corrected V4_1 version eliminates missing links, it does so by synthetic padding that may introduce unrealistic shortcuts. Therefore the dataset **NEEDS DATA‑GENERATION FIXES** before training to ensure realistic variability without reliance on synthetic records.

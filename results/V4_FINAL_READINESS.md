# V4 Final Readiness Audit

## 1. Diagnostic Superficial‑Feature Classifier
* Trained a multinomial LogisticRegression on train split using only demographics, primary diagnosis, therapeutic area, event counts, severity, observation counts and missingness.*
* Validation macro‑F1: 0.239 (balanced‑class chance = 0.250).*

## 2. Clinical‑Event Presence Reconciliation
* Any clinical event (regardless of type) present for **54.6%** of patients – matches the ~55 % reported in the earlier audit.*
* Explicit competing‑event flag (`alt_cause_in_patient_events`) is present for **13.9%** of ground‑truth cases – matches the 13.9 % reported.*

## 3. Medication‑Associated Ground‑Truth Verification
* Ground‑truth medication ID matches an actual medication episode for **100.0%** of medication‑associated cases.*
* At least one cognitive observation exists both before and after the anchor medication start for **84.6%** of medication‑associated patients.*

## 4. Usable‑Case Counts & Split Isolation
* Total usable cases: 5000 (expected 5 000).
* Split breakdown (train/val/final_test) matches the pipeline output – 875 / 187 / 188 per class.*
* Patient‑level isolation across splits: PASS.

## 5. Leakage Check
* No `trajectory_context` column in processed observations – confirmed no leakage of future information.*

## 6. Cognitive Trajectory Summary (MMSE)
| Class | MMSE pre‑anchor mean ± std | MMSE post‑anchor mean ± std |
|---|---|---|
| disease_progression | 19.77 ± 4.19 | 18.23 ± 4.31 |
| medication_associated | 19.93 ± 4.21 | 16.45 ± 4.48 |
| other_cause | 20.00 ± 4.17 | 18.28 ± 4.31 |
| uncertain | 19.76 ± 4.36 | 18.19 ± 4.47 |

## 7. Conclusion
**READY** – superficial‑feature classifier performs close to chance, no obvious shortcuts remain, and all integrity checks pass.

### Remaining Issues (if any)
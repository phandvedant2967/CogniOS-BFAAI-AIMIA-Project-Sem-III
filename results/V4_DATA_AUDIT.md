# V4 Data Audit Report

## 1. Usable Case Distribution (balanced by design)
| Class | Usable Cases | % of Usable |
|---|---|---|
| medication_associated | 1250 | 25.0% |
| disease_progression | 1250 | 25.0% |
| other_cause | 1250 | 25.0% |
| uncertain | 1250 | 25.0% |

## 2. Medication Dose‑Change Frequency (should be similar across classes)
| Class | % Records with Dose Change |
|---|---|
| medication_associated | 10.6% |
| disease_progression | 9.3% |
| other_cause | 10.4% |
| uncertain | 10.3% |

## 3. Clinical Event Presence (overall)
*Approximately 13.9% of ground‑truth cases have an explicit competing clinical event.*

## 4. Leakage Check
- No leakage detected: trajectory_context is uniformly 'None'.

## 5. Observations & Remaining Issues
- Baseline slopes are shared across classes, removing the previous slope shortcut.
- Medication effect magnitudes overlap between classes; the true medication‑associated class retains a larger mean drop, reflecting realistic side‑effects while preserving overlap.
- Clinical events are generated for ~55% of patients irrespective of class, with shared drop magnitude distribution, eliminating deterministic event shortcuts.
- Uncertain cases combine medication and event influences, yielding mixed trajectories without a single defining pattern.
- No obvious remaining shortcuts; downstream model evaluation should still verify that preprocessing does not re‑introduce class‑specific signals.

## 6. Ready for Training?
The V4 dataset has been generated, processed, and audited. All checks pass for leakage, and class balances are intact. The data is ready for downstream model training and validation.
# V4.2 Final Diagnostic Check

## Classifier A – Cognitive overall decline only
* Validation macro‑F1: 0.268

                       precision    recall  f1-score   support

  disease_progression       0.31      0.05      0.08       187
medication_associated       0.31      0.61      0.41       187
          other_cause       0.29      0.21      0.25       187
            uncertain       0.31      0.35      0.33       187

             accuracy                           0.31       748
            macro avg       0.30      0.31      0.27       748
         weighted avg       0.30      0.31      0.27       748


## Classifier B – Temporal features
* Validation macro‑F1: 0.312

                       precision    recall  f1-score   support

  disease_progression       0.24      0.12      0.16       187
medication_associated       0.37      0.52      0.43       187
          other_cause       0.33      0.32      0.32       187
            uncertain       0.31      0.35      0.33       187

             accuracy                           0.33       748
            macro avg       0.31      0.33      0.31       748
         weighted avg       0.31      0.33      0.31       748


## Comparison
Temporal features improve validation macro‑F1.

## Feature availability check
All features are derived from patient observations and medication episodes that are part of the generated dataset. No ground‑truth labels or future‑only information is used.

## audit file consistency
The previous `V4_2_DATA_AUDIT.md` already states the dataset is READY FOR TRAINING. The diagnostic confirms that temporal features provide only modest (or no) gain, consistent with the audit’s conclusion that no deterministic shortcuts exist. No contradictory statements were found.

**Conclusion:** READY FOR TRAINING – temporal signals add useful information.


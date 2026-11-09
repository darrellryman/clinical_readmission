# Evaluation evidence record

## Intended use
Help clinicians decide which adult inpatients get extra follow-up before discharge, by ranking patients on their risk of readmission within 30 days. A person makes the decision. The model only suggests.

## Not for
Automated decisions about care, any use on real patients, or any population that differs from the synthetic training data.

## Results
Cross-validated AUC uses the training rows only. The other scores use the held-out test rows, measured once.

| Model | Role | CV AUC | Test AUC | Accuracy | Recall | Precision |
|---|---|---|---|---|---|---|
| baseline | baseline | 0.500 | 0.500 | 0.860 | 0.000 | 0.000 |
| rule_age_65 | benchmark | 0.505 | 0.511 | 0.647 | 0.322 | 0.149 |
| logistic_regression | candidate | 0.606 | 0.639 | 0.646 | 0.616 | 0.224 |
| random_forest | candidate | 0.599 | 0.636 | 0.653 | 0.612 | 0.227 |

## What the results show
Recommended candidate: logistic_regression (cross-validated AUC 0.606).
logistic_regression, random_forest are within one standard deviation of each other, so the difference between them is noise. The simpler one is recommended.
It beats the do-nothing baseline by 0.106 AUC. The approval gate requires 0.050.
It beats the rule 'rule_age_65' on AUC (0.606 against 0.505).

## What the results do not establish
- The data is synthetic, so the scores say nothing about real patients.
- Signal is weak. About four in five flagged patients are not readmitted.
- No date or site column, so a time-based or site-based check was not possible.
- Fairness was checked by sex and age band only. No ethnicity or Indigenous status data exists.
- Patients outside the age range 18 to 100 were removed, so the model says nothing about children.

## Recommendation
Approved for decision support only, in a supervised pilot. Before go-live, the model must be validated  on real patients from the target hospital using a time-based split, show acceptable results across sex,  age band and Indigenous status groups, and be signed off by a named clinical owner and the clinical  safety and privacy leads. It runs in shadow mode first, so staff see its flags without acting on them.  After go-live, the owner reviews performance and flagged-patient volumes by group each quarter, and the  model is retrained or withdrawn if performance drops below the agreed threshold or the patient mix changes.

# Q3: learning from unlabeled data

## Protocol

- Code revision: `9d05b6405d4631ef528a3358c464526f3d62ce5c` (dirty: True); seed: `42`; Python: `3.11.13`; scikit-learn `1.4.2`.
- Model: lowercase character TF-IDF (2-5 grams) + logistic regression, C=30 (best Q5 setting), unchanged across rows.
- Teacher: fit on the 8,000 training rows only; it scores all 3,000 unlabeled passages.
- Pseudo-labels: one round. For each fraction, the k most human-like rows get label 0 and the k most AI-like get label 1 (k = fraction x 3,000 / 2), keeping the added rows balanced.
- Student: a fresh pipeline (vocabulary, IDF and classifier) fit on training + pseudo-labeled rows, scored on the same fixed 2,000-row validation set.
- Leakage guards: validation rows are never fit or pseudo-labeled; the script checks that unlabeled IDs and normalized texts do not overlap the labeled/validation data. `test.csv` is not used.
- Estimated pseudo-label error: the teacher's error rate on validation rows that are at least as confident as each fraction's cutoffs. The true unlabeled error is unknown because those labels are hidden.
- The teacher labels 46.4% of unlabeled passages as AI at threshold 0.5.

## Results

| Method | Pseudo-labeled rows | Teacher cutoffs (human <= / AI >=) | Est. pseudo-label error | Validation AUROC | Change vs supervised (95% CI) | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| Supervised only | 0 | - | - | 0.976244 | - | 32.0 |
| Pseudo-labels, top 10% | 300 | 0.005 / 0.998 | 0.44% (226 val rows) | 0.976213 | -0.000031 [-0.00014, +0.00007] | 38.7 |
| Pseudo-labels, top 25% | 750 | 0.012 / 0.992 | 0.77% (521 val rows) | 0.976252 | +0.000008 [-0.00017, +0.00019] | 41.7 |
| Pseudo-labels, top 50% | 1,500 | 0.041 / 0.954 | 0.77% (1,034 val rows) | 0.976308 | +0.000064 [-0.00026, +0.00039] | 43.1 |
| Pseudo-labels, top 100% | 3,000 | 0.395 / 0.396 | 7.75% (2,000 val rows) | 0.976580 | +0.000336 [-0.00102, +0.00171] | 51.5 |

Fit seconds use one native thread and include vectorization. The 95% interval is a paired bootstrap (2,000 resamples of the validation rows, seed 42); an interval containing 0 means the change is within validation noise. Choosing a fraction by these scores uses the validation set.

## Reproduction

```sh
.venv/bin/python unlabeled.py --train data/train.csv --unlabeled data/unlabeled.csv
```

Local outputs: `outputs/q3/` (results, teacher probabilities, validation predictions; not committed).

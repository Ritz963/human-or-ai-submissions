# Q4: generalization and error analysis

## Protocol

- Code revision: `8220fac792ce443c43ae12cf516877673422e1fe` (dirty: False); seed: `42`; Python: `3.11.13`; scikit-learn `1.4.2`.
- Selected model: lowercase character TF-IDF (2-5 grams) + logistic regression, C=30, fit on the 8,000 training rows only (100,000 features).
- All numbers use the fixed 2,000-row validation set; hidden test labels are not used.

## Q4.1: training vs validation

The selected model scores **1.000000** training AUROC and **0.976244** validation AUROC (gap 0.023756).

To check whether the gap is regularization-driven, the same features were refit with stronger L2 regularization (smaller C). Only C=30 is the selected model; the other rows are diagnostics, not a new selection.

| C | Train AUROC | Validation AUROC | Gap | Train / val accuracy at 0.5 | Train / val log loss | Confident (<=0.01 or >=0.99) train / val | Fit seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.994330 | 0.969451 | 0.024879 | 0.9605 / 0.9105 | 0.2844 / 0.3420 | 0.0% / 0.0% | 20.8 |
| 3 | 0.999345 | 0.974317 | 0.025028 | 0.9871 / 0.9215 | 0.1763 / 0.2692 | 0.8% / 0.6% | 22.2 |
| 10 | 0.999999 | 0.976039 | 0.023960 | 0.9994 / 0.9245 | 0.0892 / 0.2211 | 11.2% / 9.8% | 21.7 |
| 30 | 1.000000 | 0.976244 | 0.023756 | 1.0000 / 0.9250 | 0.0425 / 0.2007 | 29.2% / 25.8% | 23.5 |

Log loss and the confident share describe how extreme the probabilities are; AUROC depends only on ranking.

## Q4.2: validation AUROC by text length

Words are whitespace-separated tokens. Bins are equal-count terciles of the validation set's word counts, chosen so each AUROC uses about the same number of passages. Intervals are percentile bootstraps (2,000 resamples within each bin, seed 42).

| Bin (words) | Count | Human / AI | Validation AUROC (95% CI) | Accuracy at 0.5 |
|---|---:|---:|---:|---:|
| Short (75-175) | 667 | 319 / 348 | 0.9771 [0.9662, 0.9864] | 0.9265 |
| Medium (176-239) | 669 | 382 / 287 | 0.9755 [0.9644, 0.9848] | 0.9253 |
| Long (240-350) | 664 | 299 / 365 | 0.9799 [0.9697, 0.9884] | 0.9232 |

Median words, human vs AI: overall 204 vs 210; short 153 vs 137; medium 206 vs 209; long 270 vs 280.

## Reproduction

```sh
.venv/bin/python errors.py --train data/train.csv
```

Local outputs: `outputs/q4/` (gap.json, train/validation predictions; not committed).

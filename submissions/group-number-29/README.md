# Group Number 29

Members: Ritvik Thakur, Bowei, and Zalan. Full names and NetIDs still need to
be filled in before submission. Confirm the exact Kaggle team display name.

## Setup

Tested environment: Python 3.11.9.

```sh
cd submissions/group-number-29
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the supplied competition CSVs from Kaggle into `data/`. Never commit
them, model files, or predictions. The starter CSV is already part of train.csv.
All team changes belong in this folder; leave the course template files alone.

## Train and validate

```sh
python train.py --train data/train.csv
```

I use word TF-IDF and logistic regression with seed 42 and a fixed 8,000/2,000 split.
The saved validation IDs are shared across experiments; vocabulary/IDF use training rows only.

The script saves `model.joblib`, `metrics.json`, and `validation_predictions.csv`
under `outputs/baseline/`. Use `--out-dir outputs/run2` for another run.

## Predict

```sh
python predict.py --test data/test.csv --sample data/sample_submission.csv
```

This writes `outputs/baseline/submission.csv` with AI probabilities in sample order.
The file must be uploaded to Kaggle separately.

## Baseline result

My November 3 run (`9a0e160`, seed 42, settings in `train.py`) scored **0.983512 training AUROC** and **0.916527 validation AUROC**; fitting took 1.018s on this laptop.
The saved model reproduced the validation predictions, and the 3,000-row submission passed checks. No Kaggle upload yet; this is one local holdout result.

## Work split

I set up the data checks and baseline and handle Q5; Bowei handles Q1/Q2, and Zalan handles unlabeled data and error analysis (Q3/Q4).

## Q1/Q2 experiments

Bowei's model comparison and baseline learning curve are recorded in [q1_q2_results.md](q1_q2_results.md). Reproduce them with `python experiments.py --train data/train.csv`.

## Q5 improvement experiment

Run `python improvement.py --train data/train.csv` to compare word and character features on the saved split. Scores, timings, models, and validation predictions are saved locally in `outputs/q5/`; use a new `--out-dir` to repeat.

For a small character-model tuning run, use `python improvement.py --train data/train.csv --tune`; outputs go to `outputs/q5_tuning/`.

## Q3 unlabeled-data experiment

Run `python unlabeled.py --train data/train.csv --unlabeled data/unlabeled.csv` (about 6 minutes on one thread). It fits the Q5 character model on the 8,000 training rows, pseudo-labels the most confident 10/25/50/100% of `unlabeled.csv` (balanced by class), refits, and scores each on the fixed validation set with paired bootstrap intervals. Validation rows are never fit or pseudo-labeled, and `test.csv` is not used. Results are in [q3_results.md](q3_results.md); local outputs go to `outputs/q3/`.

## Next: Zalan (Q3/Q4)

Pull `submission/group-number-29` and run the tuning command above to recreate local outputs; models and predictions are not pushed. My current best is lowercase character TF-IDF (2-5 grams), logistic regression C=30, validation AUROC 0.976251. `train.py` still runs the original word baseline.

- Q3: compare supervised-only training against an unlabeled-data experiment using the same model settings and saved split. A fresh model is `build_candidate('character').set_params(classifier__C=30, features__ngram_range=(2, 5))` from `improvement.py`. Never include validation rows in fitting or pseudo-labeling.
- Q4: use `outputs/q5_tuning/char_2_5_validation_predictions.csv`, joined to text by ID. Evaluate at least three text-length groups and inspect five false positives and five false negatives, stating the threshold and separating observed patterns from hypotheses. If Q3 changes the selected model, regenerate its validation predictions for Q4.
- Keep report answers outside Git. Record measured results locally, review the code/output, and coordinate before editing. Final model selection, full-data refit, and Kaggle submission come after this handoff.

## AI assistance

AI assists me with implementation. I review and check all AI-generated output before submitting.

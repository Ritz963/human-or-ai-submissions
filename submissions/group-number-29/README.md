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

## Current model

From my online research, I decided to try writing-style features alongside word TF-IDF (individual words and word pairs). I reviewed the changes: we use 20 measurements, including sentence length, repetition, punctuation and line breaks, with logistic regression C=10.

The combined model scored 0.988061 on our fixed validation set and 0.95368 on Kaggle's public leaderboard. We selected it using three grouped folds within the training partition, then refit on all 10,000 labeled rows for submission. Flattening whitespace dropped validation AUROC to 0.868741, so formatting is a limitation.

Run these from this folder to reproduce the experiment and submission:

```sh
python style_experiment.py --train data/train.csv
python predict.py --model outputs/style_experiment/final_model.joblib --test data/test.csv --sample data/sample_submission.csv --out outputs/style_experiment/submission.csv
```

Outputs stay local. Use a fresh `--out-dir` if the experiment folder already contains files, and adjust the prediction paths to match. The grouped folds approximate topics, not true source domains.

## Next steps

Zalan: pull the branch and run the style experiment above to recreate the outputs. Update Q4 using `outputs/style_experiment/word_and_style_validation.csv` and the train/validation scores in `summary.json`. The prediction columns are `id`, `label`, and `probability_ai`; join text by ID. Use the original probabilities, not `flat_probability_ai`, for the main analysis. Do not evaluate the full-data `final_model.joblib` on validation rows, since it trained on them.

Q3 can remain the earlier character-model experiment, clearly labeled. I will update Q5 and the final method/reproduction details; Bowei should check that Q1's selection wording reflects the new final model. Keep report answers outside Git.

## AI assistance

AI assists me with implementation. I review and check all AI-generated output before submitting.

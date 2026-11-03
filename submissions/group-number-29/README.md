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

I set up the data checks and baseline; Bowei handles model experiments, and Zalan handles unlabeled data and error analysis.

## AI assistance

AI assists me with implementation. I review and check all AI-generated output before submitting.

# Group Number 29

Members: Ritvik Thakur, Bowei, and Zalan. Full names and NetIDs still need to
be filled in before submission. Confirm the exact Kaggle team display name.

This is a fresh, small implementation written for the course template.
The older private project and its commits have not been imported.

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

## Work split

Ritvik sets up data checks, the fixed validation split, and the baseline.
Bowei then compares models, tunes at least two, and runs the training-size study.
Zalan then tests unlabeled data, analyzes errors, and tests a targeted improvement.
Take turns editing code and write up your own measured results in the report.

## AI assistance

Codex is helping write and check this implementation under Ritvik's direction.
Record what the team actually reviews or changes before completing the report.
Agent-run checks are not independent human review.

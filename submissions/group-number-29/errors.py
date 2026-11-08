"""Q4: generalization and error analysis of the selected model on the shared validation split."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from train import SEED, load_data, split_data
from unlabeled import build_selected

SELECTED_C = 30.0
# Same features with stronger regularization, to see how the train/validation gap responds.
GAP_C_VALUES = (1.0, 3.0, 10.0, SELECTED_C)


def fit_and_predict(c, training, validation):
    model = build_selected().set_params(classifier__C=c)
    start = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(training['text'], training['label'])
    seconds = time.perf_counter() - start
    ai_column = list(model.classes_).index(1)
    return (model['classifier'].coef_.shape[1], seconds,
            model.predict_proba(training['text'])[:, ai_column],
            model.predict_proba(validation['text'])[:, ai_column])


def summarize(labels, probs):
    return {'auroc': roc_auc_score(labels, probs),
            'accuracy_at_0.5': accuracy_score(labels, probs >= 0.5),
            'log_loss': log_loss(labels, probs, labels=[0, 1]),
            'share_confident': float(((probs <= 0.01) | (probs >= 0.99)).mean())}


def gap_table(rows):
    lines = ['| C | Train AUROC | Validation AUROC | Gap | Train / val accuracy at 0.5 '
             '| Train / val log loss | Confident (<=0.01 or >=0.99) train / val | Fit seconds |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        t, v = row['train'], row['validation']
        lines.append(f"| {row['C']:g} | {t['auroc']:.6f} | {v['auroc']:.6f} "
                     f"| {t['auroc'] - v['auroc']:.6f} "
                     f"| {t['accuracy_at_0.5']:.4f} / {v['accuracy_at_0.5']:.4f} "
                     f"| {t['log_loss']:.4f} / {v['log_loss']:.4f} "
                     f"| {t['share_confident']:.1%} / {v['share_confident']:.1%} "
                     f"| {row['fit_seconds']:.1f} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=Path(__file__).with_name('validation_ids.json'))
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/q4'))
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Output directory is not empty; choose a new --out-dir')
    data = load_data(args.train, labeled=True)
    fingerprint = hashlib.sha256(args.train.read_bytes()).hexdigest()
    training, validation = split_data(data, args.split, fingerprint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    metadata = {
        'code_commit': commit, 'working_tree_dirty': dirty, 'seed': SEED,
        'train_sha256': fingerprint,
        'split_sha256': hashlib.sha256(args.split.read_bytes()).hexdigest(),
        'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'train_rows': len(training), 'validation_rows': len(validation), 'native_threads': 1,
        'model': 'lowercase char TF-IDF 2-5, logistic regression', 'selected_C': SELECTED_C,
        'gap_C_values': list(GAP_C_VALUES),
    }
    (args.out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    rows = []
    for c in GAP_C_VALUES:
        print(f'Fitting C={c:g}...', flush=True)
        features, seconds, train_probs, val_probs = fit_and_predict(c, training, validation)
        rows.append({'C': c, 'features': features, 'fit_seconds': seconds,
                     'train': summarize(training['label'], train_probs),
                     'validation': summarize(validation['label'], val_probs)})
        print(json.dumps(rows[-1]), flush=True)
        if c == SELECTED_C:
            # Saved for the subgroup and error analysis; joined to text by ID, never by position.
            validation[['id', 'label']].assign(probability_ai=val_probs).to_csv(
                args.out_dir / 'validation_predictions.csv', index=False)
            training[['id', 'label']].assign(probability_ai=train_probs).to_csv(
                args.out_dir / 'train_predictions.csv', index=False)
    (args.out_dir / 'gap.json').write_text(json.dumps(rows, indent=2) + '\n')

    selected = next(row for row in rows if row['C'] == SELECTED_C)
    report = f"""# Q4: generalization and error analysis

## Protocol

- Code revision: `{commit}` (dirty: {dirty}); seed: `{SEED}`; Python: `{platform.python_version()}`; scikit-learn `{sklearn.__version__}`.
- Selected model: lowercase character TF-IDF (2-5 grams) + logistic regression, C={SELECTED_C:g}, fit on the 8,000 training rows only ({selected['features']:,} features).
- All numbers use the fixed 2,000-row validation set; hidden test labels are not used.

## Q4.1: training vs validation

The selected model scores **{selected['train']['auroc']:.6f}** training AUROC and **{selected['validation']['auroc']:.6f}** validation AUROC (gap {selected['train']['auroc'] - selected['validation']['auroc']:.6f}).

To check whether the gap is regularization-driven, the same features were refit with stronger L2 regularization (smaller C). Only C={SELECTED_C:g} is the selected model; the other rows are diagnostics, not a new selection.

{gap_table(rows)}

Log loss and the confident share describe how extreme the probabilities are; AUROC depends only on ranking.

## Reproduction

```sh
.venv/bin/python errors.py --train data/train.csv
```

Local outputs: `{args.out_dir.as_posix()}/` (gap.json, train/validation predictions; not committed).
"""
    Path(__file__).with_name('q4_results.md').write_text(report)
    print(f'Results: q4_results.md; outputs: {args.out_dir}')


if __name__ == '__main__':
    main()

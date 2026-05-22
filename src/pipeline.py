import os
import sys
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all
from src.data_cleaner import clean_data
from src.ratings import compute_all_ratings
from src.features import (build_tourney_training_data, augment_with_game_flip,
                          FEATURE_COLUMNS)
from src.model import (train_all_models, calibrate_predictions,
                       expanding_window_cv)
from src.generate_submission import generate_submission


def main(data_dir='.', output_dir='submissions', evaluate_only=False):
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print('[1/6] loading + cleaning data')
    ds = clean_data(load_all(data_dir))

    print('[2/6] computing ratings')
    ratings = compute_all_ratings(ds)

    print('[3/6] building features')
    train_df = build_tourney_training_data(ds, ratings)
    train_aug = augment_with_game_flip(train_df)

    print('[4/6] expanding-window CV')
    cv_results, oof_preds, oof_per_model, opt_weights = expanding_window_cv(
        train_aug, FEATURE_COLUMNS, min_train_years=5)

    brier = [r['brier'] for r in cv_results]
    print(f'  cv mean brier {np.mean(brier):.4f}  '
          f'best {min(brier):.4f}  worst {max(brier):.4f}')
    print(f'  vs seed-only ~0.21, coin flip 0.25')

    if evaluate_only:
        return

    print('[5/6] training on full data')
    X = train_aug[FEATURE_COLUMNS].values
    y = train_aug['y'].values
    sw = train_aug['SampleWeight'].values
    models = train_all_models(X, y, sw, feature_names=FEATURE_COLUMNS)

    # OOF weights, not training-set weights (avoids leakage)
    weights = opt_weights
    print('  ensemble weights: ' +
          ', '.join(f'{k} {v:.2f}' for k, v in weights.items()))

    # Calibrate on OOF (training-set calibration overfits)
    valid = ~np.isnan(oof_preds)
    _, calibrators = calibrate_predictions(
        train_aug.loc[valid, 'y'].values,
        oof_preds[valid],
        train_aug.loc[valid, 'gender'].values,
    )

    print('[6/6] generating submission')
    out = os.path.join(output_dir, 'submission_v1.csv')
    sub = generate_submission(models, weights, calibrators, ds, ratings, out)

    elapsed = time.time() - t_start
    print(f'done in {elapsed/60:.1f}min  {len(sub)} rows  cv {np.mean(brier):.4f}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='.')
    p.add_argument('--output-dir', default='submissions')
    p.add_argument('--evaluate-only', action='store_true')
    a = p.parse_args()
    main(a.data_dir, a.output_dir, a.evaluate_only)

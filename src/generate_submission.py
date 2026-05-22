import numpy as np
import pandas as pd
from .data_loader import DataStore
from .features import build_prediction_matchups, FEATURE_COLUMNS
from .model import predict_model, apply_calibration


def generate_submission(models, weights, calibrators, ds, ratings,
                        output_path, stage=2):
    template = ds.submission_stage2 if stage == 2 else ds.submission_stage1
    print(f'building features for {len(template)} matchups')
    pred_df = build_prediction_matchups(template, ds, ratings)

    # LR pipeline imputes internally; trees handle NaN; tabpfn wrapper handles NaN
    X = pred_df[FEATURE_COLUMNS].values

    preds = {}
    for name, m in models.items():
        try:
            preds[name] = predict_model(m, X)
        except Exception as e:
            print(f'  {name} failed: {e}')
            preds[name] = np.full(len(X), 0.5)

    # weighted average + renormalize in case a model dropped out
    ens = np.zeros(len(X))
    total = 0.0
    for name, w in weights.items():
        if name in preds:
            ens += w * preds[name]
            total += w
    if total > 0 and abs(total - 1.0) > 0.001:
        ens /= total

    gender = X[:, FEATURE_COLUMNS.index('gender')]
    out = apply_calibration(ens, gender, calibrators) if calibrators else np.clip(ens, 0.02, 0.98)

    sub = pd.DataFrame({'ID': pred_df['ID'].values, 'Pred': out})

    assert len(sub) == len(template), f'{len(sub)} vs {len(template)}'
    assert set(sub['ID']) == set(template['ID']), 'id mismatch'
    assert sub['Pred'].isna().sum() == 0
    assert sub['Pred'].min() >= 0.005
    assert sub['Pred'].max() <= 0.995

    sub.to_csv(output_path, index=False)
    print(f'wrote {output_path}  n={len(sub)}  '
          f'pred [{sub["Pred"].min():.3f}, {sub["Pred"].max():.3f}]  '
          f'mean {sub["Pred"].mean():.3f}')
    return sub


if __name__ == '__main__':
    print('run pipeline.py')

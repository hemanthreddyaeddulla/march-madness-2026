import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss
from scipy.optimize import minimize
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


DEFAULT_XGB = {
    'max_depth': 3,
    'learning_rate': 0.01,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
}


def train_xgboost(X, y, sample_weight=None, use_optuna=False):
    params = _optuna_tune_xgb(X, y, sample_weight) if use_optuna else DEFAULT_XGB
    model = xgb.XGBClassifier(
        **params, objective='binary:logistic', eval_metric='logloss',
        random_state=42, verbosity=0)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _optuna_tune_xgb(X, y, sample_weight=None, n_trials=50):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print('  optuna missing, defaults')
        return DEFAULT_XGB

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        }
        scores = []
        for tr, va in tscv.split(X):
            m = xgb.XGBClassifier(
                **params, objective='binary:logistic', eval_metric='logloss',
                random_state=42, verbosity=0)
            sw = sample_weight[tr] if sample_weight is not None else None
            m.fit(X[tr], y[tr], sample_weight=sw)
            scores.append(brier_score_loss(y[va], m.predict_proba(X[va])[:, 1]))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f'  optuna best brier {study.best_value:.4f}')
    return study.best_params


def train_logistic(X, y, sample_weight=None):
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)),
    ])
    pipe.fit(X, y, lr__sample_weight=sample_weight)
    return pipe


def train_tabpfn(X, y):
    # tabpfn is a pretrained tabular transformer; no hyperparameters,
    # but it rejects NaN at predict-time and caps at 10k training rows
    try:
        from tabpfn import TabPFNClassifier
        import torch
        Xc = np.nan_to_num(X, nan=-999.0)
        if len(Xc) > 10000:
            idx = np.random.RandomState(42).choice(len(Xc), 10000, replace=False)
            Xc, y = Xc[idx], y[idx]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        m = TabPFNClassifier(device=device, n_estimators=16,
                             ignore_pretraining_limits=True)
        m.fit(Xc, y)
        return m, True
    except Exception as e:
        print(f'  tabpfn failed: {e}')
        return None, False


class TabPFNWrapper:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(np.nan_to_num(X, nan=-999.0))


def train_lightgbm(X, y, sample_weight=None):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        num_leaves=8, learning_rate=0.01, n_estimators=500,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1)
    m.fit(X, y, sample_weight=sample_weight)
    return m


def train_catboost(X, y, sample_weight=None):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(
        depth=3, learning_rate=0.03, iterations=500,
        l2_leaf_reg=5, random_state=42, verbose=0)
    m.fit(X, y, sample_weight=sample_weight)
    return m


class FeatureSubsetModel:
    def __init__(self, model, feature_indices):
        self.model = model
        self.feature_indices = feature_indices

    def predict_proba(self, X):
        Xs = (X[:, self.feature_indices]
              if isinstance(X, np.ndarray)
              else X.iloc[:, self.feature_indices])
        return self.model.predict_proba(Xs)


class SeedBaselineModel:
    # historical P(lower_team wins | seed_diff). anchors extreme matchups
    # where the trees get overconfident (12 vs 5, 1 vs 16 etc).
    def __init__(self):
        self.seed_diff_map = {}
        self.default_prob = 0.5
        self.seed_diff_col_idx = None

    def fit(self, X, y, seed_diff_col_idx=None):
        if seed_diff_col_idx is None:
            return self
        sd = (X[:, seed_diff_col_idx]
              if isinstance(X, np.ndarray)
              else X.iloc[:, seed_diff_col_idx])
        df = pd.DataFrame({'sd': pd.Series(sd).round(0), 'y': y})
        self.seed_diff_map = df.groupby('sd')['y'].mean().to_dict()
        self.default_prob = float(np.mean(y))
        self.seed_diff_col_idx = seed_diff_col_idx
        return self

    def predict_proba(self, X):
        sd = (X[:, self.seed_diff_col_idx]
              if isinstance(X, np.ndarray)
              else X.iloc[:, self.seed_diff_col_idx])
        p = np.array([self.seed_diff_map.get(round(s), self.default_prob) for s in sd])
        return np.column_stack([1 - p, p])


# women get wider bounds - fewer historic upsets so 0.99 is honest
CLIP_BOUNDS = {0: (0.02, 0.98), 1: (0.01, 0.99)}


def train_all_models(X_train, y_train, sample_weight=None, feature_names=None):
    from .features import FEATURE_COLUMNS, RATING_FEATURES, PERFORMANCE_FEATURES
    if feature_names is None:
        feature_names = FEATURE_COLUMNS

    rating_idx = [i for i, f in enumerate(feature_names) if f in RATING_FEATURES]
    perf_idx = [i for i, f in enumerate(feature_names) if f in PERFORMANCE_FEATURES]
    seed_diff_idx = (feature_names.index('seed_diff')
                     if 'seed_diff' in feature_names else None)

    models = {}

    print('  xgb (all features, optuna)...')
    models['xgb'] = train_xgboost(X_train, y_train, sample_weight, use_optuna=True)

    # lgb and cb get different feature subsets - otherwise they just echo xgb
    # and the weight optimizer zeros them out
    print('  lgb (ratings only)...')
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        num_leaves=16, learning_rate=0.01, n_estimators=500,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.9,
        random_state=43, verbose=-1)
    m.fit(X_train[:, rating_idx], y_train, sample_weight=sample_weight)
    models['lgb_ratings'] = FeatureSubsetModel(m, rating_idx)

    print('  cb (performance only)...')
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(
        depth=4, learning_rate=0.03, iterations=500,
        l2_leaf_reg=3, random_state=44, verbose=0)
    m.fit(X_train[:, perf_idx], y_train, sample_weight=sample_weight)
    models['cb_perf'] = FeatureSubsetModel(m, perf_idx)

    print('  lr...')
    models['lr'] = train_logistic(X_train, y_train, sample_weight)

    print('  tabpfn...')
    tabpfn, ok = train_tabpfn(X_train, y_train)
    if ok:
        models['tabpfn'] = TabPFNWrapper(tabpfn)
    else:
        print('  tabpfn: skipped')

    if seed_diff_idx is not None:
        print('  seed baseline...')
        models['seed_baseline'] = SeedBaselineModel().fit(
            X_train, y_train, seed_diff_col_idx=seed_diff_idx)

    return models


def predict_model(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def optimize_ensemble_weights(predictions_dict, y_true, gender_mask=None):
    # nelder-mead on |w| / sum|w| keeps the weights on the simplex
    names = list(predictions_dict.keys())
    P = np.column_stack([predictions_dict[n] for n in names])

    def obj(w):
        w = np.abs(w)
        w = w / w.sum()
        return np.mean((P @ w - y_true) ** 2)

    x0 = np.ones(len(names)) / len(names)
    res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 10000})
    w = np.abs(res.x); w = w / w.sum()
    return {n: wi for n, wi in zip(names, w)}, res.fun


def ensemble_predict(models, X, weights=None):
    preds = {name: predict_model(m, X) for name, m in models.items()}
    if weights is None:
        weights = {name: 1.0 / len(models) for name in models}
    out = np.zeros(len(X))
    for name, w in weights.items():
        if name in preds:
            out += w * preds[name]
    return out


def calibrate_predictions(y_true, y_pred, gender_flags):
    # isotonic per gender - women's bracket has way less variance
    from sklearn.isotonic import IsotonicRegression
    calibrators = {}
    out = y_pred.copy()
    for g in (0, 1):
        mask = gender_flags == g
        if mask.sum() < 20:
            continue
        lo, hi = CLIP_BOUNDS[g]
        cal = IsotonicRegression(y_min=lo, y_max=hi, out_of_bounds='clip')
        cal.fit(y_pred[mask], y_true[mask])
        out[mask] = cal.predict(y_pred[mask])
        calibrators[g] = cal
    return out, calibrators


def apply_calibration(y_pred, gender_flags, calibrators):
    out = y_pred.copy()
    for g, cal in calibrators.items():
        mask = gender_flags == g
        if mask.sum() > 0:
            out[mask] = cal.predict(y_pred[mask])
    for g, (lo, hi) in CLIP_BOUNDS.items():
        mask = gender_flags == g
        out[mask] = np.clip(out[mask], lo, hi)
    return out


def expanding_window_cv(train_df, feature_cols, min_train_years=5):
    # train on [start..N], test on N+1
    seasons = sorted(train_df['Season'].unique())
    results = []
    oof = np.full(len(train_df), np.nan)
    oof_per_model = {}

    for i in range(min_train_years, len(seasons)):
        test_season = seasons[i]
        tr_mask = train_df['Season'].isin(seasons[:i])
        te_mask = train_df['Season'] == test_season
        if te_mask.sum() == 0:
            continue

        X_tr = train_df.loc[tr_mask, feature_cols].values
        y_tr = train_df.loc[tr_mask, 'y'].values
        sw_tr = train_df.loc[tr_mask, 'SampleWeight'].values
        X_te = train_df.loc[te_mask, feature_cols].values
        y_te = train_df.loc[te_mask, 'y'].values
        g_te = train_df.loc[te_mask, 'gender'].values
        te_idx = train_df.index[te_mask]

        models = train_all_models(X_tr, y_tr, sw_tr, feature_names=feature_cols)

        for name, m in models.items():
            p = predict_model(m, X_te)
            if name not in oof_per_model:
                oof_per_model[name] = np.full(len(train_df), np.nan)
            oof_per_model[name][te_idx] = p

        # equal weights inside CV -> honest brier. weights come later from OOF.
        weights = {name: 1.0 / len(models) for name in models}
        ens = ensemble_predict(models, X_te, weights)

        b = brier_score_loss(y_te, ens)
        bm = (brier_score_loss(y_te[g_te == 0], ens[g_te == 0])
              if (g_te == 0).sum() > 0 else np.nan)
        bw = (brier_score_loss(y_te[g_te == 1], ens[g_te == 1])
              if (g_te == 1).sum() > 0 else np.nan)
        results.append({
            'test_season': test_season,
            'brier': b, 'brier_men': bm, 'brier_women': bw,
            'n_games': len(y_te), 'weights': weights,
        })
        oof[te_idx] = ens
        print(f'  {test_season}: brier {b:.4f}  m {bm:.4f}  w {bw:.4f}  n {len(y_te)}')

    valid = ~np.isnan(oof)
    print(f"\n  oof brier (equal weights): "
          f"{brier_score_loss(train_df.loc[valid, 'y'], oof[valid]):.4f}")

    # optimize weights on OOF preds (no leakage: model never saw these labels)
    oof_valid = {n: arr[valid] for n, arr in oof_per_model.items()
                 if not np.all(np.isnan(arr[valid]))}
    y_valid = train_df.loc[valid, 'y'].values
    if oof_valid:
        opt_weights, opt_brier = optimize_ensemble_weights(oof_valid, y_valid)
        print('  oof-opt: ' + ', '.join(f'{k} {v:.3f}' for k, v in opt_weights.items()))
        print(f'  oof-opt brier: {opt_brier:.4f}')
    else:
        opt_weights = weights

    return results, oof, oof_per_model, opt_weights


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, '.')
    from src.data_loader import load_all
    from src.data_cleaner import clean_data
    from src.ratings import compute_all_ratings
    from src.features import (build_tourney_training_data, augment_with_game_flip,
                              FEATURE_COLUMNS)
    ds = clean_data(load_all('.'))
    ratings = compute_all_ratings(ds)
    aug = augment_with_game_flip(build_tourney_training_data(ds, ratings))
    expanding_window_cv(aug, FEATURE_COLUMNS, min_train_years=5)

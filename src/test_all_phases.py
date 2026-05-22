import sys
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = 0
ERRORS = []


def t(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        msg = f'fail: {name} -- {detail}'
        print(msg)
        ERRORS.append(msg)


# loader
from src.data_loader import load_all, parse_seed, parse_submission_id, get_gender

t("seed W01", parse_seed('W01') == 1)
t("seed X16a", parse_seed('X16a') == 16)
t("seed Z11b", parse_seed('Z11b') == 11)
t("subid", parse_submission_id('2026_1104_1181') == (2026, 1104, 1181))
t("gender m", get_gender(1181) == 'M')
t("gender w", get_gender(3181) == 'W')

ds = load_all('.')

t("m teams", len(ds.m_teams) > 300)
t("w teams", len(ds.w_teams) > 300)
t("m reg compact", len(ds.m_reg_compact) > 100_000)
t("w reg compact", len(ds.w_reg_compact) > 100_000)
t("m reg detailed", len(ds.m_reg_detailed) > 100_000)
t("w reg detailed", len(ds.w_reg_detailed) > 80_000)
t("m tourney", len(ds.m_tourney_compact) > 2000)
t("w tourney", len(ds.w_tourney_compact) > 1000)
t("m seeds", len(ds.m_seeds) > 2000)
t("w seeds", len(ds.w_seeds) > 1000)
t("massey", len(ds.m_massey) > 5_000_000)
t("stage2", len(ds.submission_stage2) >= 132_132)
t("m conf tourney", len(ds.m_conf_tourney_games) > 5000)

t("seednum m", 'SeedNum' in ds.m_seeds.columns)
t("seednum w", 'SeedNum' in ds.w_seeds.columns)
t("seed range m", ds.m_seeds['SeedNum'].min() == 1 and ds.m_seeds['SeedNum'].max() == 16)
t("seed range w", ds.w_seeds['SeedNum'].min() == 1 and ds.w_seeds['SeedNum'].max() == 16)
t("2026 m seeds 68", len(ds.m_seeds[ds.m_seeds['Season'] == 2026]) == 68)
t("2026 w seeds 68", len(ds.w_seeds[ds.w_seeds['Season'] == 2026]) == 68)

# chronological sort matters for Elo
sorted_check = ds.m_reg_compact[['Season', 'DayNum']].equals(
    ds.m_reg_compact[['Season', 'DayNum']].sort_values(['Season', 'DayNum']).reset_index(drop=True))
t("m compact sorted", sorted_check)
t("no dup m games",
  ds.m_reg_compact.duplicated(subset=['Season', 'DayNum', 'WTeamID', 'LTeamID']).sum() == 0)

t("m team id range", ds.m_teams['TeamID'].between(1000, 1999).all())
t("w team id range", ds.w_teams['TeamID'].between(3000, 3999).all())


# cleaner
from src.data_cleaner import clean_data, compute_team_season_stats, compute_conf_tourney_stats
ds = clean_data(ds)

t("poss m", 'Possessions' in ds.m_reg_detailed.columns)
t("poss w", 'Possessions' in ds.w_reg_detailed.columns)

poss = ds.m_reg_detailed[ds.m_reg_detailed['Season'] == 2026]['Possessions']
t("poss mean 55-85", 55 < poss.mean() < 85, f'mean {poss.mean():.1f}')
t("poss > 0", (poss <= 0).sum() == 0)

for col in ['WOffEff', 'WeFGPct', 'WTOVPct', 'WORBPct', 'WFTRate', 'WFourFactors']:
    t(f"{col} present", col in ds.m_reg_detailed.columns)

efg = ds.m_reg_detailed['WeFGPct']
t("efg in [0,1]", efg.min() >= 0 and efg.max() <= 1.0)

t("no 2020 tourney", len(ds.m_tourney_compact[ds.m_tourney_compact['Season'] == 2020]) == 0)
t("2021 weight 0.7",
  (ds.m_tourney_compact[ds.m_tourney_compact['Season'] == 2021]['SampleWeight'] == 0.7).all())

# raw point sanity (pre-OT-normalization)
raw = pd.read_csv('MRegularSeasonDetailedResults.csv')
t("score = 2(FGM-3PM) + 3(3PM) + FTM",
  (2 * (raw['WFGM'] - raw['WFGM3']) + 3 * raw['WFGM3'] + raw['WFTM'] == raw['WScore']).all())

stats = compute_team_season_stats(ds.m_reg_detailed, ds.m_reg_compact, 2026)
t("team stats 2026", len(stats) > 350)
t("winpct [0,1]", stats['WinPct'].between(0, 1).all())
t("pyth col", 'PythWinPct' in stats.columns)
t("luck col", 'Luck' in stats.columns)
t("pyth [0,1]", stats['PythWinPct'].between(0, 1).all())

ct = compute_conf_tourney_stats(ds.m_conf_tourney_games, ds.m_reg_compact, 2026)
t("conf tourney stats", len(ct) > 100)
t("champ binary", ct['ConfTourneyChamp'].isin([0, 1]).all())


# ratings
from src.ratings import (compute_elo, compute_glicko2, compute_trueskill,
                         compute_bradley_terry, compute_massey_features, compute_all_ratings)

elo_m = compute_elo(ds.m_reg_compact, ds.m_team_conferences, 'M')
t("elo m rows", len(elo_m) > 10_000)
t("elo cols", all(c in elo_m.columns for c in ['Season', 'TeamID', 'Elo']))

elo_2026 = elo_m[elo_m['Season'] == 2026]
t("elo 2026", len(elo_2026) > 350)
t("elo range", 700 < elo_2026['Elo'].min() and elo_2026['Elo'].max() < 2500)
t("elo vol >= 0", (elo_2026['EloVolatility'] >= 0).all())
t("duke top 20", 1181 in elo_2026.nlargest(20, 'Elo')['TeamID'].values)

glicko_m = compute_glicko2(ds.m_reg_compact, 'M')
t("glicko rows", len(glicko_m) > 10_000)
t("glickomu", 'GlickoMu' in glicko_m.columns)
gl_2026 = glicko_m[glicko_m['Season'] == 2026]
t("glicko range", gl_2026['GlickoMu'].min() > 500 and gl_2026['GlickoMu'].max() < 3000)

ts_m = compute_trueskill(ds.m_reg_compact, 'M')
t("ts rows", len(ts_m) > 10_000)
ts_2026 = ts_m[ts_m['Season'] == 2026]
t("ts range", ts_2026['TSMu'].min() > 10 and ts_2026['TSMu'].max() < 50)

bt_m = compute_bradley_terry(ds.m_reg_compact, 'M')
t("bt rows", len(bt_m) > 10_000)
bt_2026 = bt_m[bt_m['Season'] == 2026]
t("bt has spread", bt_2026['BTStrength'].std() > 0.1)

massey = compute_massey_features(ds.m_massey)
t("massey rows", len(massey) > 5000)
t("massey_pom", 'Massey_POM' in massey.columns)
t("massey_mor", 'Massey_MOR' in massey.columns)
t("massey 2026", len(massey[massey['Season'] == 2026]) > 350)

# the whole point of multiple rating systems is they should disagree
# at the margins -- check they're correlated but not collinear
merged = elo_2026[['TeamID', 'Elo']].merge(
    gl_2026[['TeamID', 'GlickoMu']], on='TeamID').merge(
    ts_2026[['TeamID', 'TSMu']], on='TeamID').merge(
    bt_2026[['TeamID', 'BTStrength']], on='TeamID')
corr = merged[['Elo', 'GlickoMu', 'TSMu', 'BTStrength']].corr()
t("elo-ts corr < 0.99", corr.loc['Elo', 'TSMu'] < 0.99, f"r={corr.loc['Elo', 'TSMu']:.3f}")
t("all corr > 0.7", corr.min().min() > 0.7)

elo_w = compute_elo(ds.w_reg_compact, ds.w_team_conferences, 'W')
t("w elo", len(elo_w) > 5000)
ts_w = compute_trueskill(ds.w_reg_compact, 'W')
t("w ts", len(ts_w) > 5000)


# features
from src.features import (FEATURE_COLUMNS, RATING_FEATURES, PERFORMANCE_FEATURES,
                          build_tourney_training_data, augment_with_game_flip)

t("feature cols", len(FEATURE_COLUMNS) > 30)
t("no dup features", len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)))
t("rating subset", all(f in FEATURE_COLUMNS for f in RATING_FEATURES))
t("perf subset", all(f in FEATURE_COLUMNS for f in PERFORMANCE_FEATURES))
t("rating + perf = all", len(RATING_FEATURES) + len(PERFORMANCE_FEATURES) == len(FEATURE_COLUMNS))

ratings = compute_all_ratings(ds)
train_df = build_tourney_training_data(ds, ratings)
t("train rows", len(train_df) > 2000)
t("has y", 'y' in train_df.columns)
t("y binary", train_df['y'].isin([0, 1]).all())
t("no 2020", 2020 not in train_df['Season'].values)
t("has m and w", set(train_df['Gender']) == {'M', 'W'})
t("all cols present", all(c in train_df.columns for c in FEATURE_COLUMNS))

# sanity: elo_diff should correlate positively with y (better team wins more often)
elo_corr = train_df['elo_diff'].corr(train_df['y'])
t("elo_diff vs y > 0", elo_corr > 0, f"r={elo_corr:.3f}")
seed_corr = train_df['seed_diff'].corr(train_df['y'])
t("seed_diff vs y nonzero", abs(seed_corr) > 0.05, f"r={seed_corr:.3f}")

train_aug = augment_with_game_flip(train_df)
t("flip doubles rows", len(train_aug) == 2 * len(train_df))
t("y balanced", abs(train_aug['y'].mean() - 0.5) < 0.01)
t("flipgroup col", 'FlipGroup' in train_aug.columns)

# verify mirror symmetry: original row + flipped row should sum to invariants
orig = train_aug.iloc[0]
flip = train_aug.iloc[len(train_df)]
t("flip negates elo_diff", abs(orig['elo_diff'] + flip['elo_diff']) < 0.01)
t("flip inverts y", orig['y'] + flip['y'] == 1)
t("flip preserves seed_product", orig['seed_product'] == flip['seed_product'])
t("flip preserves gender", orig['gender'] == flip['gender'])


# models
from src.model import (train_xgboost, train_logistic, train_all_models,
                       predict_model, optimize_ensemble_weights,
                       ensemble_predict, calibrate_predictions, apply_calibration,
                       SeedBaselineModel, expanding_window_cv)

X = train_aug[FEATURE_COLUMNS].values
y = train_aug['y'].values
sw = train_aug['SampleWeight'].values
gender = train_aug['gender'].values

xgb_model = train_xgboost(X, y, sw, use_optuna=False)
xp = predict_model(xgb_model, X)
t("xgb preds shape", len(xp) == len(X))
t("xgb preds [0,1]", xp.min() >= 0 and xp.max() <= 1)

lr_model = train_logistic(X, y, sw)
lp = predict_model(lr_model, X)
t("lr preds [0,1]", lp.min() >= 0 and lp.max() <= 1)

sb = SeedBaselineModel().fit(X, y, seed_diff_col_idx=FEATURE_COLUMNS.index('seed_diff'))
sp = predict_model(sb, X)
t("seed preds [0,1]", sp.min() >= 0 and sp.max() <= 1)

models = train_all_models(X, y, sw, feature_names=FEATURE_COLUMNS)
t("models trained", len(models) >= 4)
for name, m in models.items():
    p = predict_model(m, X)
    ok = (len(p) == len(X) and p.min() >= 0 and p.max() <= 1
          and not np.any(np.isnan(p)))
    t(f"{name} preds valid", ok)

preds = {n: predict_model(m, X) for n, m in models.items()}
weights, brier = optimize_ensemble_weights(preds, y)
t("weights sum 1", abs(sum(weights.values()) - 1.0) < 0.01)
t("weights >= 0", all(w >= 0 for w in weights.values()))
t("opt brier < 0.25", brier < 0.25)

ens = ensemble_predict(models, X, weights)
t("ens shape", len(ens) == len(X))
t("ens [0,1]", ens.min() >= 0 and ens.max() <= 1)

cal, calibrators = calibrate_predictions(y, ens, gender)
t("cal shape", len(cal) == len(X))
t("cal both genders", 0 in calibrators and 1 in calibrators)

new = apply_calibration(ens, gender, calibrators)
t("cal m bounded", new[gender == 0].min() >= 0.02)
t("cal w bounded", new[gender == 1].min() >= 0.01)


# quick CV (3 folds only)
cv_results, oof_preds, oof_per_model, opt_weights = expanding_window_cv(
    train_aug, FEATURE_COLUMNS, min_train_years=14)
t("cv folds > 0", len(cv_results) > 0)
t("oof shape", len(oof_preds) == len(train_aug))
t("per-model oof", len(oof_per_model) > 0)
t("opt weights sum 1", abs(sum(opt_weights.values()) - 1.0) < 0.01)
b_scores = [r['brier'] for r in cv_results]
t("brier < 0.25", all(b < 0.25 for b in b_scores))
t("brier > 0.05", all(b > 0.05 for b in b_scores))


# submission file sanity
sub_path = 'submissions/submission_v1.csv'
if os.path.exists(sub_path):
    sub = pd.read_csv(sub_path)
    template = ds.submission_stage2
    t("sub rows == template", len(sub) == len(template))
    t("sub has ID", 'ID' in sub.columns)
    t("sub has Pred", 'Pred' in sub.columns)
    t("sub no nan", sub['Pred'].isna().sum() == 0)
    t("pred >= 0.005", sub['Pred'].min() >= 0.005)
    t("pred <= 0.995", sub['Pred'].max() <= 0.995)
    t("pred mean ~0.5", 0.45 < sub['Pred'].mean() < 0.55)
    t("ids match", set(sub['ID']) == set(template['ID']))
    t("has confident lows", (sub['Pred'] < 0.1).sum() > 1000)
    t("has confident highs", (sub['Pred'] > 0.9).sum() > 1000)
    t("has uncertain", ((sub['Pred'] > 0.4) & (sub['Pred'] < 0.6)).sum() > 1000)

    sub['TeamLow'] = sub['ID'].str.split('_').str[1].astype(int)
    mp = sub[sub['TeamLow'] < 3000]['Pred']
    wp = sub[sub['TeamLow'] >= 3000]['Pred']
    t("men preds", len(mp) > 50_000)
    t("women preds", len(wp) > 50_000)
    # women should be at least as confident as men (chalkier tournament)
    t("w std >= m std * 0.9", wp.std() >= mp.std() * 0.9)


# box-score sanity (catches obvious data corruption)
det = ds.m_reg_detailed
t("FGM <= FGA", (det['WFGM'] > det['WFGA']).sum() == 0)
t("FGM3 <= FGA3", (det['WFGM3'] > det['WFGA3']).sum() == 0)
t("FGM3 <= FGM", (det['WFGM3'] > det['WFGM']).sum() == 0)
t("FTM <= FTA", (det['WFTM'] > det['WFTA']).sum() == 0)
t("score >= 0", (det['WScore'] < 0).sum() == 0)
t("winner > loser", (det['WScore'] > det['LScore']).all())

t("2026 conf rows", len(ds.m_team_conferences[ds.m_team_conferences['Season'] == 2026]) > 350)

m_2026 = ds.m_massey[ds.m_massey['Season'] == 2026]
t("POM 2026", len(m_2026[m_2026['SystemName'] == 'POM']) > 350)
t("MOR 2026", len(m_2026[m_2026['SystemName'] == 'MOR']) > 350)


print(f'\n{PASS} passed, {FAIL} failed (of {PASS + FAIL})')
if ERRORS:
    sys.exit(1)

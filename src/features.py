import numpy as np
import pandas as pd
from .data_loader import DataStore, parse_seed, get_gender
from .data_cleaner import compute_team_season_stats, compute_conf_tourney_stats, clean_data


# all features are (lower_team_id) - (higher_team_id)
FEATURE_COLUMNS = [
    # ratings
    'elo_diff', 'elo_volatility_diff',
    'glicko_mu_diff', 'glicko_phi_diff',
    'trueskill_mu_diff', 'trueskill_sigma_diff',
    'bt_strength_diff',
    'massey_pom_diff', 'massey_mor_diff', 'massey_avg_diff', 'massey_best_diff',
    'meta_rating_diff',
    # seeds
    'seed_diff', 'seed_product', 'seed_sum',
    # efficiency
    'adj_off_eff_diff', 'adj_def_eff_diff', 'net_eff_diff',
    'efg_pct_diff', 'tov_pct_diff', 'orb_pct_diff', 'ft_rate_diff',
    'four_factors_diff',
    # performance
    'win_pct_diff', 'point_diff_avg_diff', 'sos_elo_diff',
    'last14_win_pct_diff', 'last14_elo_trend_diff',
    'away_neutral_win_pct_diff', 'quality_win_pct_diff', 'close_game_win_pct_diff',
    'seed_x_elo',
    'pyth_win_pct_diff', 'luck_diff',
    'conf_tourney_wins_diff', 'conf_tourney_champ_diff',
    'gender', 'conf_power_diff', 'coach_tourney_exp_diff',
]

# split for the specialist tree models (LGB and CB train on different subsets)
RATING_FEATURES = [c for c in FEATURE_COLUMNS if any(
    c.startswith(p) for p in ['elo_', 'glicko_', 'trueskill_', 'bt_', 'massey_', 'meta_'])]
PERFORMANCE_FEATURES = [c for c in FEATURE_COLUMNS if c not in RATING_FEATURES]


def _build_team_feature_table(season, gender, ds, ratings):
    prefix = 'm' if gender == 'M' else 'w'
    reg_detailed = getattr(ds, f'{prefix}_reg_detailed')
    reg_compact = getattr(ds, f'{prefix}_reg_compact')
    seeds_df = getattr(ds, f'{prefix}_seeds')
    conferences_df = getattr(ds, f'{prefix}_team_conferences')

    team_stats = compute_team_season_stats(reg_detailed, reg_compact, season, min_games=1)
    if len(team_stats) == 0:
        # fall back to compact when detailed isn't available (pre-2003 men, pre-2010 women)
        comp = reg_compact[(reg_compact['Season'] == season) & (reg_compact['DayNum'] <= 132)]
        rows = []
        for tid in set(comp['WTeamID']) | set(comp['LTeamID']):
            wins = comp[comp['WTeamID'] == tid]
            losses = comp[comp['LTeamID'] == tid]
            n = len(wins) + len(losses)
            if n == 0:
                continue
            rows.append({
                'TeamID': tid,
                'WinPct': len(wins) / n,
                'PointDiffAvg': ((wins['WScore'].sum() + losses['LScore'].sum()) -
                                 (wins['LScore'].sum() + losses['WScore'].sum())) / n,
                'Games': n,
            })
        team_stats = pd.DataFrame(rows).set_index('TeamID')

    rating_joins = [
        (f'elo_{prefix}', ['Elo', 'EloVolatility']),
        (f'glicko_{prefix}', ['GlickoMu', 'GlickoPhi']),
        (f'trueskill_{prefix}', ['TSMu', 'TSSigma']),
        (f'bt_{prefix}', ['BTStrength']),
    ]
    for key, cols in rating_joins:
        if key in ratings:
            rdf = ratings[key]
            team_stats = team_stats.join(
                rdf[rdf['Season'] == season].set_index('TeamID')[cols], how='left')

    if gender == 'M' and 'massey' in ratings:
        m = ratings['massey']
        ms = m[m['Season'] == season].set_index('TeamID')
        mcols = [c for c in ms.columns if c.startswith('Massey')]
        if mcols:
            team_stats = team_stats.join(ms[mcols], how='left')

    season_seeds = seeds_df[seeds_df['Season'] == season].set_index('TeamID')
    if 'SeedNum' in season_seeds.columns:
        team_stats = team_stats.join(season_seeds[['SeedNum']], how='left')
    team_stats['SeedNum'] = team_stats.get('SeedNum', pd.Series(dtype=float)).fillna(16)

    # conference power = average Elo of teams in the conference
    if len(conferences_df) > 0 and 'Elo' in team_stats.columns:
        conf_season = conferences_df[conferences_df['Season'] == season]
        team_conf = conf_season.set_index('TeamID')['ConfAbbrev']
        team_stats = team_stats.join(team_conf.rename('ConfAbbrev'), how='left')
        conf_power = team_stats.groupby('ConfAbbrev')['Elo'].mean()
        team_stats['ConfPower'] = team_stats['ConfAbbrev'].map(conf_power)
        team_stats.drop(columns=['ConfAbbrev'], inplace=True, errors='ignore')

    # coach tournament experience - only count seasons strictly before current season
    if gender == 'M' and hasattr(ds, 'm_coaches') and len(ds.m_coaches) > 0:
        coaches = ds.m_coaches
        current = coaches[(coaches['Season'] == season) & (coaches['LastDayNum'] >= 132)]
        if len(current) > 0:
            prior_tourneys = [s for s in ds.m_tourney_compact['Season'].unique() if s < season]
            coach_exp = coaches[coaches['Season'].isin(prior_tourneys)].groupby(
                'CoachName')['Season'].nunique().rename('CoachTourneyExp')
            current = current.set_index('TeamID').join(coach_exp, on='CoachName', how='left')
            team_stats = team_stats.join(current[['CoachTourneyExp']], how='left')

    conf_tourney_df = getattr(ds, f'{prefix}_conf_tourney_games', pd.DataFrame())
    if len(conf_tourney_df) > 0:
        ct = compute_conf_tourney_stats(conf_tourney_df, reg_compact, season)
        if len(ct) > 0:
            team_stats = team_stats.join(ct, how='left')
    for col in ['ConfTourneyWins', 'ConfTourneyGames', 'ConfTourneyChamp']:
        if col not in team_stats.columns:
            team_stats[col] = 0
        team_stats[col] = team_stats[col].fillna(0)

    # SOS = mean Elo of opponents played
    if 'Elo' in team_stats.columns:
        comp = reg_compact[(reg_compact['Season'] == season) & (reg_compact['DayNum'] <= 132)]
        elo_lookup = team_stats['Elo'].to_dict()
        sos = {}
        for tid in team_stats.index:
            opps = np.concatenate([
                comp[comp['WTeamID'] == tid]['LTeamID'].values,
                comp[comp['LTeamID'] == tid]['WTeamID'].values,
            ])
            sos[tid] = (np.mean([elo_lookup.get(o, 1500) for o in opps])
                        if len(opps) > 0 else 1500)
        team_stats['SOS_Elo'] = pd.Series(sos)

    # quality wins = win rate vs top-50 Elo opponents
    if 'Elo' in team_stats.columns:
        comp = reg_compact[(reg_compact['Season'] == season) & (reg_compact['DayNum'] <= 132)]
        top50 = team_stats.nlargest(50, 'Elo').index
        q = {}
        for tid in team_stats.index:
            w = len(comp[(comp['WTeamID'] == tid) & (comp['LTeamID'].isin(top50))])
            l = len(comp[(comp['LTeamID'] == tid) & (comp['WTeamID'].isin(top50))])
            total = w + l
            q[tid] = w / total if total > 0 else 0.0
        team_stats['QualityWinPct'] = pd.Series(q)

    team_stats['CoachTourneyExp'] = team_stats.get(
        'CoachTourneyExp', pd.Series(0, index=team_stats.index)).fillna(0)
    team_stats['ConfPower'] = team_stats.get(
        'ConfPower', pd.Series(1500, index=team_stats.index)).fillna(1500)

    # meta-rating: z-scored mean of all available rating systems
    z_cols = []
    for col in ['Elo', 'GlickoMu', 'TSMu', 'BTStrength']:
        if col in team_stats.columns:
            mu, sd = team_stats[col].mean(), team_stats[col].std()
            if sd > 0:
                team_stats[f'{col}_z'] = (team_stats[col] - mu) / sd
                z_cols.append(f'{col}_z')
    if z_cols:
        team_stats['MetaRating'] = team_stats[z_cols].mean(axis=1)
        team_stats.drop(columns=z_cols, inplace=True)
    else:
        team_stats['MetaRating'] = 0.0

    return team_stats


def _compute_matchup_features(low_stats, high_stats, gender):
    def diff(col):
        a = low_stats.get(col, np.nan)
        b = high_stats.get(col, np.nan)
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return a - b

    f = {}

    f['elo_diff'] = diff('Elo')
    f['elo_volatility_diff'] = diff('EloVolatility')
    f['glicko_mu_diff'] = diff('GlickoMu')
    f['glicko_phi_diff'] = diff('GlickoPhi')
    f['trueskill_mu_diff'] = diff('TSMu')
    f['trueskill_sigma_diff'] = diff('TSSigma')
    f['bt_strength_diff'] = diff('BTStrength')
    f['massey_pom_diff'] = diff('Massey_POM') if gender == 'M' else 0.0
    f['massey_mor_diff'] = diff('Massey_MOR') if gender == 'M' else 0.0
    f['massey_avg_diff'] = diff('MasseyAvg') if gender == 'M' else 0.0
    f['massey_best_diff'] = diff('MasseyBest') if gender == 'M' else 0.0
    f['meta_rating_diff'] = diff('MetaRating')

    s_low = low_stats.get('SeedNum', 16)
    s_high = high_stats.get('SeedNum', 16)
    f['seed_diff'] = s_low - s_high
    f['seed_product'] = s_low * s_high
    f['seed_sum'] = s_low + s_high

    f['adj_off_eff_diff'] = diff('OffEff')
    # flip sign so "positive = team_low better defense"
    f['adj_def_eff_diff'] = -diff('DefEff')
    f['net_eff_diff'] = diff('NetEff')
    f['efg_pct_diff'] = diff('eFGPct')
    # flip sign so "positive = team_low takes care of the ball"
    f['tov_pct_diff'] = -diff('TOVPct')
    f['orb_pct_diff'] = diff('ORBPct')
    f['ft_rate_diff'] = diff('FTRate')
    f['four_factors_diff'] = diff('FourFactors')

    f['win_pct_diff'] = diff('WinPct')
    f['point_diff_avg_diff'] = diff('PointDiffAvg')
    f['sos_elo_diff'] = diff('SOS_Elo')
    f['last14_win_pct_diff'] = diff('Last14WinPct')
    f['last14_elo_trend_diff'] = diff('EloVolatility')
    f['away_neutral_win_pct_diff'] = diff('AwayNeutralWinPct')
    f['quality_win_pct_diff'] = diff('QualityWinPct')
    f['close_game_win_pct_diff'] = diff('CloseGameWinPct')

    f['conf_tourney_wins_diff'] = diff('ConfTourneyWins')
    f['conf_tourney_champ_diff'] = diff('ConfTourneyChamp')

    sd, ed = f['seed_diff'], f['elo_diff']
    f['seed_x_elo'] = sd * ed if not (pd.isna(sd) or pd.isna(ed)) else np.nan

    f['pyth_win_pct_diff'] = diff('PythWinPct')
    f['luck_diff'] = diff('Luck')

    f['gender'] = 0 if gender == 'M' else 1
    f['conf_power_diff'] = diff('ConfPower')
    f['coach_tourney_exp_diff'] = diff('CoachTourneyExp')

    return f


def build_tourney_training_data(ds, ratings):
    rows = []

    for gender in ['M', 'W']:
        prefix = 'm' if gender == 'M' else 'w'
        tourney_compact = getattr(ds, f'{prefix}_tourney_compact')
        if len(tourney_compact) == 0:
            continue

        for season in sorted(tourney_compact['Season'].unique()):
            if season == 2020:
                continue  # COVID, no tournament
            if season < 2003 and gender == 'M':
                continue  # detailed box scores start in 2003
            if season < 2010 and gender == 'W':
                continue

            team_table = _build_team_feature_table(season, gender, ds, ratings)
            if len(team_table) == 0:
                continue

            for _, game in tourney_compact[tourney_compact['Season'] == season].iterrows():
                w_id, l_id = game['WTeamID'], game['LTeamID']
                lo, hi = min(w_id, l_id), max(w_id, l_id)
                if lo not in team_table.index or hi not in team_table.index:
                    continue

                f = _compute_matchup_features(
                    team_table.loc[lo], team_table.loc[hi], gender)
                f['y'] = 1 if w_id == lo else 0
                f['Season'] = season
                f['TeamID_Low'] = lo
                f['TeamID_High'] = hi
                f['Gender'] = gender
                f['SampleWeight'] = game.get('SampleWeight', 1.0)
                rows.append(f)

    df = pd.DataFrame(rows)
    print(f"train: {len(df)} games "
          f"({(df['Gender']=='M').sum()} m, {(df['Gender']=='W').sum()} w)")
    return df


def build_prediction_matchups(submission_df, ds, ratings):
    parsed = submission_df['ID'].str.split('_', expand=True)
    parsed.columns = ['Season', 'TeamLow', 'TeamHigh']
    parsed = parsed.astype(int)

    out = []
    for season in parsed['Season'].unique():
        s_matchups = parsed[parsed['Season'] == season]

        for gender in ['M', 'W']:
            if gender == 'M':
                gmask = s_matchups['TeamLow'] < 3000
            else:
                gmask = s_matchups['TeamLow'] >= 3000
            g_matchups = s_matchups[gmask]
            if len(g_matchups) == 0:
                continue

            team_table = _build_team_feature_table(season, gender, ds, ratings)

            for idx, row in g_matchups.iterrows():
                lo, hi = row['TeamLow'], row['TeamHigh']
                if lo in team_table.index and hi in team_table.index:
                    f = _compute_matchup_features(
                        team_table.loc[lo], team_table.loc[hi], gender)
                else:
                    f = {c: np.nan for c in FEATURE_COLUMNS}
                    f['gender'] = 0 if gender == 'M' else 1
                f['ID'] = submission_df.loc[idx, 'ID']
                out.append(f)

    result = pd.DataFrame(out)
    print(f"predict: {len(result)} matchups")
    return result


def augment_with_game_flip(train_df):
    # for each row (A vs B, y), add the mirror (B vs A, 1-y)
    # caveat: products of two negated quantities don't flip, skip those
    flipped = train_df.copy()
    no_flip = {'gender', 'seed_product', 'seed_sum', 'seed_x_elo'}
    for col in FEATURE_COLUMNS:
        if col in no_flip:
            continue
        if col in flipped.columns:
            flipped[col] = -flipped[col]
    flipped['y'] = 1 - flipped['y']
    flipped['TeamID_Low'], flipped['TeamID_High'] = (
        train_df['TeamID_High'].values, train_df['TeamID_Low'].values)

    # FlipGroup keeps mirror pairs in the same CV fold
    train_df['FlipGroup'] = range(len(train_df))
    flipped['FlipGroup'] = range(len(train_df))

    out = pd.concat([train_df, flipped], ignore_index=True)
    print(f"after flip: {len(out)} rows")
    return out


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_all
    from src.data_cleaner import clean_data
    from src.ratings import compute_all_ratings

    ds = clean_data(load_all('.'))
    ratings = compute_all_ratings(ds)
    train_df = build_tourney_training_data(ds, ratings)
    for c in FEATURE_COLUMNS:
        if c in train_df.columns:
            n = train_df[c].isna().sum()
            if n > 0:
                print(f"{c}: {n} nan ({n/len(train_df)*100:.1f}%)")
    aug = augment_with_game_flip(train_df)
    print(f"target mean {aug['y'].mean():.3f} (~0.5)")

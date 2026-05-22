import numpy as np
import pandas as pd
from collections import defaultdict
import trueskill
import glicko2
from sklearn.linear_model import LogisticRegression


# --- elo ---
ELO_INITIAL = 1500
ELO_NEW_TEAM = 985
ELO_K_EARLY = 56
ELO_K_LATE = 38
ELO_K_TOURNEY_MULT = 1.25
ELO_EARLY_GAMES = 5
ELO_LATE_GAME = 20
ELO_HOME_ADV = 82
ELO_CARRYOVER_M = 0.65
ELO_CARRYOVER_W = 0.70


def _elo_expected(a, b):
    return 1.0 / (1.0 + 10.0 ** (-(a - b) / 400.0))


def _elo_mov_mult(mov, elo_diff):
    # MOV multiplier with autocorrelation adjustment (538-style)
    return ((abs(mov) + 3.0) ** 0.8) / (7.5 + 0.006 * abs(elo_diff))


def _elo_k_factor(game_num, is_tournament=False):
    if game_num <= ELO_EARLY_GAMES:
        k = ELO_K_EARLY
    elif game_num < ELO_LATE_GAME:
        progress = (game_num - ELO_EARLY_GAMES) / (ELO_LATE_GAME - ELO_EARLY_GAMES)
        k = ELO_K_EARLY - (ELO_K_EARLY - ELO_K_LATE) * progress
    else:
        k = ELO_K_LATE
    if is_tournament:
        k *= ELO_K_TOURNEY_MULT
    return k


def compute_elo(results_df, conferences_df, gender='M', margin_cap=50):
    carryover = ELO_CARRYOVER_M if gender == 'M' else ELO_CARRYOVER_W

    elo = defaultdict(lambda: ELO_INITIAL)
    seasons = sorted(results_df['Season'].unique())
    records = []

    for season in seasons:
        season_games = results_df[results_df['Season'] == season].sort_values('DayNum')

        # offseason regression toward conference mean
        if season != seasons[0]:
            if conferences_df is not None and len(conferences_df) > 0:
                conf_season = conferences_df[conferences_df['Season'] == season]
                conf_avg = {}
                for _, row in conf_season.iterrows():
                    conf_avg.setdefault(row['ConfAbbrev'], []).append(elo[row['TeamID']])
                conf_avg = {c: np.mean(v) for c, v in conf_avg.items()}
                global_avg = np.mean(list(conf_avg.values())) if conf_avg else ELO_INITIAL

                team_conf = dict(zip(conf_season['TeamID'], conf_season['ConfAbbrev']))
                for tid in set(season_games['WTeamID']) | set(season_games['LTeamID']):
                    target = conf_avg.get(team_conf.get(tid), global_avg)
                    elo[tid] = carryover * elo[tid] + (1 - carryover) * target
            else:
                tids = set(season_games['WTeamID']) | set(season_games['LTeamID'])
                global_avg = np.mean([elo[tid] for tid in tids])
                for tid in tids:
                    elo[tid] = carryover * elo[tid] + (1 - carryover) * global_avg

        season_team_counts = defaultdict(int)
        season_elo_changes = defaultdict(list)

        for _, game in season_games.iterrows():
            w_id, l_id = game['WTeamID'], game['LTeamID']
            w_score, l_score = game['WScore'], game['LScore']
            loc = game.get('WLoc', 'N')
            is_tourney = game['DayNum'] > 132

            # HCA on the home team only (not both - that double-counts)
            w_ha = ELO_HOME_ADV if loc == 'H' else 0
            l_ha = ELO_HOME_ADV if loc == 'A' else 0

            w_elo = elo[w_id] + w_ha
            l_elo = elo[l_id] + l_ha
            w_expected = _elo_expected(w_elo, l_elo)

            mov = min(abs(w_score - l_score), margin_cap)
            mov_mult = _elo_mov_mult(mov, w_elo - l_elo)

            season_team_counts[w_id] += 1
            season_team_counts[l_id] += 1
            k = _elo_k_factor(
                min(season_team_counts[w_id], season_team_counts[l_id]),
                is_tourney)

            change = k * mov_mult * (1 - w_expected)
            elo[w_id] += change
            elo[l_id] -= change

            season_elo_changes[w_id].append(change)
            season_elo_changes[l_id].append(-change)

        # snapshot at season end (just before tournament)
        all_teams = set(season_games['WTeamID']) | set(season_games['LTeamID'])
        for tid in all_teams:
            changes = season_elo_changes.get(tid, [])
            volatility = np.std(changes[-10:]) if len(changes) >= 3 else 0.0
            records.append({
                'Season': season,
                'TeamID': tid,
                'Elo': elo[tid],
                'EloVolatility': volatility,
                'Gender': gender
            })

    return pd.DataFrame(records)


# --- glicko-2 ---
def compute_glicko2(results_df, gender='M'):
    players = {}
    seasons = sorted(results_df['Season'].unique())
    records = []

    for season in seasons:
        season_games = results_df[results_df['Season'] == season].sort_values('DayNum')
        all_teams = set(season_games['WTeamID']) | set(season_games['LTeamID'])
        for tid in all_teams:
            if tid not in players:
                players[tid] = glicko2.Player()

        for _, game in season_games.iterrows():
            w_id, l_id = game['WTeamID'], game['LTeamID']
            wp, lp = players[w_id], players[l_id]
            # snapshot before update - both players need pre-game opponent state
            w_r, w_rd = wp.getRating(), wp.getRd()
            l_r, l_rd = lp.getRating(), lp.getRd()
            wp.update_player([l_r], [l_rd], [1])
            lp.update_player([w_r], [w_rd], [0])

        for tid in all_teams:
            p = players[tid]
            records.append({
                'Season': season,
                'TeamID': tid,
                'GlickoMu': p.getRating(),
                'GlickoPhi': p.getRd(),
                'GlickoSigma': p.vol if hasattr(p, 'vol') else 0.06,
                'Gender': gender
            })

    return pd.DataFrame(records)


# --- trueskill ---
def compute_trueskill(results_df, gender='M'):
    env = trueskill.TrueSkill(draw_probability=0.0)
    ratings = {}
    seasons = sorted(results_df['Season'].unique())
    records = []

    for season in seasons:
        season_games = results_df[results_df['Season'] == season].sort_values('DayNum')
        all_teams = set(season_games['WTeamID']) | set(season_games['LTeamID'])
        for tid in all_teams:
            if tid not in ratings:
                ratings[tid] = env.create_rating()

        for _, game in season_games.iterrows():
            w_id, l_id = game['WTeamID'], game['LTeamID']
            new_w, new_l = trueskill.rate_1vs1(ratings[w_id], ratings[l_id], env=env)
            ratings[w_id] = new_w
            ratings[l_id] = new_l

        for tid in all_teams:
            r = ratings[tid]
            records.append({
                'Season': season,
                'TeamID': tid,
                'TSMu': r.mu,
                'TSSigma': r.sigma,
                'Gender': gender
            })

    return pd.DataFrame(records)


# --- bradley-terry (logistic on one-hot team indicators) ---
def compute_bradley_terry(results_df, gender='M'):
    seasons = sorted(results_df['Season'].unique())
    records = []

    for season in seasons:
        season_games = results_df[(results_df['Season'] == season) &
                                  (results_df['DayNum'] <= 132)]
        if len(season_games) < 50:
            continue

        all_teams = sorted(set(season_games['WTeamID']) | set(season_games['LTeamID']))
        team_to_idx = {tid: i for i, tid in enumerate(all_teams)}
        n_teams = len(all_teams)
        n_games = len(season_games)

        # mirrored rows for symmetric P(A>B) = 1 - P(B>A)
        X = np.zeros((2 * n_games, n_teams))
        y = np.zeros(2 * n_games)
        for i, (_, game) in enumerate(season_games.iterrows()):
            w = team_to_idx[game['WTeamID']]
            l = team_to_idx[game['LTeamID']]
            X[2*i, w] = 1; X[2*i, l] = -1; y[2*i] = 1
            X[2*i+1, l] = 1; X[2*i+1, w] = -1; y[2*i+1] = 0

        try:
            lr = LogisticRegression(C=100.0, fit_intercept=False, max_iter=3000,
                                    solver='lbfgs', penalty='l2')
            lr.fit(X, y)
            strengths = lr.coef_[0]
        except Exception:
            strengths = np.zeros(n_teams)

        for tid, idx in team_to_idx.items():
            records.append({
                'Season': season,
                'TeamID': tid,
                'BTStrength': strengths[idx],
                'Gender': gender
            })

    return pd.DataFrame(records)


# --- massey ordinals (men only, Kaggle-provided third-party rankings) ---
def compute_massey_features(massey_df, target_systems=None):
    if len(massey_df) == 0:
        return pd.DataFrame()
    if target_systems is None:
        target_systems = ['POM', 'MOR']

    # day 133 == just before the tournament starts
    pre = massey_df[massey_df['RankingDayNum'] == 133]
    if len(pre) == 0:
        pre = massey_df[massey_df['RankingDayNum'] == massey_df['RankingDayNum'].max()]

    records = []
    for season in pre['Season'].unique():
        sd = pre[pre['Season'] == season]
        for tid in sd['TeamID'].unique():
            td = sd[sd['TeamID'] == tid]
            r = {'Season': season, 'TeamID': tid}
            for sys_name in target_systems:
                row = td[td['SystemName'] == sys_name]
                r[f'Massey_{sys_name}'] = (
                    _log_rank_transform(row['OrdinalRank'].values[0])
                    if len(row) > 0 else np.nan)
            ranks = td['OrdinalRank'].values
            if len(ranks) > 0:
                r['MasseyAvg'] = _log_rank_transform(np.mean(ranks))
                r['MasseyBest'] = _log_rank_transform(np.min(ranks))
            else:
                r['MasseyAvg'] = np.nan
                r['MasseyBest'] = np.nan
            records.append(r)
    return pd.DataFrame(records)


def _log_rank_transform(rank):
    # invert + compress: rank 1 -> ~97, rank 100 -> ~78, rank 350 -> ~62
    return 100 - 4 * np.log(rank + 1) - rank / 22


def compute_all_ratings(ds):
    print('elo m...'); elo_m = compute_elo(ds.m_reg_compact, ds.m_team_conferences, 'M')
    print('elo w...'); elo_w = compute_elo(ds.w_reg_compact, ds.w_team_conferences, 'W')
    print('glicko m...'); glicko_m = compute_glicko2(ds.m_reg_compact, 'M')
    print('glicko w...'); glicko_w = compute_glicko2(ds.w_reg_compact, 'W')
    print('trueskill m...'); ts_m = compute_trueskill(ds.m_reg_compact, 'M')
    print('trueskill w...'); ts_w = compute_trueskill(ds.w_reg_compact, 'W')
    print('bt m...'); bt_m = compute_bradley_terry(ds.m_reg_compact, 'M')
    print('bt w...'); bt_w = compute_bradley_terry(ds.w_reg_compact, 'W')
    print('massey...'); massey = compute_massey_features(ds.m_massey)

    return {
        'elo_m': elo_m, 'elo_w': elo_w,
        'glicko_m': glicko_m, 'glicko_w': glicko_w,
        'trueskill_m': ts_m, 'trueskill_w': ts_w,
        'bt_m': bt_m, 'bt_w': bt_w,
        'massey': massey,
    }


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_all
    ds = load_all('.')
    ratings = compute_all_ratings(ds)
    elo = ratings['elo_m']
    top = elo[elo['Season'] == 2026].nlargest(10, 'Elo')
    names = ds.m_teams.set_index('TeamID')['TeamName'].to_dict()
    for _, r in top.iterrows():
        print(f"{names.get(int(r['TeamID']), '?'):20s} {r['Elo']:.0f}")

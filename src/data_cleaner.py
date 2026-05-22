import numpy as np
import pandas as pd
from .data_loader import DataStore


MARGIN_CAP = 50          # cap blowout margins for MOV
MIN_GAMES_FOR_STATS = 15
COVID_2021_WEIGHT = 0.7  # 2021 bubble tournament was weird
REGULATION_MINUTES = 40
OT_MINUTES = 5


def clean_data(ds):
    ds = _normalize_overtime_stats(ds)
    ds = _add_possession_columns(ds)
    ds = _add_pace_adjusted_stats(ds)
    ds = _flag_covid_seasons(ds)
    return ds


def _normalize_overtime_stats(ds):
    # rescale counting stats to a 40-min basis when OT happened
    for df in [ds.m_reg_detailed, ds.m_tourney_detailed,
               ds.w_reg_detailed, ds.w_tourney_detailed]:
        if len(df) == 0:
            continue
        ot_mask = df['NumOT'] > 0
        if ot_mask.sum() == 0:
            continue
        total_minutes = REGULATION_MINUTES + OT_MINUTES * df.loc[ot_mask, 'NumOT']
        scale = REGULATION_MINUTES / total_minutes

        counting = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
                    'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
        for pre in ['W', 'L']:
            for stat in counting:
                col = f'{pre}{stat}'
                if col in df.columns:
                    df.loc[ot_mask, col] = df.loc[ot_mask, col] * scale
    return ds


def _add_possession_columns(ds):
    # Dean Oliver: poss = 0.5*((FGA + 0.475*FTA - ORB + TOV) + opp...)
    for df in [ds.m_reg_detailed, ds.m_tourney_detailed,
               ds.w_reg_detailed, ds.w_tourney_detailed]:
        if len(df) == 0:
            continue
        w_poss = df['WFGA'] + 0.475 * df['WFTA'] - df['WOR'] + df['WTO']
        l_poss = df['LFGA'] + 0.475 * df['LFTA'] - df['LOR'] + df['LTO']
        df['Possessions'] = (0.5 * (w_poss + l_poss)).clip(lower=1)
    return ds


def _add_pace_adjusted_stats(ds):
    for df in [ds.m_reg_detailed, ds.m_tourney_detailed,
               ds.w_reg_detailed, ds.w_tourney_detailed]:
        if len(df) == 0 or 'Possessions' not in df.columns:
            continue
        poss = df['Possessions']

        for pre in ['W', 'L']:
            opp = 'L' if pre == 'W' else 'W'

            df[f'{pre}OffEff'] = df[f'{pre}Score'] / poss * 100
            df[f'{pre}DefEff'] = df[f'{opp}Score'] / poss * 100

            fga = df[f'{pre}FGA']
            df[f'{pre}eFGPct'] = np.where(
                fga > 0,
                (df[f'{pre}FGM'] + 0.5 * df[f'{pre}FGM3']) / fga,
                0.0)

            denom = fga + 0.475 * df[f'{pre}FTA'] + df[f'{pre}TO']
            df[f'{pre}TOVPct'] = np.where(denom > 0, df[f'{pre}TO'] / denom, 0.0)

            orb_total = df[f'{pre}OR'] + df[f'{opp}DR']
            df[f'{pre}ORBPct'] = np.where(
                orb_total > 0, df[f'{pre}OR'] / orb_total, 0.0)

            df[f'{pre}FTRate'] = np.where(fga > 0, df[f'{pre}FTM'] / fga, 0.0)
            df[f'{pre}FGPct'] = np.where(fga > 0, df[f'{pre}FGM'] / fga, 0.0)

            # four factors composite, Oliver weights
            df[f'{pre}FourFactors'] = (
                0.40 * df[f'{pre}eFGPct'] +
                0.25 * (1 - df[f'{pre}TOVPct']) +
                0.20 * df[f'{pre}ORBPct'] +
                0.15 * df[f'{pre}FTRate'])
    return ds


def _flag_covid_seasons(ds):
    for df in [ds.m_tourney_compact, ds.m_tourney_detailed,
               ds.w_tourney_compact, ds.w_tourney_detailed]:
        if len(df) == 0:
            continue
        df['SampleWeight'] = 1.0
        df.loc[df['Season'] == 2021, 'SampleWeight'] = COVID_2021_WEIGHT
    return ds


def compute_conf_tourney_stats(conf_tourney_games, compact_df, season):
    ct = conf_tourney_games[conf_tourney_games['Season'] == season]
    if len(ct) == 0:
        return pd.DataFrame()

    records = []
    all_teams = set(ct['WTeamID']) | set(ct['LTeamID'])
    for tid in all_teams:
        wins = len(ct[ct['WTeamID'] == tid])
        losses = len(ct[ct['LTeamID'] == tid])
        games = wins + losses
        # champion: won their last game and lost nothing after it
        is_champ = 1 if wins > 0 and losses == 0 else 0
        if wins > 0:
            last_win_day = ct[ct['WTeamID'] == tid]['DayNum'].max()
            lost_after = ct[(ct['LTeamID'] == tid) & (ct['DayNum'] > last_win_day)]
            is_champ = 1 if len(lost_after) == 0 else 0

        records.append({
            'TeamID': tid,
            'ConfTourneyWins': wins,
            'ConfTourneyGames': games,
            'ConfTourneyChamp': is_champ,
        })

    if records:
        return pd.DataFrame(records).set_index('TeamID')
    return pd.DataFrame()


def compute_team_season_stats(detailed_df, compact_df, season,
                              min_games=MIN_GAMES_FOR_STATS):
    # regular season only: DayNum <= 132
    det = detailed_df[(detailed_df['Season'] == season) &
                      (detailed_df['DayNum'] <= 132)].copy()
    comp = compact_df[(compact_df['Season'] == season) &
                      (compact_df['DayNum'] <= 132)].copy()

    if len(comp) == 0:
        return pd.DataFrame()

    teams = set(comp['WTeamID'].unique()) | set(comp['LTeamID'].unique())
    records = []

    for team_id in teams:
        wins = comp[comp['WTeamID'] == team_id]
        losses = comp[comp['LTeamID'] == team_id]
        n_wins = len(wins)
        n_losses = len(losses)
        n_games = n_wins + n_losses

        if n_games < min_games:
            records.append({'TeamID': team_id, 'Games': n_games})
            continue

        win_pct = n_wins / n_games

        pts_for = wins['WScore'].sum() + losses['LScore'].sum()
        pts_against = wins['LScore'].sum() + losses['WScore'].sum()
        point_diff_avg = (pts_for - pts_against) / n_games

        # Morey-13.91 pythag for basketball
        exp = 13.91
        if pts_for > 0 and pts_against > 0:
            pyth_win_pct = (pts_for ** exp) / (pts_for ** exp + pts_against ** exp)
        else:
            pyth_win_pct = 0.5
        luck = win_pct - pyth_win_pct  # > 0 = overperforming pythag

        all_games = pd.concat([
            wins[['Season', 'DayNum', 'WTeamID']].rename(columns={'WTeamID': 'TeamID'}).assign(Won=1),
            losses[['Season', 'DayNum', 'LTeamID']].rename(columns={'LTeamID': 'TeamID'}).assign(Won=0),
        ]).sort_values('DayNum')
        last14 = all_games.tail(14)
        last14_win_pct = last14['Won'].mean() if len(last14) > 0 else win_pct

        # away/neutral perf is the tournament proxy
        away_neutral_wins = len(wins[wins['WLoc'].isin(['A', 'N'])])
        home_losses = len(losses[losses['WLoc'] == 'H'])
        neutral_losses = len(losses[losses['WLoc'] == 'N'])
        an_games = away_neutral_wins + home_losses + neutral_losses
        away_neutral_win_pct = away_neutral_wins / an_games if an_games > 0 else win_pct

        close_wins = len(wins[(wins['WScore'] - wins['LScore']) <= 5])
        close_losses = len(losses[(losses['WScore'] - losses['LScore']) <= 5])
        close_games = close_wins + close_losses
        close_game_win_pct = close_wins / close_games if close_games > 0 else 0.5

        record = {
            'TeamID': team_id,
            'Games': n_games,
            'WinPct': win_pct,
            'PointDiffAvg': point_diff_avg,
            'PythWinPct': pyth_win_pct,
            'Luck': luck,
            'Last14WinPct': last14_win_pct,
            'AwayNeutralWinPct': away_neutral_win_pct,
            'CloseGameWinPct': close_game_win_pct,
        }

        if len(det) > 0 and 'Possessions' in det.columns:
            t_wins = det[det['WTeamID'] == team_id]
            t_losses = det[det['LTeamID'] == team_id]

            off_eff_vals = []
            def_eff_vals = []
            efg_vals = []
            tov_vals = []
            orb_vals = []
            ftr_vals = []
            def_efg_vals = []
            def_tov_vals = []

            if len(t_wins) > 0:
                off_eff_vals.extend(t_wins['WOffEff'].values)
                def_eff_vals.extend(t_wins['WDefEff'].values)
                efg_vals.extend(t_wins['WeFGPct'].values)
                tov_vals.extend(t_wins['WTOVPct'].values)
                orb_vals.extend(t_wins['WORBPct'].values)
                ftr_vals.extend(t_wins['WFTRate'].values)
                def_efg_vals.extend(t_wins['LeFGPct'].values)
                def_tov_vals.extend(t_wins['LTOVPct'].values)

            if len(t_losses) > 0:
                off_eff_vals.extend(t_losses['LOffEff'].values)
                def_eff_vals.extend(t_losses['LDefEff'].values)
                efg_vals.extend(t_losses['LeFGPct'].values)
                tov_vals.extend(t_losses['LTOVPct'].values)
                orb_vals.extend(t_losses['LORBPct'].values)
                ftr_vals.extend(t_losses['LFTRate'].values)
                def_efg_vals.extend(t_losses['WeFGPct'].values)
                def_tov_vals.extend(t_losses['WTOVPct'].values)

            if off_eff_vals:
                record['OffEff'] = np.mean(off_eff_vals)
                record['DefEff'] = np.mean(def_eff_vals)
                record['NetEff'] = record['OffEff'] - record['DefEff']
                record['eFGPct'] = np.mean(efg_vals)
                record['TOVPct'] = np.mean(tov_vals)
                record['ORBPct'] = np.mean(orb_vals)
                record['FTRate'] = np.mean(ftr_vals)
                record['DefeFGPct'] = np.mean(def_efg_vals)
                record['DefTOVPct'] = np.mean(def_tov_vals)
                record['FourFactors'] = (
                    0.40 * record['eFGPct'] +
                    0.25 * (1 - record['TOVPct']) +
                    0.20 * record['ORBPct'] +
                    0.15 * record['FTRate']
                )

        records.append(record)

    result = pd.DataFrame(records)
    if len(result) > 0:
        result = result.set_index('TeamID')
    return result


def compute_opponent_adjusted_efficiency(team_stats):
    # full opponent adjustment lives in features.py (SOS via Elo);
    # leave raw values here as the placeholder
    if 'OffEff' not in team_stats.columns or len(team_stats) == 0:
        return team_stats
    national_avg_off = team_stats['OffEff'].mean()
    national_avg_def = team_stats['DefEff'].mean()
    team_stats['AdjOffEff'] = team_stats.get('OffEff', national_avg_off)
    team_stats['AdjDefEff'] = team_stats.get('DefEff', national_avg_def)
    return team_stats


if __name__ == '__main__':
    from .data_loader import load_all
    ds = clean_data(load_all('.'))
    stats = compute_team_season_stats(ds.m_reg_detailed, ds.m_reg_compact, 2026)
    poss = ds.m_reg_detailed[ds.m_reg_detailed['Season'] == 2026]['Possessions']
    print(f'2026 m teams: {len(stats)}, poss mean {poss.mean():.1f} (~67-70 ok)')
    if 'NetEff' in stats.columns:
        print(stats.nlargest(10, 'NetEff')[['WinPct', 'OffEff', 'DefEff', 'NetEff']])

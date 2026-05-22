import os
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class DataStore:
    # Men's data
    m_teams: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_seasons: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_reg_compact: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_reg_detailed: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_tourney_compact: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_tourney_detailed: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_seeds: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_tourney_slots: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_seed_round_slots: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_massey: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_coaches: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_team_conferences: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_conf_tourney_games: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_secondary_tourney_teams: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_secondary_tourney_compact: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_game_cities: pd.DataFrame = field(default_factory=pd.DataFrame)
    m_team_spellings: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Women's data
    w_teams: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_seasons: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_reg_compact: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_reg_detailed: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_tourney_compact: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_tourney_detailed: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_seeds: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_tourney_slots: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_team_conferences: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_conf_tourney_games: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_secondary_tourney_teams: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_secondary_tourney_compact: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_game_cities: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_team_spellings: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Shared data
    cities: pd.DataFrame = field(default_factory=pd.DataFrame)
    conferences: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Submission templates
    submission_stage1: pd.DataFrame = field(default_factory=pd.DataFrame)
    submission_stage2: pd.DataFrame = field(default_factory=pd.DataFrame)


def parse_seed(seed_str):
    # 'W01' -> 1, 'X16a' -> 16, 'Z11b' -> 11
    return int(seed_str[1:].rstrip('ab'))


def parse_submission_id(id_str):
    parts = id_str.split('_')
    return int(parts[0]), int(parts[1]), int(parts[2])


def get_gender(team_id):
    return 'M' if team_id < 3000 else 'W'


def load_all(data_dir):
    ds = DataStore()

    def load(filename):
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        print(f'missing: {filename}')
        return pd.DataFrame()

    # Men's data
    ds.m_teams = load('MTeams.csv')
    ds.m_seasons = load('MSeasons.csv')
    ds.m_reg_compact = load('MRegularSeasonCompactResults.csv')
    ds.m_reg_detailed = load('MRegularSeasonDetailedResults.csv')
    ds.m_tourney_compact = load('MNCAATourneyCompactResults.csv')
    ds.m_tourney_detailed = load('MNCAATourneyDetailedResults.csv')
    ds.m_seeds = load('MNCAATourneySeeds.csv')
    ds.m_tourney_slots = load('MNCAATourneySlots.csv')
    ds.m_seed_round_slots = load('MNCAATourneySeedRoundSlots.csv')
    ds.m_massey = load('MMasseyOrdinals.csv')
    ds.m_coaches = load('MTeamCoaches.csv')
    ds.m_team_conferences = load('MTeamConferences.csv')
    ds.m_conf_tourney_games = load('MConferenceTourneyGames.csv')
    ds.m_secondary_tourney_teams = load('MSecondaryTourneyTeams.csv')
    ds.m_secondary_tourney_compact = load('MSecondaryTourneyCompactResults.csv')
    ds.m_game_cities = load('MGameCities.csv')
    ds.m_team_spellings = load('MTeamSpellings.csv')

    # Women's data
    ds.w_teams = load('WTeams.csv')
    ds.w_seasons = load('WSeasons.csv')
    ds.w_reg_compact = load('WRegularSeasonCompactResults.csv')
    ds.w_reg_detailed = load('WRegularSeasonDetailedResults.csv')
    ds.w_tourney_compact = load('WNCAATourneyCompactResults.csv')
    ds.w_tourney_detailed = load('WNCAATourneyDetailedResults.csv')
    ds.w_seeds = load('WNCAATourneySeeds.csv')
    ds.w_tourney_slots = load('WNCAATourneySlots.csv')
    ds.w_team_conferences = load('WTeamConferences.csv')
    ds.w_conf_tourney_games = load('WConferenceTourneyGames.csv')
    ds.w_secondary_tourney_teams = load('WSecondaryTourneyTeams.csv')
    ds.w_secondary_tourney_compact = load('WSecondaryTourneyCompactResults.csv')
    ds.w_game_cities = load('WGameCities.csv')
    ds.w_team_spellings = load('WTeamSpellings.csv')

    # Shared
    ds.cities = load('Cities.csv')
    ds.conferences = load('Conferences.csv')

    # Submission templates
    ds.submission_stage1 = load('SampleSubmissionStage1.csv')
    ds.submission_stage2 = load('SampleSubmissionStage2.csv')

    for seeds_df in [ds.m_seeds, ds.w_seeds]:
        if len(seeds_df) > 0:
            seeds_df['SeedNum'] = seeds_df['Seed'].apply(parse_seed)

    # Elo needs chronological order
    for df in [ds.m_reg_compact, ds.m_reg_detailed, ds.w_reg_compact, ds.w_reg_detailed]:
        if len(df) > 0:
            df.sort_values(['Season', 'DayNum'], inplace=True)
            df.reset_index(drop=True, inplace=True)

    print(f"loaded: {len(ds.m_reg_compact)} m-games, "
          f"{len(ds.w_reg_compact)} w-games, "
          f"{len(ds.m_massey)} massey")

    return ds


if __name__ == '__main__':
    ds = load_all('.')
    print(ds.m_seeds[ds.m_seeds['Season'] == 2026].head())

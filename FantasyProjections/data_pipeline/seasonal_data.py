
import dateutil.parser as dateparse
import pandas as pd
from config import data_files_config
from data_pipeline import team_abbreviations
from data_pipeline.single_game_data import SingleGamePbpParser
from data_pipeline.data_helper_functions import compute_team_record, parse_year_from_date, clean_team_names
from misc.manage_files import collect_input_dfs


class SeasonalDataCollector():
    def __init__(self, year, team_names, weeks=range(1,19), **kwargs):
        game_times = kwargs.get('game_times', 'all')
        filter_df= kwargs.get('filter_df', None)

        self.year = year
        self.weeks = weeks
        self.team_names = clean_team_names(team_names, self.year)

        # Collect input data
        self.pbp_df, self.raw_rosters_df, *_ = collect_input_dfs(self.year, self.weeks, data_files_config.local_file_paths,
                                                        data_files_config.online_file_paths, online_avail=True)

        # Process team records, game sites, and other game info for every game of the year, for every team
        self.all_game_info_df = self.get_game_info()

        # Gather team roster for all teams, all weeks of input year
        self.all_rosters_df = self.process_rosters(filter_df)

        # List of SingleGameData objects
        self.games = self.gen_and_process_games(game_times)

        # Gather all stats (midgame and final) from the individual teams
        self.midgame_df, self.final_stats_df = self.gather_all_game_stats()

    # PUBLIC METHODS

    def gather_all_game_stats(self):

        midgame_df = pd.DataFrame()
        final_stats_df = pd.DataFrame()
        for game in self.games:
            midgame_df = pd.concat((midgame_df, game.midgame_df))
            final_stats_df = pd.concat((final_stats_df, game.final_stats_df))

        print(f"\tdata rows: {midgame_df.shape[0]}")

        return midgame_df, final_stats_df

    def gen_and_process_games(self, game_times):
        games = []
        team_abbrevs = [team_abbreviations.pbp_abbrevs[name] for name in self.team_names]
        for game_id in self.all_game_info_df.index.unique():
            game_info = self.all_game_info_df.loc[game_id].iloc[0]
            week = int(game_info['Week'])
            home_team = game_info['Home Team Abbrev']
            away_team = game_info['Away Team Abbrev']
            if week in self.weeks and (home_team in team_abbrevs or away_team in team_abbrevs):
                print(f"{game_id}")
                # Process data/stats for single game
                game = SingleGamePbpParser(self, game_id, week, game_times=game_times)
                # Add to list of games
                games.append(game)

        return games


    def get_game_info(self):
        """Generates info on every game for each team in a given year: who is home vs away, and records of each team going into the game.

            Also generates url to get game stats from pro-football-reference.com.
            Note that each game is included twice - once from the perspective of each team (i.e. "Team" and "Opponent" info are swapped).

            Attributes Modified:
                all_game_info_df: DataFrame containing game info on each unique game in the NFL season being processed (teams, pre-game team records, etc.)
        """
        pbp_df = self.pbp_df.copy()

        # Filter df to only the final play of each game
        pbp_df = pbp_df.drop_duplicates(subset='game_id',keep='last')

        # Filter to only regular-season games
        pbp_df = pbp_df[pbp_df['season_type']=='REG']

        # Output data frame
        all_game_info_df = pd.DataFrame()

        for team_abbr in team_abbreviations.pbp_abbrevs.values():
            # Filter to only games including the team of interest
            scores_df = pbp_df.copy()
            scores_df = scores_df[(scores_df['home_team'] == team_abbr)
                                | (scores_df['away_team'] == team_abbr)]

            # Track team name and opponent name
            scores_df['Team Abbrev'] = team_abbr
            scores_df['Opp Abbrev'] = scores_df.apply(lambda x:
                x['away_team'] if x['home_team'] == x['Team Abbrev'] else x['home_team'],
                axis=1)

            # Track game site and home/away team
            scores_df['Site'] = scores_df.apply(
                lambda x: 'Home' if x['home_team'] == x['Team Abbrev'] else 'Away', axis=1)
            scores_df = scores_df.rename(columns={'week':'Week','home_team':'Home Team Abbrev','away_team':'Away Team Abbrev'})

            # URL to get stats from
            scores_df['game_date'] = scores_df.apply(lambda x:
                dateparse.parse(x['game_date']).strftime('%Y%m%d'), axis=1)
            scores_df['PFR URL'] = scores_df.apply(lambda x:
                data_files_config.URL_INTRO + x['game_date'] + '0' +
                team_abbreviations.convert_abbrev(x['Home Team Abbrev'],
                team_abbreviations.pbp_abbrevs,team_abbreviations.roster_website_abbrevs
                ) + '.htm', axis=1)

            # Track ties, wins, and losses
            scores_df = compute_team_record(scores_df)

            # Remove unnecessary columns
            columns_to_keep = [
                'Week',
                'Team Abbrev',
                'Opp Abbrev',
                'game_id',
                'PFR URL',
                'Site',
                'Home Team Abbrev',
                'Away Team Abbrev',
                'Team Wins',
                'Team Losses',
                'Team Ties']
            scores_df = scores_df[columns_to_keep].set_index(['Week', 'Team Abbrev']).sort_index()

            # Append to dataframe of all teams' games
            all_game_info_df = pd.concat([all_game_info_df, scores_df])

        # Clean up df for output
        all_game_info_df = all_game_info_df.reset_index().set_index(['game_id']).sort_index()
        all_game_info_df['Team Name'] = all_game_info_df['Team Abbrev'].apply(lambda x:
            team_abbreviations.invert(team_abbreviations.pbp_abbrevs)[x])

        return all_game_info_df


    def process_rosters(self, filter_df=None):
        """Trims DataFrame of all NFL week-by-week rosters in a given year to include only players of interest and data columns of interest.

            Args:
                filter_df (pandas.DataFrame, optional): Pre-determined list of players to include. Defaults to None.

            Attributes Modified:
                pandas.DataFrame: all_rosters_df filtered to players of interest, several columns removed, and indexed on Team & Week
        """

        # Copy of object attribute
        all_rosters_df = self.raw_rosters_df.copy()

        # Filter to only the desired weeks
        all_rosters_df = all_rosters_df[all_rosters_df.apply(
            lambda x: x['week'] in self.weeks, axis=1)]

        # Filter to only the desired teams
        all_rosters_df = all_rosters_df[all_rosters_df.apply(lambda x: x['team'] in
            [team_abbreviations.pbp_abbrevs[name] for name in self.team_names], axis=1)]

        # Optionally filter based on subset of desired players
        if filter_df is not None:
            all_rosters_df = all_rosters_df[all_rosters_df.apply(
                lambda x: x['full_name'] in filter_df['Name'].to_list(), axis=1)]

        # Filter to only skill positions
        # Positions currently being tracked for stats
        skill_positions = ['QB', 'RB', 'FB', 'HB', 'WR', 'TE']
        all_rosters_df = all_rosters_df[all_rosters_df.apply(
            lambda x: x['position'] in skill_positions, axis=1)]

        # Filter to only active players
        valid_statuses = ['ACT']
        all_rosters_df = all_rosters_df[all_rosters_df.apply(
            lambda x: x['status'] in valid_statuses, axis=1)]


        # Compute age based on birth date
        all_rosters_df['Age'] = all_rosters_df['season'] - all_rosters_df['birth_date'].apply(parse_year_from_date)

        # Trim to just the fields that are useful
        all_rosters_df=all_rosters_df[[
            'team',
            'week',
            'position',
            'jersey_number',
            'full_name',
            'gsis_id',
            'Age']]
        # Reformat
        all_rosters_df=all_rosters_df.rename(columns=
            {
            'team': 'Team',
            'week': 'Week',
            'position': 'Position',
            'jersey_number': 'Number',
            'full_name': 'Name',
            'gsis_id': 'Player ID'
            }).set_index(['Team','Week']).sort_index()

        return all_rosters_df

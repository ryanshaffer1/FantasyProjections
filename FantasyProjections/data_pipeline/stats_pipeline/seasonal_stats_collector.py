"""Creates and exports class to be used in NFL player statistics data collection.

    Classes:
        SeasonalStatsCollector : Collects all player stats for all games in an NFL season. Automatically pulls data from nfl-verse and processes upon initialization.
"""

import logging

import dateutil.parser as dateparse
import pandas as pd
from config import data_files_config
from data_pipeline.seasonal_data_collector import SeasonalDataCollector
from data_pipeline.stats_pipeline.single_game_pbp_parser import SingleGamePbpParser
from data_pipeline.utils import team_abbreviations as team_abbrs
from data_pipeline.utils.data_helper_functions import compute_team_record

# Set up logger
logger = logging.getLogger('log')

class SeasonalStatsCollector(SeasonalDataCollector):
    """Collects all player stats for all games in an NFL season. Automatically pulls data from nfl-verse and processes upon initialization.
    
        Args:
            year (int): Year for season (e.g. 2023).
            team_names (str | list, optional): Either "all" or a list of full team names (e.g. ["Arizona Cardinals", "Baltimore Ravens", ...]). Defaults to "all".
            weeks (list, optional): Weeks in the NFL season to collect data for. Defaults to range(1,19).
        Keyword Arguments: 
            game_times (list): Elapsed time steps to save data for (e.g. every minute of the game, every 5 minutes, etc.). Defaults to "all", meaning every play.
                Not stored as an object attribute.
            filter_df (pandas.DataFrame): Filter for roster, i.e. the list of players to collect data for. Defaults to None (collect data for every player).
                Not stored as an object attribute.

        Additional Attributes Created during Initialization:
            pbp_df (pandas.DataFrame): Play-by-play data for all plays in an NFL season, taken from nfl-verse.
            raw_rosters_df (pandas.DataFrame): Weekly roster information for all teams in an NFL season, taken from nfl-verse.
            all_game_info_df (pandas.DataFrame): Information setting context for each game in an NFL season, including home/away teams, team records, etc.
            all_rosters_df (pandas.DataFrame): Filtered roster dataframe to include only players of interest (skill positions, in the optional filter_df, etc.)
            games (list): List of SingleGamePbpParser objects containing data for every game in the NFL season.
            midgame_df (pandas.DataFrame): All midgame statistics for each player of interest, in every game of the NFL season. 
                Sampled throughout the game according to optional game_times input.
            final_stats_df (pandas.DataFrame): All final statistics for each player of interest, in every game of the NFL season.

        Objects Created:
            List of SingleGamePbpParser objects
        
        Public Methods: 
            generate_games : Creates a SingleGamePbpParser object for each unique game included in the SeasonalDataCollector.
            get_game_info : Generates info on every game for each team in a given year: who is home vs away, and records of each team going into the game.     
    """

    def __init__(self, year, team_names='all', weeks=None, **kwargs):
        """Constructor for SeasonalDataCollector class.

            Args:
                year (int): Year for season (e.g. 2023).
                team_names (str | list, optional): Either "all" or a list of full team names (e.g. ["Arizona Cardinals", "Baltimore Ravens", ...]). Defaults to "all".
                weeks (list, optional): Weeks in the NFL season to collect data for. Defaults to range(1,19).
            Keyword Arguments: 
                game_times (list | str, optional): Elapsed time steps to save data for (e.g. every minute of the game, every 5 minutes, etc.). Defaults to "all", meaning every play.
                    Not stored as an object attribute.
                filter_df (pandas.DataFrame, optional): Filter for roster, i.e. the list of players to collect data for. Defaults to None (collect data for every player).
                    Not stored as an object attribute.

            Additional Attributes Created during Initialization:
                pbp_df (pandas.DataFrame): Play-by-play data for all plays in an NFL season, taken from nfl-verse.
                raw_rosters_df (pandas.DataFrame): Weekly roster information for all teams in an NFL season, taken from nfl-verse.
                all_game_info_df (pandas.DataFrame): Information setting context for each game in an NFL season, including home/away teams, team records, etc.
                all_rosters_df (pandas.DataFrame): Filtered roster dataframe to include only players of interest (skill positions, in the optional filter_df, etc.)
                games (list): List of SingleGamePbpParser objects containing data for every game in the NFL season.
                midgame_df (pandas.DataFrame): All midgame statistics for each player of interest, in every game of the NFL season. 
                    Sampled throughout the game according to optional game_times input.
                final_stats_df (pandas.DataFrame): All final statistics for each player of interest, in every game of the NFL season.

        """

        # Initialize SeasonalDataCollector
        super().__init__(year=year, team_names=team_names, weeks=weeks, **kwargs)

        # Optional keyword arguments
        game_times = kwargs.get('game_times', 'all')

        # Process team records, game sites, and other game info for every game of the year, for every team
        self.all_game_info_df = self.get_game_info()

        # List of SingleGameData objects
        self.games = self.generate_games(game_times)

        # Gather all stats (midgame and final) from the individual teams
        self.midgame_df, self.final_stats_df, *_ = self.gather_all_game_data(['midgame_df', 'final_stats_df'])


    # PUBLIC METHODS

    def generate_games(self, game_times):
        """Creates a SingleGamePbpParser object for each unique game included in the SeasonalDataCollector.

            Args:
                game_times (list | str): Elapsed time steps to save data for (e.g. every minute of the game, every 5 minutes, etc.). May be "all", meaning every play.

            Returns:
                list: List of SingleGamePbpParser objects containing data for each game in the SeasonalDataCollector
        """

        team_abbrevs_to_process = [team_abbrs.pbp_abbrevs[name] for name in self.team_names]

        # Generate list of game IDs to process (based on weeks and teams to include)
        game_ids = []
        for game_id in self.all_game_info_df.index.unique():
            game_info = self.all_game_info_df.loc[game_id].iloc[0]
            week = int(game_info['Week'])
            home_team = game_info['Home Team Abbrev']
            away_team = game_info['Away Team Abbrev']
            if week in self.weeks and (home_team in team_abbrevs_to_process or away_team in team_abbrevs_to_process):
                game_ids.append(game_id)

        # Create objects that process stats for each game of interest
        games = []
        n_games = len(game_ids)
        logger.info(f'Processing {n_games} games:')
        for i, game_id in enumerate(game_ids):
            logger.info(f'({i+1} of {n_games}): {game_id}')
            # Process data/stats for single game
            game = SingleGamePbpParser(self, game_id, game_times=game_times)
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

        for team_abbr in team_abbrs.pbp_abbrevs.values():
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
                data_files_config.PFR_BOXSCORE_URL_INTRO + x['game_date'] + '0' +
                team_abbrs.convert_abbrev(x['Home Team Abbrev'],
                team_abbrs.pbp_abbrevs,team_abbrs.roster_website_abbrevs
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
            team_abbrs.invert(team_abbrs.pbp_abbrevs)[x])

        return all_game_info_df

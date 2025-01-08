"""Creates and exports class to be used in NFL player statistics data collection.

    Classes:
        SeasonalDataCollector : Collects all player stats for all games in an NFL season. Automatically pulls data from nfl-verse and processes upon initialization.
"""

import logging
import json
from datetime import datetime
import requests
from data_pipeline.seasonal_data_collector import SeasonalDataCollector
from data_pipeline.utils import team_abbreviations as team_abbrs
from data_pipeline.odds_pipeline.single_game_odds_gatherer import SingleGameOddsGatherer
from data_pipeline.odds_pipeline.odds_api_helper_functions import SPORT_KEY, DATE_FMT, get_odds_api_key, log_api_usage
from data_pipeline.utils.data_helper_functions import construct_game_id
from data_pipeline.utils.time_helper_functions import week_to_date_range, date_to_nfl_week

# Set up logger
logger = logging.getLogger('log')

class SeasonalOddsCollector(SeasonalDataCollector):
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
            gather_all_game_stats : Concatenates all statistics (midgame_df and final_stats_df) from individual games in self.games into larger DataFrames for the full season.
            generate_games : Creates a SingleGamePbpParser object for each unique game included in the SeasonalDataCollector.
            get_game_info : Generates info on every game for each team in a given year: who is home vs away, and records of each team going into the game.     
            process_rosters : Trims DataFrame of all NFL week-by-week rosters in a given year to include only players of interest and data columns of interest.
    """

    def __init__(self, year, team_names='all', weeks=range(1,19), **kwargs):
        """Constructor for SeasonalDataCollector class.

            Args:
                year (int): Year for season (e.g. 2023).
                team_names (str | list, optional): Either "all" or a list of full team names (e.g. ["Arizona Cardinals", "Baltimore Ravens", ...]). Defaults to "all".
                weeks (list, optional): Weeks in the NFL season to collect data for. Defaults to range(1,19).
            Keyword Arguments: 
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
        player_props = kwargs.get('player_props', None)
        odds_file = kwargs.get('odds_file', None)


        # API Key
        self.api_key = get_odds_api_key()

        # List of SingleGameData objects
        self.games = self.generate_games(odds_file=odds_file, player_props=player_props)

        # Gather all stats (midgame and final) from the individual teams
        self.odds_df, *_ = self.gather_all_game_data(['odds_df'])


    # PUBLIC METHODS

    def generate_games(self, odds_file=None, player_props=None):
        team_abbrevs_to_process = [team_abbrs.pbp_abbrevs[name] for name in self.team_names]

        year_start_date, _ = week_to_date_range(self.year, week=1)
        year_start_date = datetime.strftime(year_start_date, DATE_FMT)
        # Get all events in week
        response = requests.get(f'https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/events', params={
            'api_key': self.api_key,
            'date': year_start_date
        }, timeout=10)
        log_api_usage(response)
        events_list = json.loads(response.text)['data']

        # Extract IDs from each event
        event_ids = []
        game_ids = []
        for event in events_list:
            event_ids.append(event['id'])
            year, week = date_to_nfl_week(event['commence_time'])
            home_team = team_abbrs.pbp_abbrevs[event['home_team']]
            away_team = team_abbrs.pbp_abbrevs[event['away_team']]
            game_data = {'Year': year,
                         'Week': week,
                         'Home Team': home_team,
                         'Away Team': away_team}
            if week in self.weeks and (home_team in team_abbrevs_to_process or away_team in team_abbrevs_to_process):
                game_ids.append(construct_game_id(game_data))

        # Create objects that process odds for each game of interest
        games = []
        n_games = len(game_ids)
        logger.info(f'Processing {n_games} games:')
        for i, (event_id, game_id) in enumerate(zip(event_ids, game_ids)):
            logger.info(f'({i+1} of {n_games}): {game_id}')
            # Process data/stats for single game
            game = SingleGameOddsGatherer(self, event_id, game_id, odds_file=odds_file, player_props=player_props)
            # Add to list of games
            games.append(game)

        return games

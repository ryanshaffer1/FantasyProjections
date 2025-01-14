"""Creates and exports class to be used in NFL player statistics data collection.

    Classes:
        SingleGamePbpParser : Collects all player stats for a single NFL game. Automatically processes data upon initialization.
"""
from datetime import datetime
import logging
import json
import dateutil.parser as dateparse
import requests
import numpy as np
import pandas as pd
from config import stats_config
from config.player_id_config import PRIMARY_PLAYER_ID
from data_pipeline.single_game_data_worker import SingleGameDataWorker
from data_pipeline.odds_pipeline.odds_api_helper_functions import SPORT_KEY, DATE_FMT, BOOKMAKER, log_api_usage
from data_pipeline.utils.name_matching import find_matching_name_ind
from data_pipeline.utils.time_helper_functions import find_prev_time_index

# Set up logger
logger = logging.getLogger('log')

class SingleGameOddsGatherer(SingleGameDataWorker):
    """Collects all player stats for a single NFL game. Automatically processes data upon initialization.
    
        Args:
            seasonal_data (SeasonalDataCollector): "Parent" object containing data relating to the NFL season.
                Not stored as an object attribute.
            game_id (str): Game ID for specific game, as used by nfl-verse. Format is "{year}_{week}_{awayteam}_{home_team}", ex: "2021_01_ARI_TEN"
        Keyword Arguments: 
            game_times (list | str, optional): Elapsed time steps to save data for (e.g. every minute of the game, every 5 minutes, etc.). Defaults to "all", meaning every play.
                Not stored as an object attribute.

        Additional Attributes Created during Initialization:
            year (int): Year of game being processed
            week (int): Week in NFL season of game being processed
            game_info (pandas.DataFrame): Information setting context for the game, including home/away teams, team records, etc.
            roster_df (pandas.DataFrame): Players in the game (from both teams) to collect stats for.
            pbp_df (pandas.DataFrame): Play-by-play data for all plays in the game, taken from nfl-verse.
            midgame_df (pandas.DataFrame): All midgame statistics for each player of interest over the course of the game. 
                Sampled throughout the game according to optional game_times input.
            final_stats_df (pandas.DataFrame): All final statistics for each player of interest at the end of the game.
        
        Public Methods: 
            parse_play_by_play : Calculates midgame and final statistics for all players in a game, using play-by-play data describing passes, rushes, etc.
    """

    def __init__(self, seasonal_data, event_id, game_id, **kwargs):
        """Constructor for SingleGamePbpParser object.

            Args:
                seasonal_data (SeasonalDataCollector): "Parent" object containing data relating to the NFL season.
                    Not stored as an object attribute.
                game_id (str): Game ID for specific game, as used by nfl-verse. Format is "{year}_{week}_{awayteam}_{home_team}", ex: "2021_01_ARI_TEN"

            Additional Attributes Created during Initialization:
                year (int): Year of game being processed
                week (int): Week in NFL season of game being processed
                game_info (pandas.DataFrame): Information setting context for the game, including home/away teams, team records, etc.
                roster_df (pandas.DataFrame): Players in the game (from both teams) to collect stats for.
                pbp_df (pandas.DataFrame): Play-by-play data for all plays in the game, taken from nfl-verse.
                midgame_df (pandas.DataFrame): All midgame statistics for each player of interest over the course of the game. 
                    Sampled throughout the game according to optional game_times input.
                final_stats_df (pandas.DataFrame): All final statistics for each player of interest at the end of the game.
        """

        # Initialize SingleGameDataWorker
        super().__init__(seasonal_data=seasonal_data, game_id=game_id)

        # Optional keyword arguments
        player_props = kwargs.get('player_props', None)
        odds_file = kwargs.get('odds_file', None)

        # Basic info
        self.api_key = seasonal_data.api_key
        self.event_id = event_id

        # Helpful: maps game time to UTC time
        self.all_game_times = self.pbp_df['time_of_day'].dropna()

        # Load existing set of odds for this game
        try:
            odds_df = pd.read_csv(odds_file)
            odds_df = odds_df[odds_df['Game ID'] == self.game_id]
        except (FileNotFoundError, ValueError):
            odds_df = pd.DataFrame()

        # Determine the list of odds to collect
        self.set_markets(odds_df, player_props)

        # Collect additional odds from API
        if len(self.markets) > 0:
            self.odds_df = self.gather_historical_odds(odds_df=odds_df, game_times=[0,30])
        else:
            self.odds_df = odds_df

        # Format self.odds_df for output
        self.reformat_odds_df()


    def gather_historical_odds(self, odds_df=None, game_times=None):
        # Default game time is just 0 (pre-game odds)
        if game_times is None:
            game_times = [0]

        # Optional input: pre-existing dataframe to append to
        if odds_df is None:
            odds_df = pd.DataFrame()

        for game_time in game_times:
            pbp_index = find_prev_time_index(game_time, self.all_game_times)
            time = datetime.strftime(dateparse.parse(self.all_game_times.iloc[pbp_index]), DATE_FMT)

            response = requests.get(f'https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/events/{self.event_id}/odds', params={
                'api_key': self.api_key,
                'regions': 'us',
                'markets': ','.join(self.markets),
                'date': time,
                }, timeout=10)
            log_api_usage(response)
            try:
                event_data = json.loads(response.text)['data']
            except KeyError:
                logger.warning(f'Key Error encountered in API request. Aborting game {self.game_id} early.')
                return odds_df

            market_data = pd.DataFrame(event_data['bookmakers']).set_index('key').loc[BOOKMAKER, 'markets']
            for market in market_data:
                # Process Odds data into a DataFrame
                df = pd.DataFrame(market['outcomes']
                                ).rename(columns={'description': 'Player Name', 'name': 'Line'})
                # Add Player Prop Stat and UTC time from market data
                df['Player Prop Stat'] = market['key']
                df['UTC Time'] = market['last_update']
                # Add to running list of all odds
                odds_df = pd.concat((odds_df, df))

            return odds_df


    def reformat_odds_df(self):

        # Modify into the desired DataFrame structure
        if 'Line' in self.odds_df.columns:
            self.odds_df = self.odds_df.set_index(['Player Name', 'UTC Time'])
            for line in ['Over','Under']:
                self.odds_df[f'{line} Point'] = self.odds_df.reset_index().set_index(['Player Name','UTC Time','Line']).xs(line,level=2)['point']
                self.odds_df[f'{line} Price'] = self.odds_df.reset_index().set_index(['Player Name','UTC Time','Line']).xs(line,level=2)['price']
            self.odds_df = self.odds_df.reset_index().drop(columns=['Line','price','point']).drop_duplicates(keep='first')

        # Add Year, Week, and Game ID
        self.odds_df['Year'] = self.year
        self.odds_df['Week'] = self.week
        self.odds_df['Game ID'] = self.game_id

        # Add Player ID based on roster
        roster_inds = self.odds_df['Player Name'].apply(find_matching_name_ind, args=(self.roster_df['Player Name'].unique(),))
        self.odds_df['Player ID'] = roster_inds.apply(lambda x: self.roster_df.reset_index().iloc[int(x)][PRIMARY_PLAYER_ID] if not np.isnan(x) else np.nan)

        elapsed_time_inds = self.odds_df['UTC Time'].apply(
            find_prev_time_index, args=(self.all_game_times,))
        self.odds_df['Elapsed Time'] = self.all_game_times.index[elapsed_time_inds].to_list()

        self.odds_df = self.odds_df.set_index('Player ID')


    def set_markets(self, odds_df=None, player_props=None, remove_existing=True):

        # Handle optional inputs
        if odds_df is None:
            odds_df = pd.DataFrame()
        if player_props is None:
            player_props = stats_config.default_stat_list

        # Set default list of stats to include (formatted correctly for The Odds)
        stat_names_dict = stats_config.labels_df_to_odds
        self.markets = [stat_names_dict[stat] for stat in list(set(player_props) & set(stat_names_dict))]

        # Remove any markets already in odds_df (assumes no updates are needed to that data)
        if remove_existing and 'Player Prop Stat' in odds_df.columns:
            existing_markets = odds_df['Player Prop Stat'].unique()
            self.markets = [market for market in self.markets if market not in existing_markets]

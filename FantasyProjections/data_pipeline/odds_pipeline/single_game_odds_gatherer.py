"""Creates and exports class to be used in NFL player statistics data collection.

    Classes:
        SingleGameOddsGatherer : Collects all player props ("odds") for a single NFL game. Automatically processes data upon initialization.
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
from data_pipeline.utils import team_abbreviations as team_abbrs

# Set up logger
logger = logging.getLogger('log')

class SingleGameOddsGatherer(SingleGameDataWorker):
    """Collects all player props ("odds") for a single NFL game. Automatically processes data upon initialization.

        Subclass of SingleGameDataWorker.

        Args:
            seasonal_data (SeasonalDataCollector): "Parent" object containing data relating to the NFL season.
                Not stored as an object attribute.
            event_id (str): Game ID for specific game ("event"), as used by The Odds API. Format is random characters.
            game_id (str): Game ID for specific game, as used by nfl-verse. Format is "{year}_{week}_{awayteam}_{home_team}", ex: "2021_01_ARI_TEN"
        Keyword Arguments:
            player_props (list, optional): List of stats to gather gambling odds for. Defaults to None.
            odds_file (str, optional): File path to csv containing all previously-obtained player prop odds.

        Additional Attributes Created during Initialization:
            year (int): Year of game being processed
            week (int): Week in NFL season of game being processed
            pbp_df (pandas.DataFrame): Play-by-play data for all plays in the game, taken from nfl-verse.
            roster_df (pandas.DataFrame): Players in the game (from both teams) to collect stats for.
            api_key (str): Key to use in requests to The Odds API.
            all_game_times (pandas.Series): Maps elapsed time in game to UTC ("real world") time.
            odds_df (pandas.DataFrame): All collected (including previously-collected) player prop lines for the game, at potentially varying game times.

        Public Methods:
            gather_historical_odds : Requests historical odds data from The Odds API for the player props of interest at game times of interest.
            reformat_odds_df : Formats data collected from The Odds into consistent style as other data products (common Player IDs, etc.) and structures as one row per player/stat.
            set_markets : Creates list of player props data to gather from The Odds for this game, optionally removing any player props already present in preexisting odds data.
    """

    def __init__(self, seasonal_data, event_id, game_id, **kwargs):
        """Constructor for SingleGamePbpParser object.

            Args:
                seasonal_data (SeasonalDataCollector): "Parent" object containing data relating to the NFL season.
                    Not stored as an object attribute.
                event_id (str): Game ID for specific game ("event"), as used by The Odds API. Format is random characters.
                game_id (str): Game ID for specific game, as used by nfl-verse. Format is "{year}_{week}_{awayteam}_{home_team}", ex: "2021_01_ARI_TEN"
            Keyword Arguments:
                player_props (list, optional): List of stats to gather gambling odds for. Defaults to None.
                odds_file (str, optional): File path to csv containing all previously-obtained player prop odds. Defaults to None.

            Additional Attributes Created during Initialization:
                year (int): Year of game being processed
                week (int): Week in NFL season of game being processed
                pbp_df (pandas.DataFrame): Play-by-play data for all plays in the game, taken from nfl-verse.
                roster_df (pandas.DataFrame): Players in the game (from both teams) to collect stats for.
                api_key (str): Key to use in requests to The Odds API.
                all_game_times (pandas.Series): Maps elapsed time in game to UTC ("real world") time.
                odds_df (pandas.DataFrame): All collected (including previously-collected) player prop lines for the game, at potentially varying game times.
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
            odds_df = pd.read_csv(odds_file, index_col=0)
            odds_df = odds_df[odds_df['Game ID'] == self.game_id]
        except (FileNotFoundError, ValueError):
            odds_df = pd.DataFrame()

        # Determine the list of odds to collect
        self.set_markets(odds_df, player_props)

        # Collect additional odds from API
        if len(self.markets) > 0:
            self.odds_df = self.gather_historical_odds(odds_df=odds_df, game_times=[0,30])
        else:
            self.odds_df = self.reformat_odds_df(odds_df)


    def gather_historical_odds(self, odds_df=None, game_times=None):
        """Requests historical odds data from The Odds API for the player props of interest at game times of interest.

            Args:
                odds_df (pandas.DataFrame, optional): Preexisting player prop data. Any data collected in this function will be appended. Defaults to None.
                game_times (list, optional): List of game elapsed times to collect odds data for (e.g. pregame odds at time=0, halftime odds at time=30). Defaults to [0] (pregame).

            Returns:
                pandas.DataFrame: All collected (including previously-collected) player prop lines for the game, at potentially varying game times.
        """

        # Default game time is just 0 (pre-game odds)
        if game_times is None:
            game_times = [0]

        # Optional input: pre-existing dataframe to append to
        if odds_df is None:
            odds_df = pd.DataFrame()

        # Make a separate API request for each desired game time
        for game_time in game_times:
            # Determine time of the most recent play in order to request the odds from that time
            pbp_index = find_prev_time_index(game_time, self.all_game_times)
            time = datetime.strftime(dateparse.parse(self.all_game_times.iloc[pbp_index]), DATE_FMT)

            # Make the request to The Odds for data on the selected markets at the correct time
            response = requests.get(f'https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/events/{self.event_id}/odds', params={
                'api_key': self.api_key,
                'regions': 'us',
                'markets': ','.join(self.markets),
                'date': time,
                }, timeout=10)
            # Track usage of API key
            log_api_usage(response)

            # Process API Response to obtain market data. Exit gracefully if response does not contain "data" or data does not contain "markets".
            try:
                event_data = json.loads(response.text)['data']
            except KeyError:
                logger.warning(f'Key Error encountered in API request. Aborting game {self.game_id} early.')
                return odds_df
            try:
                market_data = pd.DataFrame(event_data['bookmakers']).set_index('key').loc[BOOKMAKER, 'markets']
            except KeyError:
                logger.warning(f'Key Error encountered: {BOOKMAKER} missing. Aborting game {self.game_id} early.')
                return odds_df

            for market in market_data:
                # Process Odds data into a DataFrame
                df = pd.DataFrame(market['outcomes']
                                ).rename(columns={'description': 'Player Name', 'name': 'Line'})
                # Add Player Prop Stat and UTC time from market data
                df['Player Prop Stat'] = team_abbrs.invert(stats_config.labels_df_to_odds)[market['key']]
                df['UTC Time'] = market['last_update']
                # Format dataframe for output
                df = self.reformat_odds_df(df)
                # Add to running list of all odds
                odds_df = pd.concat((odds_df, df))

            return odds_df


    def reformat_odds_df(self, df):
        """Formats data collected from The Odds into consistent style as other data products (common Player IDs, etc.) and structures as one row per player/stat.

            Args:
                df (pandas.DataFrame): Data collected from The Odds with some minor formatting/added info already applied.

            Returns:
                pandas.DataFrame: Data in consistent structure/style as other data products.
        """

        # Modify into the desired DataFrame structure
        if 'Line' in df.columns:
            df = df.set_index(['Player Name', 'UTC Time'])
            for line in ['Over','Under']:
                df[f'{line} Point'] = df.reset_index().set_index(['Player Name','UTC Time','Line']).xs(line,level=2)['point']
                df[f'{line} Price'] = df.reset_index().set_index(['Player Name','UTC Time','Line']).xs(line,level=2)['price']
            df = df.reset_index().drop(columns=['Line','price','point']).drop_duplicates(keep='first')

        # Add Year, Week, and Game ID
        df['Year'] = self.year
        df['Week'] = self.week
        df['Game ID'] = self.game_id

        # Add Player ID based on roster
        roster_inds = df['Player Name'].apply(find_matching_name_ind, args=(self.roster_df['Player Name'].unique(),))
        df['Player ID'] = roster_inds.apply(lambda x: self.roster_df.reset_index().iloc[int(x)][PRIMARY_PLAYER_ID] if not np.isnan(x) else np.nan)

        elapsed_time_inds = df['UTC Time'].apply(
            find_prev_time_index, args=(self.all_game_times,))
        df['Elapsed Time'] = self.all_game_times.index[elapsed_time_inds].to_list()

        df = df.set_index('Player ID')

        return df

    def set_markets(self, odds_df=None, player_props=None, remove_existing=True):
        """Creates list of player props data to gather from The Odds for this game, optionally removing any player props already present in preexisting odds data.

            Args:
                odds_df (pandas.DataFrame, optional): Pre-collected player prop lines, used to prevent duplication of data. Defaults to None.
                player_props (list, optional): List of player props to include in odds data. Defaults to None.
                remove_existing (bool, optional): Whether to remove player props already present in preexisting odds data. Defaults to True.

            Properties Modified:
                markets (list): player props to request from The Odds, in the correct format used by The Odds.

        """

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
            existing_markets = [stat_names_dict[stat] for stat in odds_df['Player Prop Stat'].unique()]
            self.markets = [market for market in self.markets if market not in existing_markets]

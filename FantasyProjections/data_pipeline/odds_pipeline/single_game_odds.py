"""Creates and exports class to be used in NFL player statistics data collection.

    Classes:
        SingleGamePbpParser : Collects all player stats for a single NFL game. Automatically processes data upon initialization.
"""
from datetime import datetime
import json
import dateutil.parser as dateparse
import requests
import numpy as np
import pandas as pd
from config import stats_config
from data_pipeline.odds_pipeline.odds_api_helper_functions import SPORT_KEY, DATE_FMT, BOOKMAKER, log_api_usage
from data_pipeline.utils.data_helper_functions import single_game_play_by_play
from data_pipeline.utils.name_matching import find_matching_name_ind
from data_pipeline.utils.time_helper_functions import find_prev_time_index
from config.player_id_config import PRIMARY_PLAYER_ID

class SingleGameOdds():
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

    def __init__(self, seasonal_data, event_id, game_id, odds_file=None):
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

        # Basic info
        self.year = seasonal_data.year
        self.week = int(game_id.split('_')[1])
        self.api_key = seasonal_data.api_key
        self.event_id = event_id
        self.game_id = game_id
        # Play-by-Play info
        self.pbp_df = single_game_play_by_play(seasonal_data.pbp_df, game_id)
        # Helpful: maps game time to UTC time
        self.game_times = self.pbp_df.loc[np.logical_not(self.pbp_df['time_of_day'].isna()), 'time_of_day']
        # Roster info for this game from the two teams' seasonal data
        self.roster_df = seasonal_data.all_rosters_df.loc[
            seasonal_data.all_rosters_df.index.intersection(
                [(team, self.week) for team in self.pbp_df[['home_team','away_team']].iloc[0].to_list()])
            ].reset_index().set_index(PRIMARY_PLAYER_ID)
        # Existing set of odds for this game
        try:
            self.odds_df = pd.read_csv(odds_file)
            self.odds_df = self.odds_df[self.odds_df['Game ID'] == self.game_id]
        except (FileNotFoundError, ValueError):
            self.odds_df = pd.DataFrame()

        self.set_markets()

        if len(self.markets) > 0:
            self.get_historical_odds(game_times=[0,30])

        self.reformat_odds_df()


    def set_markets(self, remove_existing=True):
        # Set default list of stats to include (formatted correctly for The Odds)
        stat_names_dict = stats_config.labels_df_to_odds
        stats_to_include = stats_config.default_stat_list
        stats_to_include = ['Rec Yds']
        self.markets = [stat_names_dict[stat] for stat in list(set(stats_to_include) & set(stat_names_dict))]

        # Remove any markets already in odds_df (assumes no updates are needed to that data)
        if remove_existing and 'Player Prop Stat' in self.odds_df.columns:
            existing_markets = self.odds_df['Player Prop Stat'].unique()
            self.markets = [market for market in self.markets if market not in existing_markets]

    def get_historical_odds(self, game_times=None):
        # Default game time is just 0 (pre-game odds)
        if game_times is None:
            game_times = [0]

        for game_time in game_times:
            pbp_ind = find_prev_time_index(game_time, self.game_times)
            time = datetime.strftime(dateparse.parse(self.game_times.iloc[pbp_ind]), DATE_FMT)

            response = requests.get(f'https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/events/{self.event_id}/odds', params={
                'api_key': self.api_key,
                'regions': 'us',
                'markets': ','.join(self.markets),
                'date': time,
                }, timeout=10)
            log_api_usage(response)
            event_data = json.loads(response.text)['data']
            market_data = pd.DataFrame(event_data['bookmakers']).set_index('key').loc[BOOKMAKER, 'markets']
            for market in market_data:
                # Process Odds data into a DataFrame
                df = pd.DataFrame(market['outcomes']
                                ).rename(columns={'description': 'Player Name', 'name': 'Line'})
                # Add Player Prop Stat and UTC time from market data
                df['Player Prop Stat'] = market['key']
                df['UTC Time'] = market['last_update']
                # Add to running list of all odds
                self.odds_df = pd.concat((self.odds_df, df))

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
            find_prev_time_index, args=(self.game_times,))
        self.odds_df['Elapsed Time'] = self.game_times.index[elapsed_time_inds].to_list()

        self.odds_df = self.odds_df.set_index('Player ID')

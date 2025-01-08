"""Creates and exports class to be used in NFL player statistics data collection.

    Classes:
        SingleGamePbpParser : Collects all player stats for a single NFL game. Automatically processes data upon initialization.
"""

from config.player_id_config import PRIMARY_PLAYER_ID
from data_pipeline.utils.data_helper_functions import calc_game_time_elapsed

class SingleGameDataWorker():
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

    def __init__(self, seasonal_data, game_id):
        """Constructor for SingleGamePbpParser object.

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
        """

        # Basic info
        self.game_id = game_id
        self.year = seasonal_data.year
        self.week = int(self.game_id.split('_')[1])

        # Filter seasonal play-by-play database to just the plays in this game
        self.pbp_df = self.single_game_play_by_play(seasonal_data.pbp_df)

        # Roster info for this game from the two teams' seasonal data
        self.roster_df = seasonal_data.all_rosters_df.loc[
            seasonal_data.all_rosters_df.index.intersection(
                [(team, self.week) for team in self.pbp_df[['home_team','away_team']].iloc[0].to_list()])
            ].reset_index().set_index(PRIMARY_PLAYER_ID)


    # PUBLIC METHODS

    def single_game_play_by_play(self, pbp_df):
        """Filters and cleans play-by-play data for a specific game; keeps all plays from that game and sorts by increasing elapsed game time.

            Args:
                pbp_df (pandas.DataFrame): Play-by-play data for all plays in an NFL season, taken from nfl-verse. 

            Returns:
                pandas.DataFrame: pbp_df input, filtered to only the plays with matching game_id. Elapsed Time is added as a column and set as the index.
        """

        # Make a copy of the input play-by-play df
        pbp_df = pbp_df.copy()

        # Filter to only the game of interest (using game_id)
        pbp_df = pbp_df[pbp_df['game_id'] == self.game_id]

        # Elapsed time
        pbp_df.loc[:,'Elapsed Time'] = pbp_df.apply(calc_game_time_elapsed, axis=1)

        # Sort by ascending elapsed time
        pbp_df = pbp_df.set_index('Elapsed Time').sort_index(ascending=True)

        return pbp_df

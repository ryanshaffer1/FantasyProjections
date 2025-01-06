"""Creates and exports class to be used in NFL player statistics data collection.

    Classes:
        SingleGamePbpParser : Collects all player stats for a single NFL game. Automatically processes data upon initialization.
"""
import numpy as np
import pandas as pd
from config import stats_config
from config.player_id_config import PRIMARY_PLAYER_ID, PLAYER_IDS
from misc.stat_utils import stats_to_fantasy_points
from data_pipeline.data_helper_functions import single_game_play_by_play, subsample_game_time

class SingleGamePbpParser():
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

    def __init__(self, seasonal_data, game_id, **kwargs):
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

        # Optional keyword arguments
        game_times = kwargs.get('game_times','all')

        # Basic info
        self.year = seasonal_data.year
        self.week = int(game_id.split('_')[1])
        # Game info
        self.game_info = seasonal_data.all_game_info_df.loc[game_id]
        # Roster info for this game from the two teams' seasonal data
        self.roster_df = seasonal_data.all_rosters_df.loc[
            seasonal_data.all_rosters_df.index.intersection(
                [(team, self.week) for team in self.game_info['Team Abbrev'].unique()])
            ].reset_index().set_index(PRIMARY_PLAYER_ID)

        # Parse Play-by-Play to generate statistics dataframes
        self.pbp_df = single_game_play_by_play(seasonal_data.pbp_df, game_id)
        self.midgame_df, self.final_stats_df = self.parse_play_by_play(game_times)


    def parse_play_by_play(self, game_times='all'):
        """Calculates midgame and final statistics for all players in a game, using play-by-play data describing passes, rushes, etc.

            Args:
                pbp_df (pandas.DataFrame): Play-by-play data obtained from nfl-verse
                roster_df (pandas.DataFrame): List of players on a team to track stats for.
                game_info (pandas.Series): Information related to the game, including home team, away team, etc.
                game_times (list | str, optional): Times in the game to output midgame stats for. May be a list of elapsed times in minutes, 
                    in which case the soonest play-by-play time after that elapsed time will be used as the stats at that time. Defaults to 'all' (every play is included).

            Returns:
                pandas.DataFrame: Stats accrued over the course of the game (at time intervals defined by input game_times) for all players in the input roster.
                pandas.DataFrame: Stats at the end of the game for all players in the input roster.
        """

        # Re-format optional inputs
        if game_times != 'all':
            game_times = np.array(game_times)

        # Compute stats for each player on the team in this game
        list_of_player_dfs = self.roster_df.reset_index().apply(self.__parse_player_stats,
                            args=(game_times,),
                            axis=1)
        final_stats_df = pd.concat([row.iloc[-1] for row in list_of_player_dfs],axis=1)
        midgame_stats_df = pd.concat(list_of_player_dfs.tolist())

        # Reformat box score df for output
        final_stats_df = final_stats_df.transpose()

        # Add fantasy points (not sure if this is needed)
        final_stats_df = stats_to_fantasy_points(final_stats_df)

        # Add some seasonal/weekly context to the dataframes
        for df in [midgame_stats_df, final_stats_df]:
            df[['Year', 'Week']] = [self.year, self.week]
            df['Team'] = self.roster_df.loc[(df[PRIMARY_PLAYER_ID],'Team')].tolist()
            df['Opponent'] = self.game_info.set_index('Team Abbrev').loc[df['Team'],'Opp Abbrev'].to_list()
            df['Site'] = self.game_info.set_index('Team Abbrev').loc[df['Team'],'Site'].to_list()
            df[['Team Wins','Team Losses','Team Ties']] = self.game_info.set_index('Team Abbrev').loc[
                df['Team']][['Team Wins', 'Team Losses', 'Team Ties']].to_numpy()
            df[['Opp Wins','Opp Losses','Opp Ties']] = self.game_info.set_index('Team Abbrev').loc[
                df['Opponent']][['Team Wins', 'Team Losses', 'Team Ties']].to_numpy()

        final_stats_df = final_stats_df.set_index(PRIMARY_PLAYER_ID)

        # Assign outputs
        return midgame_stats_df, final_stats_df


    # PRIVATE METHODS

    def __parse_player_stats(self, player_info, game_times):
        """Determines the mid-game stats for one player throughout the game.

            Args:
                player_info (pandas.Series): Roster information for one player (number, ID, position, etc.).
                game_times (list | str): Times in the game to output midgame stats for. May be a list of elapsed times in minutes, 
                    in which case the soonest play-by-play time after that elapsed time will be used as the stats at that time. If a string is passed, no filtering occurs.

            Returns:
                pandas.DataFrame: Player's accumulated stat line at each time in the game. May have an additional (redundant) row for the final stat line.
        """

        # Team sites
        game_site = self.game_info.set_index('Team Abbrev').loc[player_info['Team'],'Site'].lower()
        opp_game_site = ['home', 'away'][(['home', 'away'].index(game_site) + 1) % 2]

        # Set up dataframe covering player's contributions each play
        player_stats_df = pd.DataFrame()  # Output array
        # Game time elapsed
        player_stats_df['Elapsed Time'] = self.pbp_df.reset_index()['Elapsed Time']
        player_stats_df = player_stats_df.set_index('Elapsed Time')
        # Possession
        player_stats_df['Possession'] = self.pbp_df['posteam'] == player_info['Team']
        # Field Position
        player_stats_df['Field Position'] = self.pbp_df.apply(lambda x: x['yardline_100'] if (
            x['posteam'] == player_info['Team']) else 100 - x['yardline_100'], axis=1)
        # Score
        player_stats_df['Team Score'] = self.pbp_df[f'total_{game_site}_score']
        player_stats_df['Opp Score'] = self.pbp_df[f'total_{opp_game_site}_score']

        # Assign stats to player (e.g. passing yards, rushing TDs, etc.)
        player_stats_df = self.__assign_stats_from_plays(player_stats_df, player_info, team_abbrev=player_info['Team'])

        # Clean up the player dataframe
        player_stats_df = player_stats_df[['Team Score',
                                    'Opp Score',
                                    'Possession',
                                    'Field Position'] + stats_config.default_stat_list]

        # Perform cumulative sum on all columns besides the list below
        for col in set(player_stats_df.columns) ^ set(
            ['Team Score', 'Opp Score', 'Possession', 'Field Position']):
            player_stats_df[col] = player_stats_df[col].cumsum()

        # Add player identifying info
        player_stats_df.loc[:,['Player Name','Position','Age']] = player_info[['Player Name','Position','Age']].to_list()
        for id_type in PLAYER_IDS:
            player_stats_df[id_type] = player_info[id_type]

        # Trim to just the game times of interest
        player_stats_df = subsample_game_time(player_stats_df, game_times)

        return player_stats_df


    def __assign_stats_from_plays(self, player_stat_df, player_info, team_abbrev):
        # NOTE: 2-pt conversions should be excluded from stats
        """Parses play-by-play information to translate the result of plays into stats for the player currently being processed.

            Note that these stats are not cumulative, they only count stats for each play (later code outside this function accumulates the stats over the game).

            Args:
                player_stat_df (pandas.DataFrame): DataFrame containing some game info: elapsed time, as well as other relevant context (ex. Field Position).
                player_info (pandas.Series): Player information, including name, Player ID, position, etc.
                team_abbrev (str): Abbreviation for the team used in the play-by-play data

            Returns:
                pandas.DataFrame: Input player_stat_df DataFrame, with additional columns tracking whether the current player earned stats in each play.                 
        """
        player_id = player_info[PRIMARY_PLAYER_ID]

        # Passing Stats
        # Attempts (checks that selected player made the pass attempt)
        player_stat_df['Pass Att'] = self.pbp_df.apply(
            lambda x: (
                x['passer_player_id'] == player_id) & (
                x['sack'] == 0), axis=1)

        # Completions (checks that no interception was thrown and was not
        # marked incomplete)
        player_stat_df['Pass Cmp'] = self.pbp_df.apply(
            lambda x: (
                (x['complete_pass'] == 1) & (
                    x['passer_player_id'] == player_id)),
            axis=1)
        # Passing Yards
        player_stat_df['temp'] = self.pbp_df['passing_yards']
        player_stat_df['Pass Yds'] = player_stat_df.apply(
            lambda x: x['temp'] if x['Pass Cmp'] else 0, axis=1)
        # Passing Touchdowns
        player_stat_df['temp'] = self.pbp_df['pass_touchdown']
        player_stat_df['Pass TD'] = player_stat_df.apply(
            lambda x: x['Pass Cmp'] and (x['temp'] == 1), axis=1)
        # Interceptions
        player_stat_df['temp'] = self.pbp_df['interception']
        player_stat_df['Int'] = player_stat_df.apply(
            lambda x: x['Pass Att'] and (
                x['temp'] == 1), axis=1)

        # Rushing Stats
        # Rush Attempts (checks that the selected player made the rush attempt)
        player_stat_df['Rush Att'] = self.pbp_df.apply(lambda x: (
            x['rusher_player_id'] == player_id), axis=1)
        # Rushing Yards
        player_stat_df['Rush Yds'] = self.pbp_df.apply(lambda x: x['rushing_yards'] if (
            x['rusher_player_id'] == player_id) else 0, axis=1)
        # Rushing Touchdowns
        player_stat_df['Rush TD'] = self.pbp_df.apply(
            lambda x: x['td_player_id'] == player_id and (
                x['rush_touchdown'] == 1), axis=1)

        # Receiving Stats
        # Receptions (checks that the selected player made the catch
        player_stat_df['Rec'] = self.pbp_df.apply(
            lambda x: (
                x['complete_pass'] == 1) & (
                x['receiver_player_id'] == player_id),
            axis=1)
        # Receiving Yards
        player_stat_df['Rec Yds'] = self.pbp_df.apply(
            lambda x: x['receiving_yards'] if (
                x['complete_pass'] == 1) & (
                x['receiver_player_id'] == player_id) else 0, axis=1)
        # Receiving Touchdowns
        player_stat_df['Rec TD'] = self.pbp_df.apply(
            lambda x: x['td_player_id'] == player_id and not x['passer_player_id'] == player_id and (
                x['pass_touchdown'] == 1), axis=1)

        # Misc. Stats
        # Add lateral receiving yards/rushing yards
        player_stat_df['Rec Yds'] = player_stat_df['Rec Yds'] + self.pbp_df.apply(
            lambda x: x['lateral_receiving_yards'] if x['lateral_receiver_player_id'] == player_id else 0, axis=1)
        player_stat_df['Rush Yds'] = player_stat_df['Rush Yds'] + self.pbp_df.apply(
            lambda x: x['lateral_rushing_yards'] if x['lateral_rusher_player_id'] == player_id else 0, axis=1)
        # Fumbles Lost
        player_stat_df['Fmb'] = self.pbp_df.apply(
            lambda x: (
                x['fumble_lost'] == 1) & (
                ((x['fumbled_1_player_id'] == player_id) & (
                    x['fumble_recovery_1_team'] != team_abbrev)) | (
                    (x['fumbled_2_player_id'] == player_id) & (
                        x['fumble_recovery_2_team'] != team_abbrev))),
            axis=1)

        # Replace all nan's with 0
        player_stat_df = player_stat_df.fillna(value=0)

        return player_stat_df


import numpy as np
import pandas as pd
from data_pipeline import team_abbreviations
from misc.stat_utils import stats_to_fantasy_points
from .data_helper_functions import calc_game_time_elapsed

class TeamWeeklyData():
    def __init__(self, seasonal_data, team_name, week, **kwargs):
        """_summary_

            Args:
                seasonal_data (_type_): _description_
                team_name (_type_): _description_
                week (_type_): _description_
                
                pbp_df (pandas.DataFrame): Play-by-play information for every play in the game currently being processed.
        """

        game_times = kwargs.get('game_times','all')

        self.team_name = team_name
        self.week = week
        self.year = seasonal_data.year
        # Game/Roster info for this week from the team's seasonal data
        self.game_info = seasonal_data.all_game_info_df.loc[(self.team_name, self.year, self.week)]
        self.roster_df = seasonal_data.all_rosters_df.loc[(team_abbreviations.pbp_abbrevs[self.team_name], self.week)].set_index("Name")
        self.game_id = self.gen_game_id()

        # Parse Play-by-Play to generate statistics dataframes
        self.pbp_df = self.gen_game_play_by_play(seasonal_data.pbp_df)
        self.midgame_df, self.final_stats_df = self.parse_play_by_play(game_times)

    def gen_game_id(self):
        game_id = str(self.year) + '_' + str(self.week).zfill(2) + '_' + \
            self.game_info['Away Team Abbrev'] + \
            '_' + self.game_info['Home Team Abbrev']
        return game_id

    def gen_game_play_by_play(self, pbp_df):

        # Make a copy of the input play-by-play df
        pbp_df = pbp_df.copy()

        # Filter to only the game of interest (using game_id)
        pbp_df = pbp_df[pbp_df['game_id'] == self.game_id]

        # Elapsed time
        pbp_df.loc[:,'Elapsed Time'] = pbp_df.apply(calc_game_time_elapsed, axis=1)

        # Sort by ascending elapsed time
        pbp_df = pbp_df.set_index('Elapsed Time').sort_index(ascending=True)

        return pbp_df

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
        # Basic info
        year = self.year
        week = self.week

        # Compute stats for each player on the team in this game
        list_of_player_dfs = self.roster_df.apply(self.parse_player_stats,
                            args=(game_times,),
                            axis=1)
        final_stats_df = pd.concat([row.iloc[-1] for row in list_of_player_dfs],axis=1)
        midgame_stats_df = pd.concat(list_of_player_dfs.tolist())

        # Reformat box score df for output
        final_stats_df = final_stats_df.transpose().set_index('Player')

        # Add fantasy points (not sure if this is needed)
        final_stats_df = stats_to_fantasy_points(final_stats_df)

        # Add some seasonal/weekly context to the dataframes
        for df in [midgame_stats_df, final_stats_df]:
            df[['Year', 'Week']] = [year, week]
            df['Team'] = self.game_info['Team Abbrev']
            df['Opponent'] = self.game_info['Opp Abbrev']
            df['Site'] = self.game_info['Site']
            df[['Team Wins', 'Team Losses', 'Team Ties']] = self.game_info[[
                'Team Wins', 'Team Losses', 'Team Ties']].to_list()
            df[['Opp Wins', 'Opp Losses', 'Opp Ties']] = self.game_info[[
                'Opp Wins', 'Opp Losses', 'Opp Ties']].to_list()

        # Assign outputs
        return midgame_stats_df, final_stats_df


    def parse_player_stats(self, roster_df_row, game_times):
        """Determines the mid-game stats for one player throughout the game.

            Args:
                roster_df_row (pandas.Series): Roster information for one player (number, ID, position, etc.).
                game_info (pandas.Series): Information related to the game, including home team, away team, etc.
                game_times (list | str): Times in the game to output midgame stats for. May be a list of elapsed times in minutes, 
                    in which case the soonest play-by-play time after that elapsed time will be used as the stats at that time. If a string is passed, no filtering occurs.

            Returns:
                pandas.DataFrame: Player's accumulated stat line at each time in the game. May have an additional (redundant) row for the final stat line.
        """

        # Team sites
        game_site = self.game_info['Site'].lower()
        opp_game_site = ['home', 'away'][(['home', 'away'].index(game_site) + 1) % 2]

        # Set up dataframe covering player's contributions each play
        player_stats_df = pd.DataFrame()  # Output array
        # Game time elapsed
        player_stats_df['Elapsed Time'] = self.pbp_df.reset_index()['Elapsed Time']
        player_stats_df = player_stats_df.set_index('Elapsed Time')
        # Possession
        player_stats_df['Possession'] = self.pbp_df['posteam'] == self.game_info['Team Abbrev']
        # Field Position
        player_stats_df['Field Position'] = self.pbp_df.apply(lambda x: x['yardline_100'] if (
            x['posteam'] == self.game_info['Team Abbrev']) else 100 - x['yardline_100'], axis=1)
        # Score
        player_stats_df['Team Score'] = self.pbp_df[f'total_{game_site}_score']
        player_stats_df['Opp Score'] = self.pbp_df[f'total_{opp_game_site}_score']

        # Assign stats to player (e.g. passing yards, rushing TDs, etc.)
        player_stats_df = self.assign_stats_from_plays(player_stats_df,roster_df_row,team_abbrev=self.game_info['Team Abbrev'])

        # Clean up the player dataframe
        player_stats_df = player_stats_df[['Team Score',
                                    'Opp Score',
                                    'Possession',
                                    'Field Position',
                                    'Pass Att',
                                    'Pass Cmp',
                                    'Pass Yds',
                                    'Pass TD',
                                    'Int',
                                    'Rush Att',
                                    'Rush Yds',
                                    'Rush TD',
                                    'Rec',
                                    'Rec Yds',
                                    'Rec TD',
                                    'Fmb']]

        # Perform cumulative sum on all columns besides the list below
        for col in set(player_stats_df.columns) ^ set(
            ['Team Score', 'Opp Score', 'Possession', 'Field Position']):
            player_stats_df[col] = player_stats_df[col].cumsum()

        # Add player identifying info
        player_stats_df['Player'] = roster_df_row.name
        player_stats_df['Position'] = roster_df_row['Position']
        player_stats_df['Age'] = roster_df_row['Age']

        # Trim to just the game times of interest
        player_stats_df = self.filter_game_time(player_stats_df, game_times)

        return player_stats_df


    def filter_game_time(self, player_stats_df, game_times):
        """Reduces the size of the midgame stats data by sampling the data at discrete game times, rather than keeping the stats after every single play.
        
            Rounds the game time at the start of each play to the nearest value in game_times, then keeps the first instance of each value, as well as the final row (to capture final stats).
            Note that there may be an additional (redundant) row for the final stat line.

            Args:
                player_stats_df (pandas.DataFrame): Player's accumulated stat line after every single play in the game.
                game_times (list | str): List of game times to include in the output. If string is input (such as "all"), no filtering occurs.

            Returns:
                pandas.DataFrame: Player's accumulated stat line at each designated game time. May have an additional (redundant) row for the final stat line.
        """

        if not isinstance(game_times, str):
            player_stats_df['Rounded Time'] = player_stats_df.index.map(
                lambda x: game_times[abs(game_times - float(x)).argmin()])
            player_stats_df = pd.concat((player_stats_df.iloc[1:].drop_duplicates(
                subset='Rounded Time', keep='first'),player_stats_df.iloc[-1].to_frame().T))
            player_stats_df = player_stats_df.reset_index(drop=True).rename(
                columns={'Rounded Time': 'Elapsed Time'}).set_index('Elapsed Time')

        return player_stats_df


    def check_stat_for_player(self, game_df, player_df, comparisons, bools=None, operator='and'):
        """Not currently used, probably going to be re-written.

            Args:
                game_df (_type_): _description_
                player_df (_type_): _description_
                comparisons (_type_): _description_
                bools (_type_, optional): _description_. Defaults to None.
                operator (str, optional): _description_. Defaults to 'and'.

            Returns:
                _type_: _description_
        """

        if not bools:
            bools = [True]*len(comparisons)

        term_comps = []
        for ((x,y),boolean) in zip(comparisons,bools):
            if isinstance(y,str):
                term_comps.append((game_df[x] == player_df[y])==boolean)
            else:
                term_comps.append((game_df[x] == y)==boolean)
        if operator == 'and':
            full_comp = all(term_comps)
        elif operator == 'or':
            full_comp = any(term_comps)
        else:
            full_comp = False

        return full_comp


    def assign_stats_from_plays(self, player_stat_df, player_df, team_abbrev):
        """Parses play-by-play information to translate the result of plays into stats for the player currently being processed.

            Note that these stats are not cumulative, they only count stats for each play (later code outside this function accumulates the stats over the game).

            Args:
                player_stat_df (pandas.DataFrame): DataFrame containing some game info: elapsed time, as well as other relevant context (ex. Field Position).
                player_df (pandas.Series): Player information, including name, Player ID, position, etc.
                team_abbrev (str): Abbreviation for the team used in the play-by-play data

            Returns:
                pandas.DataFrame: Input player_stat_df DataFrame, with additional columns tracking whether the current player earned stats in each play.                 
        """

        # Passing Stats
        # Attempts (checks that selected player made the pass attempt)
        player_stat_df['Pass Att'] = self.pbp_df.apply(
            lambda x: (
                x['passer_player_id'] == player_df['Player ID']) & (
                x['sack'] == 0), axis=1)

        # Completions (checks that no interception was thrown and was not
        # marked incomplete)
        player_stat_df['Pass Cmp'] = self.pbp_df.apply(
            lambda x: (
                (x['complete_pass'] == 1) & (
                    x['passer_player_id'] == player_df['Player ID'])),
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
            x['rusher_player_id'] == player_df['Player ID']), axis=1)
        # Rushing Yards
        player_stat_df['Rush Yds'] = self.pbp_df.apply(lambda x: x['rushing_yards'] if (
            x['rusher_player_id'] == player_df['Player ID']) else 0, axis=1)
        # Rushing Touchdowns
        player_stat_df['Rush TD'] = self.pbp_df.apply(
            lambda x: x['td_player_id'] == player_df['Player ID'] and (
                x['rush_touchdown'] == 1), axis=1)

        # Receiving Stats
        # Receptions (checks that the selected player made the catch
        player_stat_df['Rec'] = self.pbp_df.apply(
            lambda x: (
                x['complete_pass'] == 1) & (
                x['receiver_player_id'] == player_df['Player ID']),
            axis=1)
        # Receiving Yards
        player_stat_df['Rec Yds'] = self.pbp_df.apply(
            lambda x: x['receiving_yards'] if (
                x['complete_pass'] == 1) & (
                x['receiver_player_id'] == player_df['Player ID']) else 0, axis=1)
        # Receiving Touchdowns
        player_stat_df['Rec TD'] = self.pbp_df.apply(
            lambda x: x['td_player_id'] == player_df['Player ID'] and not x['passer_player_id'] == player_df['Player ID'] and (
                x['pass_touchdown'] == 1), axis=1)

        # Misc. Stats
        # Add lateral receiving yards/rushing yards
        player_stat_df['Rec Yds'] = player_stat_df['Rec Yds'] + self.pbp_df.apply(
            lambda x: x['lateral_receiving_yards'] if x['lateral_receiver_player_id'] == player_df['Player ID'] else 0, axis=1)
        player_stat_df['Rush Yds'] = player_stat_df['Rush Yds'] + self.pbp_df.apply(
            lambda x: x['lateral_rushing_yards'] if x['lateral_rusher_player_id'] == player_df['Player ID'] else 0, axis=1)
        # Fumbles Lost
        player_stat_df['Fmb'] = self.pbp_df.apply(
            lambda x: (
                x['fumble_lost'] == 1) & (
                ((x['fumbled_1_player_id'] == player_df['Player ID']) & (
                    x['fumble_recovery_1_team'] != team_abbrev)) | (
                    (x['fumbled_2_player_id'] == player_df['Player ID']) & (
                        x['fumble_recovery_2_team'] != team_abbrev))),
            axis=1)

        # Replace all nan's with 0
        player_stat_df = player_stat_df.fillna(value=0)

        return player_stat_df

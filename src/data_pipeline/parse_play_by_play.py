import pandas as pd
import numpy as np
from misc.nn_helper_functions import stats_to_fantasy_points
from .data_helper_functions import calc_game_time_elapsed

def parse_play_by_play(pbp_df, roster_df,
                    game_info, game_times='all'):
    # Re-format optional inputs
    if game_times != 'all':
        game_times = np.array(game_times)
    # Basic info from game_info_ser
    year = game_info.name[1]
    week = game_info.name[2]

    # Make a copy of the input play-by-play df
    # (so that view vs copy warnings are avoided)
    pbp_df = pbp_df.copy()

    # Filter the play-by-play dataframe to only the desired game
    pbp_df = filter_by_game_id(pbp_df, game_info, year, week)

    # Elapsed time
    pbp_df['Elapsed Time'] = pbp_df.apply(calc_game_time_elapsed, axis=1)
    # Sort by ascending elapsed time
    pbp_df = pbp_df.set_index('Elapsed Time').sort_index(ascending=True)

    # Compute stats for each player on the team in this game
    list_of_player_dfs = roster_df.apply(parse_player_stats,
                           args=(pbp_df, game_info, game_times),
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
        df['Team'] = game_info['Team Abbrev']
        df['Opponent'] = game_info['Opp Abbrev']
        df['Site'] = game_info['Site']
        df[['Team Wins', 'Team Losses', 'Team Ties']] = game_info[[
            'Team Wins', 'Team Losses', 'Team Ties']].to_list()
        df[['Opp Wins', 'Opp Losses', 'Opp Ties']] = game_info[[
            'Opp Wins', 'Opp Losses', 'Opp Ties']].to_list()

    # Return outputs
    return midgame_stats_df, final_stats_df


def parse_player_stats(roster_df_row, pbp_df, game_info, game_times):
    # Team sites
    game_site = game_info['Site'].lower()
    opp_game_site = ['home', 'away'][(['home', 'away'].index(game_site) + 1) % 2]

    # Set up dataframe covering player's contributions each play
    player_stats_df = pd.DataFrame()  # Output array
    # Game time elapsed
    player_stats_df['Elapsed Time'] = pbp_df.reset_index()['Elapsed Time']
    player_stats_df = player_stats_df.set_index('Elapsed Time')
    # Possession
    player_stats_df['Possession'] = pbp_df['posteam'] == game_info['Team Abbrev']
    # Field Position
    player_stats_df['Field Position'] = pbp_df.apply(lambda x: x['yardline_100'] if (
        x['posteam'] == game_info['Team Abbrev']) else 100 - x['yardline_100'], axis=1)
    # Score
    player_stats_df['Team Score'] = pbp_df[f'total_{game_site}_score']
    player_stats_df['Opp Score'] = pbp_df[f'total_{opp_game_site}_score']

    # Assign stats to player (e.g. passing yards, rushing TDs, etc.)
    player_stats_df = assign_stats_from_plays(player_stats_df,pbp_df,roster_df_row,team_abbrev=game_info['Team Abbrev'])

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
    player_stats_df = filter_game_time(player_stats_df, game_times)

    return player_stats_df


def filter_by_game_id(pbp_df, game_info, year, week):
    # Filter to only the game of interest (using game_id)
    game_id = str(year) + '_' + str(week).zfill(2) + '_' + \
        game_info['Away Team Abbrev'] + \
        '_' + game_info['Home Team Abbrev']
    pbp_df = pbp_df[pbp_df['game_id'] == game_id]

    return pbp_df


def filter_game_time(player_stats_df, game_times):
    if not isinstance(game_times, str):
        player_stats_df['Rounded Time'] = player_stats_df.index.map(
            lambda x: game_times[abs(game_times - float(x)).argmin()])
        player_stats_df = pd.concat((player_stats_df.iloc[1:].drop_duplicates(
            subset='Rounded Time', keep='first'),player_stats_df.iloc[-1].to_frame().T))
        player_stats_df = player_stats_df.reset_index(drop=True).rename(
            columns={'Rounded Time': 'Elapsed Time'}).set_index('Elapsed Time')

    return player_stats_df


def check_stat_for_player(game_df,player_df,comparisons,bools=None,operator='and'):
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


def assign_stats_from_plays(player_stat_df,game_df,player_df,team_abbrev):
    # Passing Stats
    # Attempts (checks that selected player made the pass attempt)
    player_stat_df['Pass Att'] = game_df.apply(
        lambda x: (
            x['passer_player_id'] == player_df['Player ID']) & (
            x['sack'] == 0), axis=1)

    # Completions (checks that no interception was thrown and was not
    # marked incomplete)
    player_stat_df['Pass Cmp'] = game_df.apply(
        lambda x: (
            (x['complete_pass'] == 1) & (
                x['passer_player_id'] == player_df['Player ID'])),
        axis=1)
    # Passing Yards
    player_stat_df['temp'] = game_df['passing_yards']
    player_stat_df['Pass Yds'] = player_stat_df.apply(
        lambda x: x['temp'] if x['Pass Cmp'] else 0, axis=1)
    # Passing Touchdowns
    player_stat_df['temp'] = game_df['pass_touchdown']
    player_stat_df['Pass TD'] = player_stat_df.apply(
        lambda x: x['Pass Cmp'] and (x['temp'] == 1), axis=1)
    # Interceptions
    player_stat_df['temp'] = game_df['interception']
    player_stat_df['Int'] = player_stat_df.apply(
        lambda x: x['Pass Att'] and (
            x['temp'] == 1), axis=1)

    # Rushing Stats
    # Rush Attempts (checks that the selected player made the rush attempt)
    player_stat_df['Rush Att'] = game_df.apply(lambda x: (
        x['rusher_player_id'] == player_df['Player ID']), axis=1)
    # Rushing Yards
    player_stat_df['Rush Yds'] = game_df.apply(lambda x: x['rushing_yards'] if (
        x['rusher_player_id'] == player_df['Player ID']) else 0, axis=1)
    # Rushing Touchdowns
    player_stat_df['Rush TD'] = game_df.apply(
        lambda x: x['td_player_id'] == player_df['Player ID'] and (
            x['rush_touchdown'] == 1), axis=1)

    # Receiving Stats
    # Receptions (checks that the selected player made the catch
    player_stat_df['Rec'] = game_df.apply(
        lambda x: (
            x['complete_pass'] == 1) & (
            x['receiver_player_id'] == player_df['Player ID']),
        axis=1)
    # Receiving Yards
    player_stat_df['Rec Yds'] = game_df.apply(
        lambda x: x['receiving_yards'] if (
            x['complete_pass'] == 1) & (
            x['receiver_player_id'] == player_df['Player ID']) else 0, axis=1)
    # Receiving Touchdowns
    player_stat_df['Rec TD'] = game_df.apply(
        lambda x: x['td_player_id'] == player_df['Player ID'] and not x['passer_player_id'] == player_df['Player ID'] and (
            x['pass_touchdown'] == 1), axis=1)

    # Misc. Stats
    # Add lateral receiving yards/rushing yards
    player_stat_df['Rec Yds'] = player_stat_df['Rec Yds'] + game_df.apply(
        lambda x: x['lateral_receiving_yards'] if x['lateral_receiver_player_id'] == player_df['Player ID'] else 0, axis=1)
    player_stat_df['Rush Yds'] = player_stat_df['Rush Yds'] + game_df.apply(
        lambda x: x['lateral_rushing_yards'] if x['lateral_rusher_player_id'] == player_df['Player ID'] else 0, axis=1)
    # Fumbles Lost
    player_stat_df['Fmb'] = game_df.apply(
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


def assign_stats_from_plays_v2(player_stat_df,game_df,player_df,team_abbrev):
    # Passing Stats
    # Attempts (checks that selected player made the pass attempt)
    player_stat_df['Pass Att'] = game_df.apply(check_stat_for_player,
        args=(player_df,(('passer_player_id','Player ID'),('sack',0))), axis=1)
    # Completions (checks that no interception was thrown and was not
    # marked incomplete)
    player_stat_df['Pass Cmp'] = game_df.apply(check_stat_for_player,
        args=(player_df,(('complete_pass',1),('passer_player_id','Player ID'))), axis=1)
    # Passing Yards
    player_stat_df['temp'] = game_df['passing_yards']
    player_stat_df['Pass Yds'] = player_stat_df.apply(
        lambda x: x['temp'] if x['Pass Cmp'] else 0, axis=1)
    # Passing Touchdowns
    player_stat_df['temp'] = game_df['pass_touchdown']
    player_stat_df['Pass TD'] = player_stat_df.apply(
        lambda x: x['Pass Cmp'] and (x['temp'] == 1), axis=1)
    # Interceptions
    player_stat_df['temp'] = game_df['interception']
    player_stat_df['Int'] = player_stat_df.apply(
        lambda x: x['Pass Att'] and (
            x['temp'] == 1), axis=1)

    # Rushing Stats
    # Rush Attempts (checks that the selected player made the rush attempt)
    player_stat_df['Rush Att'] = game_df.apply(check_stat_for_player,
        args=(player_df,(('rusher_player_id','Player ID'),)), axis=1)
    # Rushing Yards
    # pbp_stat_df['Rush Yds'] = game_df.apply(check_stat_for_player,
    #     args=(player_df,(('rusher_player_id','Player ID'))), axis=1)
    player_stat_df['Rush Yds'] = game_df.apply(lambda x: x['rushing_yards'] if (
        x['rusher_player_id'] == player_df['Player ID']) else 0, axis=1)
    # Rushing Touchdowns
    player_stat_df['Rush TD'] = game_df.apply(check_stat_for_player,
        args=(player_df,(('td_player_id','Player ID'),('rush_touchdown',1))), axis=1)

    # Receiving Stats
    # Receptions (checks that the selected player made the catch
    player_stat_df['Rec'] = game_df.apply(check_stat_for_player,
        args=(player_df,(('complete_pass',1),
                            ('receiver_player_id','Player ID'))), axis=1)
    # Receiving Yards
    player_stat_df['Rec Yds'] = game_df.apply(
        lambda x: x['receiving_yards'] if (
            x['complete_pass'] == 1) & (
            x['receiver_player_id'] == player_df['Player ID']) else 0, axis=1)
    # Receiving Touchdowns
    player_stat_df['Rec TD'] = game_df.apply(check_stat_for_player,
        args=(player_df,(('td_player_id','Player ID'),
            ('passer_player_id','Player ID'),('pass_touchdown',1)),
                [True,False,True]), axis=1)

    # Misc. Stats
    # Add lateral receiving yards/rushing yards
    player_stat_df['Rec Yds'] = player_stat_df['Rec Yds'] + game_df.apply(
        lambda x: x['lateral_receiving_yards'] if x['lateral_receiver_player_id'] == player_df['Player ID'] else 0, axis=1)
    player_stat_df['Rush Yds'] = player_stat_df['Rush Yds'] + game_df.apply(
        lambda x: x['lateral_rushing_yards'] if x['lateral_rusher_player_id'] == player_df['Player ID'] else 0, axis=1)
    # Fumbles Lost
    player_stat_df['Fmb'] = game_df.apply(
        lambda x: (
            x['fumble_lost'] == 1) & (
            ((x['fumbled_1_player_id'] == player_df['Player ID']) & (
                x['fumble_recovery_1_team'] != team_abbrev)) | (
                (x['fumbled_2_player_id'] == player_df['Player ID']) & (
                    x['fumble_recovery_2_team'] != team_abbrev))),
        axis=1)
    # pbp_stat_df['Fmb'] = game_df.apply(check_stat_for_player,
    #     args=(player_df,(('fumble_lost',1),('fumbled_1_player_id','Player ID'))), axis=1)

    # Replace all nan's with 0
    player_stat_df = player_stat_df.fillna(value=0)

    return player_stat_df

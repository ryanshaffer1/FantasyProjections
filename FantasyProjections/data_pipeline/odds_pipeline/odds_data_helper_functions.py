
import numpy as np
import pandas as pd
from config import data_files_config
from config.player_id_config import PRIMARY_PLAYER_ID


def add_player_prop_results(odds_df, stats_df=None, stats_file=None):
    # Optional inputs
    if stats_df is None:
        if stats_file is None:
            stats_file = data_files_config.OUTPUT_FILE_FINAL_STATS
        stats_df = pd.read_csv(stats_file)

    # Configure final stats df
    stats_df = stats_df.copy().set_index([PRIMARY_PLAYER_ID, 'Year', 'Week'])

    for line in ['Over','Under']:
        odds_df[f'{line} Hit'] = odds_df.apply(apply_stat_to_line, args=(stats_df, line), axis=1)

    return odds_df


def apply_stat_to_line(odds_line, stats_df, line):
    try:
        stat = stats_df.loc[(odds_line.name, odds_line['Year'], odds_line['Week']), odds_line['Player Prop Stat']]
    except KeyError:
        return np.nan

    if line == 'Under':
        line_hit = stat < odds_line['Under Point']
    else:
        line_hit = stat > odds_line['Over Point']
    return line_hit

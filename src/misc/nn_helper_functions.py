"""Creates and exports helper functions commonly used when manipulating Fantasy Projections data.

    Functions:
        normalize_stat : Converts statistics from true values (i.e. football stats) to normalized values (scaled between 0 and 1).
        unnormalize_stat : Converts statistics from normalized values (scaled between 0 and 1) back to true values (i.e. actual football stats).
        stats_to_fantasy_points : Calculates Fantasy Points corresponding to an input stat line, based on fantasy scoring rules.
        remove_game_duplicates : Filters evaluation data to only contain one entry per unique game/player.
        gen_random_games : Generates random game/player combinations from input dataset, with no repeating.
        linear_regression : Performs Simple Linear Regression on x_data and y_data to determine line of best fit (slope, intercept) and coefficient of determination (r_squared).
"""

import logging
import numpy as np
import pandas as pd
from config import stats_config

# Set up logger
logger = logging.getLogger('log')

def normalize_stat(col, thresholds=None):
    """Converts statistics from true values (i.e. football stats) to normalized values (scaled between 0 and 1).

        Values are scaled based on notional threshold values set for each statistic, and values outside the thresholds
        are bounded ("clipped") to 0 and 1.

        Args:
            col (pandas.Series): Series of data corresponding to a football stat, with Series name matching a key in the dictionary "thresholds"
            thresholds (dict, optional): Maps stat names (e.g. "Pass Yds") to their min and max expected values, in order to scale statistics to
                lie between 0 and 1. Defaults to dictionary "default_norm_thresholds" defined in configuration files.

        Returns:
            pandas.Series: Series of normalized data where each entry in col is mapped between 0 and 1 according to the bounds in thresholds
    """

    # Optional input
    if not thresholds:
        thresholds = stats_config.default_norm_thresholds

    if col.name in thresholds:
        [lwr, upr] = thresholds[col.name]
        col = (col - lwr) / (upr - lwr)
        col = col.clip(0, 1)
    else:
        logger.warning(f'Warning: {col.name} not explicitly normalized')

    return col


def unnormalize_stat(col, thresholds=None):
    """Converts statistics from normalized values (scaled between 0 and 1) back to true values (i.e. actual football stats).

        Args:
            col (pandas.Series): Series of data corresponding to a normalized stat, with Series name matching a key in the dictionary "thresholds"
            thresholds (dict, optional): Maps stat names (e.g. "Pass Yds") to their min and max expected values, in order to scale statistics to
                lie between 0 and 1. Defaults to dictionary "default_norm_thresholds" defined in configuration files.

        Returns:
            pandas.Series: Series of unnormalized data where each entry in col is scaled up according to the bounds in thresholds
    """

    # Optional input
    if not thresholds:
        thresholds = stats_config.default_norm_thresholds

    if col.name in thresholds:
        # Unfortunately can't undo the clipping from 0 to 1 performed during
        # normalization, so there's a small chance of lost info...
        [lwr, upr] = thresholds[col.name]
        col = col * (upr - lwr) + lwr
    else:
        logger.warning(f'Warning: {col.name} not explicitly normalized')

    return col


def stats_to_fantasy_points(stat_line, stat_indices=None, normalized=False, scoring_weights=None):
    """Calculates Fantasy Points corresponding to an input stat line, based on fantasy scoring rules.

        Args:
            stat_line (pandas.Series | pandas.DataFrame | torch.tensor): Stats to use to calculate fantasy points. 
                If 1D data array, each entry is assumed to correspond to a different statistic (e.g. Pass Yds, Pass TD, etc.).
                If 2D data array, each column is assumed to correspond to a different statistic.
                Data may be normalized or un-normalized, with input "normalized" set accordingly.
            stat_indices (str | list, optional): For data without column headers or row indices, used to determine the 
                order of statistics contained in stat_line. Defaults to None. May be passed as string "default" in order to use default_stat_list.
            normalized (bool, optional): Whether stats in stat_line are already normalized (converted such that all values are between 0 and 1)
                or un-normalized (in standard football stat ranges). Defaults to False.
            scoring_weights (dict, optional): Fantasy points per unit of each statistic (e.g. points per passing yard, points per reception, etc.)
                Defaults to default_scoring_weights.

        Returns:
            pandas.DataFrame: stat_line, un-normalized and with column headers corresponding to stat indices, with an additional entry for
                Fantasy Points calculated based on the fantasy scoring rules.
    """

    # Optional input
    if not scoring_weights:
        # Scoring rules in fantasy format
        scoring_weights = stats_config.default_scoring_weights

    if stat_indices == 'default':
        stat_indices = stats_config.default_stat_list

    if stat_indices:
        stat_line = pd.DataFrame(stat_line)
        if len(stat_line.columns) == 1:
            stat_line = stat_line.transpose()
        stat_line.columns = stat_indices
    if normalized:
        for col in stat_line.columns:
            stat_line[col] = unnormalize_stat(stat_line[col])

    stat_line['Fantasy Points'] = (stat_line[scoring_weights.keys()] * scoring_weights).sum(axis=1)

    return stat_line


def remove_game_duplicates(eval_data):
    """Filters evaluation data to only contain one entry per unique game/player.
    
        Removes all but the first row in id_data for each Player/Year/Week combination. (First row is typically when Elapsed Time = 0).
        Does not modify x_data.

        Args:
            eval_data (StatsDataset): data containing NFL game/player final statistics

        Returns:
            StatsDataset: input StatsDataset, modified to only have one row per unique game/player
    """

    duplicated_rows_eval_data = eval_data.id_data.reset_index().duplicated(subset=[
        'Player', 'Year', 'Week'])
    eval_data.y_data = eval_data.y_data[np.logical_not(duplicated_rows_eval_data)]
    eval_data.id_data = eval_data.id_data.reset_index(
        drop=True).loc[np.logical_not(duplicated_rows_eval_data)].reset_index(drop=True)
    return eval_data


def gen_random_games(id_df, n_random, game_ids=None):
    """Generates random game/player combinations from input dataset, with no repeating.

        Args:
            id_df (pandas.DataFrame): DataFrame containing "Player", "Week", and "Year" as columns
            n_random (int): Number of random game/player combinations to generate
            game_ids (list, optional): List of pre-selected (not random) games/players. Defaults to None. 
                Each element in list must be a dict containing the following keys:
                    - "Player" : value -> str
                    - "Year" : value -> int
                    - "Week" : value -> int

        Returns:
            list: List of games/players, including any pre-selected as well as randomly-selected games/players. 
                Each element in list is a dict containing the following keys:
                    - "Player" : value -> str
                    - "Year" : value -> int
                    - "Week" : value -> int
    """

    if not game_ids:
        game_ids = []
    # (Optionally) Add random players/games to plot
    for _ in range(n_random):
        valid_new_entry = False
        while not valid_new_entry:
            ind = np.random.randint(id_df.shape[0])
            game_dict = {col: id_df.iloc[ind][col]
                        for col in ['Player', 'Week', 'Year']}
            if game_dict not in game_ids:
                game_ids.append(game_dict)
                valid_new_entry = True

    return game_ids


def linear_regression(x_data, y_data):
    """Performs Simple Linear Regression on x_data and y_data to determine line of best fit (slope, intercept) and coefficient of determination (r_squared).
    
        Equations taken from:
        https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/images/Regression_and_Correlation.pdf (p. 6)
        https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/coefficient-of-determination-r-squared.html

        Function has been tested against scipy.stats.linregress and is accurate to within 1e-13.

        Args:
            x_data (list | array | pandas.DataFrame): x-axis data
            y_data (list | array | pandas.DataFrame): y-axis data

        Returns:
            float: slope of regression line
            float: y-intercept of regression line
            float: r_squared (Coefficient of Determination) of regression line against data
    """

    # Basic info about x and y
    x_mean = np.nanmean(x_data)
    y_mean = np.nanmean(y_data)
    n = x_data.shape[0]
    # Calculate slope and intercept for line of best fit
    s_x_y = sum(x_data*y_data)/n - (sum(x_data)*sum(y_data))/n**2
    s_x_sq = sum(x_data**2)/n - (sum(x_data)/n)**2
    slope = s_x_y/s_x_sq
    intercept = y_mean - (slope * x_mean)
    # Calculate r_value
    y_predicted = intercept + slope*x_data
    residuals = y_data - y_predicted
    dists_from_mean = y_data - y_mean
    ssr = sum(residuals**2)
    sst = sum(dists_from_mean**2)
    r_squared = 1 - (ssr/sst)

    return slope, intercept, r_squared

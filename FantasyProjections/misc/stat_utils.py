"""Creates and exports helper functions commonly used when manipulating Fantasy Projections data.

    Functions:
        normalize_stat : Converts statistics from true values (i.e. football stats) to normalized values (scaled between 0 and 1).
        unnormalize_stat : Converts statistics from normalized values (scaled between 0 and 1) back to true values (i.e. actual football stats).
        stats_to_fantasy_points : Calculates Fantasy Points corresponding to an input stat line, based on fantasy scoring rules.
        gen_random_games : Generates random game/player combinations from input dataset, with no repeating.
        linear_regression : Performs Simple Linear Regression on x_data and y_data to determine line of best fit (slope, intercept) and coefficient of determination (r_squared).
"""  # fmt: skip

import logging

import numpy as np
import pandas as pd
from config import stats_config

# Set up logger
logger = logging.getLogger("log")


def normalize_stat(data, thresholds=None):
    """Converts statistics from true values (i.e. football stats) to normalized values (scaled between 0 and 1).

        Values are scaled based on notional threshold values set for each statistic, and values outside the thresholds
        are bounded ("clipped") to 0 and 1.

        Args:
            data (pandas.Series | pandas.DataFrame): Series or DataFrame of data.
                If Series: corresponds to a single football stat, with Series name matching a key in the dictionary "thresholds".
                If DataFrame: corresponds to multiple football stats, with column names all matching a key in the dictionary "thresholds".
            thresholds (dict, optional): Maps stat names (e.g. "Pass Yds") to their min and max expected values, in order to scale statistics to
                lie between 0 and 1. Defaults to dictionary "default_norm_thresholds" defined in configuration files.

        Returns:
            pandas.Series: Series of normalized data where each entry in col is mapped between 0 and 1 according to the bounds in thresholds
    """  # fmt: skip

    # Optional input
    if not thresholds:
        thresholds = stats_config.default_norm_thresholds

    # Normalize column-by-column, depending on type of input data
    match type(data):
        case pd.Series:
            # Only one column of data
            data = _normalize_series(data, thresholds)
        case pd.DataFrame:
            # Normalize one column at a time
            data = data.apply(_normalize_series, args=(thresholds,), axis=0)
        case _:
            # Invalid data type
            msg = "Invalid Data Type"
            raise TypeError(msg)

    return data


def unnormalize_stat(data, thresholds=None):
    """Converts statistics from normalized values (scaled between 0 and 1) back to true values (i.e. actual football stats).

        Args:
            data (pandas.Series | pandas.DataFrame): Series or DataFrame of normalized data.
                If Series: corresponds to a single normalized football stat, with Series name matching a key in the dictionary "thresholds".
                If DataFrame: corresponds to multiple normalized football stats, with column names all matching a key in the dictionary "thresholds".
            thresholds (dict, optional): Maps stat names (e.g. "Pass Yds") to their min and max expected values, in order to scale statistics to
                lie between 0 and 1. Defaults to dictionary "default_norm_thresholds" defined in configuration files.

        Returns:
            pandas.Series: Series of unnormalized data where each entry in col is scaled up according to the bounds in thresholds
    """  # fmt: skip

    # Optional input
    if not thresholds:
        thresholds = stats_config.default_norm_thresholds

    # Un-normalize column-by-column, depending on type of input data
    match type(data):
        case pd.Series:
            # Only one column of data
            data = _unnormalize_series(data, thresholds)
        case pd.DataFrame:
            # Normalize one column at a time
            data = data.apply(_unnormalize_series, args=(thresholds,), axis=0)
        case _:
            # Invalid data type
            msg = "Invalid Data Type"
            raise TypeError(msg)

    return data


def stats_to_fantasy_points(stat_line, stat_indices=None, normalized=False, norm_thresholds=None, scoring_weights=None):
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
            norm_thresholds (dict, optional): Maps stat names (e.g. "Pass Yds") to their min and max expected values, in order to scale statistics to
                lie between 0 and 1. Defaults to dictionary "default_norm_thresholds" defined in configuration files.
            scoring_weights (dict, optional): Fantasy points per unit of each statistic (e.g. points per passing yard, points per reception, etc.)
                Defaults to default_scoring_weights.

        Returns:
            pandas.DataFrame: stat_line, un-normalized and with column headers corresponding to stat indices, with an additional entry for
                Fantasy Points calculated based on the fantasy scoring rules. Output type is DataFrame regardless of input type.
    """  # fmt: skip

    # Optional inputs
    if not norm_thresholds:
        norm_thresholds = stats_config.default_norm_thresholds
    if not scoring_weights:
        # Scoring rules in fantasy format
        scoring_weights = stats_config.default_scoring_weights

    if stat_indices == "default":
        stat_indices = stats_config.default_stat_list

    # Convert stat line to data frame if need be
    if isinstance(stat_line, pd.Series):
        stat_line = pd.DataFrame(stat_line).T

    # Assign column names if stat indices are provided
    if stat_indices:
        stat_line = pd.DataFrame(stat_line)
        if len(stat_line.columns) == 1:
            stat_line = stat_line.transpose()
        try:
            stat_line.columns = stat_indices
        except ValueError as e:
            msg = "Unable to assign stat_indices to stat_line."
            raise ValueError(msg) from e

    # Un-normalize stats if necessary
    if normalized:
        stat_line = unnormalize_stat(stat_line, thresholds=norm_thresholds)

    # Trim scoring weights dictionary to only the stats that have non-zero weight
    scoring_weights_nonzero = {key: val for (key, val) in scoring_weights.items() if val != 0}

    # Calculate Fantasy Points from stat line and scoring weights
    try:
        stat_line["Fantasy Points"] = (stat_line[scoring_weights_nonzero.keys()] * scoring_weights_nonzero).sum(axis=1)
    except KeyError as e:
        msg = "Key Error: Missing data in stat_line corresponding to all stats in scoring_weights."
        raise KeyError(msg) from e
    except IndexError as e:
        msg = "Index Error: statistics cannot be matched to weights. stat_indices must be input, or set to 'default'"
        raise IndexError(msg) from e

    return stat_line


def gen_random_games(id_df, n_random, game_ids=None):
    """Generates random game/player combinations from input dataset, with no repeating.

        Args:
            id_df (pandas.DataFrame): DataFrame containing "Player ID", "Week", and "Year" as columns
            n_random (int): Number of random game/player combinations to generate
            game_ids (list, optional): List of pre-selected (not random) games/players. Defaults to None.
                Each element in list must be a dict containing the following keys:
                    - "Player ID" : value -> str
                    - "Year" : value -> int
                    - "Week" : value -> int

        Returns:
            list: List of games/players, including any pre-selected as well as randomly-selected games/players.
                Each element in list is a dict containing the following keys:
                    - "Player ID" : value -> str
                    - "Year" : value -> int
                    - "Week" : value -> int
    """  # fmt: skip

    # Keep only unique games from id_df
    unique_id_df = id_df.copy()
    unique_id_df = unique_id_df.drop_duplicates(subset=["Player ID", "Year", "Week"], keep="first")

    # Copy or initialize list of game IDs
    game_ids = game_ids.copy() if game_ids else []

    # Check that number of unique games in id_df is greater than number of games requested
    if unique_id_df.shape[0] < len(game_ids) + n_random:
        logger.warning("More game IDs requested than unique games available. Returning all games.")
        game_ids = list(unique_id_df.apply(lambda x: {col: x[col] for col in ["Player ID", "Week", "Year"]}, axis=1))
        return game_ids

    # (Optionally) Add random players/games to plot
    for _ in range(n_random):
        valid_new_entry = False
        while not valid_new_entry:
            ind = np.random.randint(unique_id_df.shape[0])
            game_dict = {col: unique_id_df.iloc[ind][col] for col in ["Player ID", "Week", "Year"]}
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
    """  # fmt: skip

    # Convert inputs to array (column vector)
    x_data = np.array(x_data).reshape([-1, 1])
    y_data = np.array(y_data).reshape([-1, 1])

    # Basic info about x and y
    x_mean = np.nanmean(x_data)
    y_mean = np.nanmean(y_data)
    n = x_data.shape[0]
    # Calculate slope and intercept for line of best fit
    s_x_y = sum(x_data * y_data) / n - (sum(x_data) * sum(y_data)) / n**2
    s_x_sq = sum(x_data**2) / n - (sum(x_data) / n) ** 2
    slope = float(s_x_y / s_x_sq)
    intercept = float(y_mean - (slope * x_mean))
    # Calculate r_value
    y_predicted = intercept + slope * x_data
    residuals = y_data - y_predicted
    dists_from_mean = y_data - y_mean
    ssr = sum(residuals**2)
    sst = sum(dists_from_mean**2)
    r_squared = float(1 - (ssr / sst))

    return slope, intercept, r_squared


# PRIVATE FUNCTIONS


def _normalize_series(col, thresholds):
    # Converts statistics of a single type (one data Series) to normalized values
    if col.name in thresholds:
        [lwr, upr] = thresholds[col.name]
        col = (col - lwr) / (upr - lwr)
        col = col.clip(0, 1)
    else:
        logger.warning(f"{col.name} not explicitly normalized")

    return col


def _unnormalize_series(col, thresholds):
    # Converts normalized statistics of a single type (one data Series) to regular values
    if col.name in thresholds:
        # Unfortunately can't undo the clipping from 0 to 1 performed during
        # normalization, so there's a small chance of lost info...
        [lwr, upr] = thresholds[col.name]
        col = col * (upr - lwr) + lwr
    else:
        logger.warning(f"Warning: {col.name} not explicitly normalized")

    return col

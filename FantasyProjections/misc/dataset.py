"""Creates and exports classes to be used for data handling when predicting NFL stats and Fantasy Football scores.

    Classes:
        StatsDataset : child of torch.utils.data.Dataset. Stores and manipulates data arrays containing
            pre-game/midgame/final stats for NFL players/games.
"""  # fmt: skip

import logging
import random

import numpy as np
import pandas as pd
import pandas.testing as pdtest
import torch

from config.log_config import LOGGING_CONFIG

# Set up logger
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")


class StatsDataset(torch.utils.data.Dataset):
    """Dataset holding input data, output data, and ID data relating to statlines for a set of NFL games.

        Child of class torch.utils.data.Dataset

        Attributes:
            name (str): name of the StatsDataset object, used for logging/display purposes.
            x_data (torch.Tensor): Tensor (matrix) containing all play-by-play (i.e. midgame) data from the NFL games in question.
                Corresponds to pre-game/mid-game inputs into a Predictor.
                The data in x_data must be gathered, parsed, and pre-processed using functions in data_pipeline.
            x_data_columns (list): Labels for each column of data in x_data
            y_data (tensor): Tensor (matrix) containing all boxscore (i.e. final) stats data from the NFL games in question.
                Corresponds to "true" stats, though in this dataset they are normalized and not true statistics.
                The data in y_data must be gathered, parsed, and pre-processed using functions in data_pipeline.
            y_data_columns (list): Labels for each column of data in y_data
            id_data (pandas.DataFrame): DataFrame containing all game/player ID data from the NFL games in question.
                The data in id_data must be gathered, parsed, and pre-processed using functions in data_pipeline.

        Public Methods:
            concat : Append two StatsDatasets into one larger StatsDataset, either in-place or returning a new StatsDataset.
            slice_by_criteria : Remove all data from a StatsDataset except the entries that meet a set of criteria.
            remove_game_duplicates : Filters evaluation data to only contain one entry per unique game/player.
            copy : Return a copy of the StatsDataset object. All attributes (e.g. DataFrames) are copies of the originals, not views.
            equals : Compares two StatsDataset objects and returns whether all their data are equal. Optionally can check non-data attributes for equality.

    """  # fmt: skip

    # CONSTRUCTOR

    def __init__(
        self,
        name,
        id_df,
        pbp_df=None,
        boxscore_df=None,
        x_data=None,
        x_data_columns=None,
        y_data=None,
        y_data_columns=None,
        **kwargs,
    ):
        """Constructor for StatsDataset.

            Args (Required):
                name (str): name of the StatsDataset object, used for logging/display purposes.
                id_df (pandas.DataFrame): DataFrame containing all game/player ID data from the NFL games in question.
                    The data in id_df must be gathered, parsed, and pre-processed using functions in data_pipeline.
            Args (Initialization Option 1):
                pbp_df (pandas.DataFrame): DataFrame containing all play-by-play (i.e. midgame) data from the NFL games in question.
                    The data in pbp_df must be gathered, parsed, and pre-processed using functions in data_pipeline.
                boxscore_df (pandas.DataFrame): DataFrame containing all boxscore (i.e. final) stats data from the NFL games in question.
                    The data in boxscore_df must be gathered, parsed, and pre-processed using functions in data_pipeline.
            Args (Initialization Option 2):
                x_data (torch.Tensor): Matrix containing all play-by-play (i.e. midgame) data from the NFL games in question.
                x_data_columns (list): Labels for each column of data in x_data
                y_data (torch.Tensor): Matrix containing all boxscore (i.e. final) stats data from the NFL games in question.
                y_data_columns (list): Labels for each column of data in y_data

            Keyword-Args:
                start_index (int, optional): First index to use in DataFrames (if taking consecutive data from the DataFrames).
                    Defaults to None (start of DataFrame).
                end_index (int, optional): First index to exclude from DataFrames (if taking consecutive data from the DataFrames).
                    Defaults to None (end of DataFrame).
                shuffle (bool, optional): Whether to shuffle the rows of the DataFrames when generating. Defaults to False.
                weeks (list, optional): Week numbers from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                years (list, optional): Year numbers from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                teams (list, optional): Team names from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                player_ids (list, optional): Player IDs from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                elapsed_time (list, optional): Game times from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
        """  # fmt: skip

        # Handle optional inputs and assign default values not passed
        start_index = kwargs.get("start_index")  # Default starts at the beginning of the array
        end_index = kwargs.get("end_index")  # Default ends at the end of the array
        shuffle = kwargs.get("shuffle", False)
        # Other valid kwargs that are not currently initialized to default
        # values: weeks, years, teams, players, elapsed_time

        # Check that ID data is valid
        if not isinstance(id_df, pd.DataFrame):
            msg = "Invalid id_df input type to StatsDataset."
            raise TypeError(msg)
        # Check that x data is valid
        if not (isinstance(pbp_df, pd.DataFrame) or (isinstance(x_data, torch.Tensor) and isinstance(x_data_columns, list))):
            msg = "Invalid x_data/play-by-play input type to StatsDataset."
            raise TypeError(msg)
        # Check that y data is valid
        if not (isinstance(boxscore_df, pd.DataFrame) or (isinstance(y_data, torch.Tensor) and isinstance(y_data_columns, list))):
            msg = "Invalid y_data/boxscore input type to StatsDataset."
            raise TypeError(msg)

        # Name
        self.name = name

        # Process DFs; convert numeric data (inputs "x" and desired
        # outputs "y") to tensors
        if pbp_df is not None:
            self.x_data = torch.tensor(pbp_df.values)
            self.x_data_columns = pbp_df.columns.to_list()
        else:
            self.x_data = x_data
            self.x_data_columns = x_data_columns

        if boxscore_df is not None:
            self.y_data = torch.tensor(boxscore_df.values)
            self.y_data_columns = boxscore_df.columns.to_list()
        else:
            self.y_data = y_data
            self.y_data_columns = y_data_columns

        self.id_data = id_df

        # Trim to only the desired data, according to multiple possible methods:
        # 1. Weeks, Years, Teams, Player IDs, and/or Elapsed Time specified
        self.valid_criteria = ["weeks", "years", "teams", "player_ids", "elapsed_time"]
        if len(set(self.valid_criteria) & set(kwargs)) > 0:
            self.slice_by_criteria(**kwargs)

        # 2. using start_index and num_to_use
        else:
            self.x_data = self.x_data[start_index:end_index]
            self.y_data = self.y_data[start_index:end_index]
            self.id_data = self.id_data.iloc[start_index:end_index]

        # Optionally shuffle the data
        if shuffle:
            indices = list(range(self.x_data.shape[0]))
            random.seed(10)
            random.shuffle(indices)
            self.x_data = self.x_data[indices]
            self.y_data = self.y_data[indices]
            self.id_data = self.id_data.iloc[indices]

    # PUBLIC METHODS

    def concat(self, other, inplace=True):
        """Appends two StatsDatasets into one larger StatsDataset, either in-place or returning a new StatsDataset.

            Args:
                other (StatsDataset): Second StatsDataset to combine with the object "self" calling this method.
                inplace (bool, optional): If True, self is modified in-place; if False, a new StatsDataset is returned. Defaults to True.

            Raises:
                NameError: Columns of one or multiple DataFrames do not match.

            Returns:
                [None | StatsDataset]: If inplace==True (default), returns None. If inplace==False, returns the concatenated StatsDataset.

        """  # fmt: skip

        # Check that data labels match each other
        if self.x_data_columns != other.x_data_columns:
            msg = f"x data labels do not match for DataFrames {self.name} and {other.name}"
            logging.error(f"Error: {msg}")
            raise NameError(msg)
        if self.y_data_columns != other.y_data_columns:
            msg = f"y data labels do not match for DataFrames {self.name} and {other.name}"
            logging.error(f"Error: {msg}")
            raise NameError(msg)
        if self.id_data.columns.to_list() != other.id_data.columns.to_list():
            msg = f"ID data labels do not match for DataFrames {self.name} and {other.name}"
            logging.error(f"Error: {msg}")
            raise NameError(msg)

        # Concatenate data structures
        joined_x_data = torch.cat((self.x_data, other.x_data))
        joined_y_data = torch.cat((self.y_data, other.y_data))
        joined_id_data = pd.concat((self.id_data, other.id_data)).reset_index(drop=True)

        # Return in place (modify self)
        if inplace:
            self.x_data = joined_x_data
            self.y_data = joined_y_data
            self.id_data = joined_id_data

        # Return new object
        else:
            new_dataset = StatsDataset(
                name=self.name,
                id_df=joined_id_data,
                x_data=joined_x_data,
                x_data_columns=self.x_data_columns,
                y_data=joined_y_data,
                y_data_columns=self.y_data_columns,
            )
            return new_dataset
        return None

    def slice_by_criteria(self, inplace=True, **kwargs):
        """Removes all data from a StatsDataset except the entries that meet a set of criteria.

            Args:
                inplace (bool, optional): If True, self is modified in-place; if False, a new StatsDataset is returned. Defaults to True.
                kwargs:
                    weeks (list, optional): Week numbers from the DataFrames to include in the StatsDataset. If not passed, ignored.
                    years (list, optional): Year numbers from the DataFrames to include in the StatsDataset. If not passed, ignored.
                    teams (list, optional): Team names from the DataFrames to include in the StatsDataset. If not passed, ignored.
                    player_ids (list, optional): Player IDs from the DataFrames to include in the StatsDataset. If not passed, ignored.
                    elapsed_time (list, optional): Game times from the DataFrames to include in the StatsDataset. If not passed, ignored.

            Returns:
                [None | StatsDataset]: If inplace==True (default), returns None. If inplace==False, returns the modified StatsDataset.

        """  # fmt: skip

        criteria_var_to_col = {
            "weeks": "Week",
            "years": "Year",
            "teams": "Team",
            "player_ids": "Player ID",
            "elapsed_time": "Elapsed Time",
        }
        df_query = " & ".join(
            [f'`{criteria_var_to_col[crit]}` in @kwargs["{crit}"]' for crit in self.valid_criteria if kwargs.get(crit)],
        )
        try:
            row_nums = self.id_data.reset_index().query(df_query).index.values
        except ValueError:
            logger.warning("Invalid criteria (or no criteria) passed to method slice_by_criteria. No slicing will be performed.")
            row_nums = self.id_data.reset_index().index.values

        if inplace:
            self.x_data = self.x_data[row_nums]
            self.y_data = self.y_data[row_nums]
            self.id_data = self.id_data.iloc[row_nums]
        else:
            new_dataset = StatsDataset(
                name=self.name,
                id_df=self.id_data.iloc[row_nums],
                x_data=self.x_data[row_nums],
                x_data_columns=self.x_data_columns,
                y_data=self.y_data[row_nums],
                y_data_columns=self.y_data_columns,
            )
            return new_dataset
        return None

    def remove_game_duplicates(self, inplace=False):
        """Filters evaluation data to only contain one entry per unique game/player.

            Removes all but the first row in id_data for each Player ID/Year/Week combination. (First row is typically when Elapsed Time = 0).
            Adjusts all applicable attributes in StatsDataset: x_data, y_data, id_data

            Args:
                inplace (bool, optional): If True, self is modified in-place; if False, a new StatsDataset is returned. Defaults to True.

            Returns:
                [None | StatsDataset]: If inplace==True (default), returns None.
                    If inplace==False, returns the modified StatsDataset with only one row per unique game/player

        """  # fmt: skip

        # Obtain indices of duplicated rows to remove
        duplicated_rows = self.id_data.reset_index().duplicated(subset=["Player ID", "Year", "Week"])

        # Remove duplicated rows from each attribute of new dataset
        new_x_data = self.x_data[np.logical_not(duplicated_rows)]
        new_y_data = self.y_data[np.logical_not(duplicated_rows)]
        new_id_data = self.id_data.reset_index(drop=True).loc[np.logical_not(duplicated_rows)].reset_index(drop=True)

        if inplace:
            self.x_data = new_x_data
            self.y_data = new_y_data
            self.id_data = new_id_data
        else:
            new_dataset = StatsDataset(
                name=self.name,
                id_df=new_id_data,
                x_data=new_x_data,
                x_data_columns=self.x_data_columns,
                y_data=new_y_data,
                y_data_columns=self.y_data_columns,
            )
            return new_dataset
        return None

    def copy(self):
        """Returns a copy of the StatsDataset object. All attributes (e.g. DataFrames) are copies of the originals, not views.

            Returns:
                StatsDataset: Copy of the StatsDataset object invoking this method. All attributes are copies, not views.

        """  # fmt: skip

        return StatsDataset(
            name=self.name,
            id_df=self.id_data,
            x_data=self.x_data,
            x_data_columns=self.x_data_columns,
            y_data=self.y_data,
            y_data_columns=self.y_data_columns,
        )

    def equals(self, other, check_non_data_attributes=False):
        """Compares two StatsDataset objects and returns whether all their data are equal. Optionally can check non-data attributes for equality.

            Non-data attributes include: name, valid_criteria.

            Args:
                other (StatsDataset): Object to compare against StatsDataset object invoking the equals method.
                check_non_data_attributes (bool, optional): Whether to enforce that all attributes (including name, etc.) match for equals=True.
                    Defaults to False.

            Returns:
                bool: Whether the StatsDataset objects have identical data (and identical other attributes, if check_non_data_attributes).

        """  # fmt: skip

        # Compare non-data attributes
        if check_non_data_attributes:
            name_equal = self.name == other.name
            valid_criteria_equal = self.valid_criteria == other.valid_criteria
            if not (name_equal and valid_criteria_equal):
                return False

        # Compare non-pandas data
        x_data_equal = torch.equal(self.x_data, other.x_data)
        x_data_columns_equal = self.x_data_columns == other.x_data_columns
        y_data_equal = torch.equal(self.y_data, other.y_data)
        y_data_columns_equal = self.y_data_columns == other.y_data_columns
        # If a non-pandas attribute is False, return False
        if not (x_data_equal and x_data_columns_equal and y_data_equal and y_data_columns_equal):
            return False

        # Compare pandas object attributes
        try:
            pdtest.assert_frame_equal(self.id_data, other.id_data, check_dtype=False)
        except AssertionError:
            return False

        # If pandas object comparison passes without AssertionError, return True
        return True

    def __len__(self):
        # Returns the length of the StatsDataset (number of rows in input data x_data)
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        # Returns the input and output data at a given index
        return self.x_data[idx], self.y_data[idx]

    def __getid__(self, idx):
        # Returns the ID data at a given index
        return self.id_data.iloc[idx]

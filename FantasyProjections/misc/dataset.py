"""Creates and exports classes to be used for data handling when predicting NFL stats and Fantasy Football scores.

    Classes:
        StatsDataset : child of torch.utils.data.Dataset. Stores and manipulates data arrays containing 
            pre-game/midgame/final stats for NFL players/games.
"""

import logging
import random
import pandas as pd
import pandas.testing as pdtest
import torch
from torch.utils.data import Dataset


class StatsDataset(Dataset):
    """Dataset holding input data, output data, and ID data relating to statlines for a set of NFL games

        Child of class torch.utils.data.Dataset

        Attributes:
            name (str): name of the StatsDataset object, used for logging/display purposes.
            x_df (pandas.DataFrame): DataFrame containing all play-by-play (i.e. midgame) data from the NFL games in question.
                The data in x_df must be gathered, parsed, and pre-processed using functions in data_pipeline.
            x_data (tensor): data from x_df converted to tensor format for faster computations.
                Corresponds to pre-game/mid-game inputs into a Predictor.
            y_df (pandas.DataFrame): DataFrame containing all boxscore (i.e. final) stats data from the NFL games in question.
                The data in y_df must be gathered, parsed, and pre-processed using functions in data_pipeline.
            y_data (tensor): data from y_df converted to tensor format for faster computations.
                Corresponds to "true" stats, though in this dataset they are normalized and not true statistics.
            id_data (pandas.DataFrame): DataFrame containing all game/player ID data from the NFL games in question.
                The data in id_data must be gathered, parsed, and pre-processed using functions in data_pipeline.

        Public Methods:
            concat : Append two StatsDatasets into one larger StatsDataset, either in-place or returning a new StatsDataset.
            slice_by_criteria : Remove all data from a StatsDataset except the entries that meet a set of criteria.
            copy : Return a copy of the StatsDataset object. All attributes (e.g. DataFrames) are copies of the originals, not views.
            equals : Compares two StatsDataset objects and returns whether all their data are equal. Optionally can check non-data attributes for equality.
    """

    # CONSTRUCTOR

    def __init__(self, name, pbp_df, boxscore_df,
                 id_df, **kwargs):
        """Constructor for StatsDataset

            Args:
                name (str): name of the StatsDataset object, used for logging/display purposes.
                pbp_df (pandas.DataFrame): DataFrame containing all play-by-play (i.e. midgame) data from the NFL games in question.
                    The data in pbp_df must be gathered, parsed, and pre-processed using functions in data_pipeline.
                boxscore_df (pandas.DataFrame): DataFrame containing all boxscore (i.e. final) stats data from the NFL games in question.
                    The data in boxscore_df must be gathered, parsed, and pre-processed using functions in data_pipeline.
                id_df (pandas.DataFrame): DataFrame containing all game/player ID data from the NFL games in question.
                    The data in id_df must be gathered, parsed, and pre-processed using functions in data_pipeline.
            
            Keyword-Args:
                start_index (int, optional): First index to use in DataFrames (if taking consecutive data from the DataFrames). 
                    Defaults to None (start of DataFrame).
                end_index (int, optional): First index to exclude from DataFrames (if taking consecutive data from the DataFrames).
                    Defaults to None (end of DataFrame).
                shuffle (bool, optional): Whether to shuffle the rows of the DataFrames when generating. Defaults to False.
                weeks (list, optional): Week numbers from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                years (list, optional): Year numbers from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                teams (list, optional): Team names from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                players (list, optional): Player names from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                elapsed_time (list, optional): Game times from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.

            Raises:
                TypeError: inputs pbp_df, boxscore_df, and/or id_df are not of type pandas.DataFrame.
        """
        # Handle optional inputs and assign default values not passed
        start_index = kwargs.get('start_index', None) # Default starts at the beginning of the array
        end_index = kwargs.get('end_index', None) # Default ends at the end of the array
        shuffle = kwargs.get('shuffle', False)
        # Other valid kwargs that are not currently initialized to default
        # values: weeks, years, teams, players, elapsed_time

        # Check input data types
        if not(isinstance(pbp_df,pd.DataFrame) and isinstance(boxscore_df,pd.DataFrame) and isinstance(id_df,pd.DataFrame)):
            raise TypeError('Invalid input type to StatsDataset.')

        # Name
        self.name = name

        # Process DFs; convert numeric data (inputs "x" and desired
        # outputs "y") to tensors
        self.x_df = pbp_df
        self.x_data = torch.tensor(self.x_df.values)
        self.y_df = boxscore_df
        self.y_data = torch.tensor(self.y_df.values)
        self.id_data = id_df

        # Trim to only the desired data, according to multiple possible methods:
        # 1. Weeks, Years, Teams, Players, and/or Elapsed Time specified
        self.valid_criteria = ['weeks', 'years', 'teams', 'players', 'elapsed_time']
        if len(set(self.valid_criteria) & set(kwargs)) > 0:
            self.slice_by_criteria(**kwargs)

        # 2. using start_index and num_to_use
        else:
            self.x_df = self.x_df.iloc[start_index:end_index]
            self.y_df = self.y_df.iloc[start_index:end_index]
            self.x_data = self.x_data[start_index:end_index]
            self.y_data = self.y_data[start_index:end_index]
            self.id_data = self.id_data.iloc[start_index:end_index]

        # Optionally shuffle the data
        if shuffle:
            indices = list(range(self.x_data.shape[0]))
            random.seed(10)
            random.shuffle(indices)
            self.x_df = self.x_df.iloc[indices]
            self.y_df = self.y_df.iloc[indices]
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
        """

        # Check that data labels match each other
        if self.x_df.columns.to_list() != other.x_df.columns.to_list():
            logging.error(f'Error: x data labels do not match for DataFrames {self.name} and {other.name}')
            raise NameError(f'x data labels do not match for DataFrames {self.name} and {other.name}')
        if self.y_df.columns.to_list() != other.y_df.columns.to_list():
            logging.error(f'Warning: y data labels do not match for DataFrames {self.name} and {other.name}')
            raise NameError(f'y data labels do not match for DataFrames {self.name} and {other.name}')
        if self.id_data.columns.to_list() != other.id_data.columns.to_list():
            logging.error(f'Warning: ID data labels do not match for DataFrames {self.name} and {other.name}')
            raise NameError(f'ID data labels do not match for DataFrames {self.name} and {other.name}')

        # Concatenate data structures
        joined_x_data = torch.cat((self.x_data, other.x_data))
        joined_y_data = torch.cat((self.y_data, other.y_data))
        joined_x_df = pd.concat((self.x_df, other.x_df)).reset_index(drop=True)
        joined_y_df = pd.concat((self.y_df, other.y_df)).reset_index(drop=True)
        joined_id_data = pd.concat((self.id_data, other.id_data)).reset_index(drop=True)

        # Return in place (modify self)
        if inplace:
            self.x_data = joined_x_data
            self.y_data = joined_y_data
            self.x_df = joined_x_df
            self.y_df = joined_y_df
            self.id_data = joined_id_data

        # Return new object
        else:
            new_dataset = StatsDataset(
                self.name,
                joined_x_df,
                joined_y_df,
                joined_id_data)
            return new_dataset
        return None


    def slice_by_criteria(self,inplace=True,**kwargs):
        """Removes all data from a StatsDataset except the entries that meet a set of criteria.

            Args:
                inplace (bool, optional): If True, self is modified in-place; if False, a new StatsDataset is returned. Defaults to True.
            
            Keyword-Args:
                weeks (list, optional): Week numbers from the DataFrames to include in the StatsDataset. If not passed, ignored.
                years (list, optional): Year numbers from the DataFrames to include in the StatsDataset. If not passed, ignored.
                teams (list, optional): Team names from the DataFrames to include in the StatsDataset. If not passed, ignored.
                players (list, optional): Player names from the DataFrames to include in the StatsDataset. If not passed, ignored.
                elapsed_time (list, optional): Game times from the DataFrames to include in the StatsDataset. If not passed, ignored.

            Returns:
                [None | StatsDataset]: If inplace==True (default), returns None. If inplace==False, returns the modified StatsDataset.
        """

        criteria_var_to_col = {
            'weeks': 'Week',
            'years': 'Year',
            'teams': 'Team',
            'players': 'Player',
            'elapsed_time': 'Elapsed Time'}
        df_query = ' & '.join(
            [f'`{criteria_var_to_col[crit]}` in @kwargs["{crit}"]'
                for crit in self.valid_criteria if kwargs.get(crit)])
        try:
            indices = self.id_data.query(df_query).index.values
        except ValueError:
            logging.warning('Invalid criteria (or no criteria) passed to method slice_by_criteria. No slicing will be performed.')
            indices = self.id_data.index.values

        if inplace:
            self.x_df = self.x_df.loc[indices]
            self.x_data = self.x_data[indices]
            self.y_df = self.y_df.loc[indices]
            self.y_data = self.y_data[indices]
            self.id_data = self.id_data.loc[indices]
        else:
            new_dataset = StatsDataset(
                self.name,
                self.x_df.loc[indices],
                self.y_df.loc[indices],
                self.id_data.loc[indices])
            return new_dataset
        return None


    def copy(self):
        """Returns a copy of the StatsDataset object. All attributes (e.g. DataFrames) are copies of the originals, not views.

            Returns:
                StatsDataset: Copy of the StatsDataset object invoking this method. All attributes are copies, not views.
        """

        return StatsDataset(name=self.name,
                            pbp_df=self.x_df,
                            boxscore_df=self.y_df,
                            id_df=self.id_data)


    def equals(self, other, check_non_data_attributes=False):
        """Compares two StatsDataset objects and returns whether all their data are equal. Optionally can check non-data attributes for equality.

            Non-data attributes include: name, valid_criteria.

            Args:
                other (StatsDataset): Object to compare against StatsDataset object invoking the equals method.
                check_non_data_attributes (bool, optional): Whether to enforce that all attributes (including name, etc.) match for equals=True. 
                    Defaults to False.
                
            Returns:
                bool: Whether the StatsDataset objects have identical data (and identical other attributes, if check_non_data_attributes).
        """
        # Compare non-data attributes
        if check_non_data_attributes:
            name_equal = self.name == other.name
            valid_criteria_equal = self.valid_criteria == other.valid_criteria
            if not (name_equal and valid_criteria_equal):
                return False

        # Compare non-pandas data
        x_data_equal = torch.equal(self.x_data, other.x_data)
        y_data_equal = torch.equal(self.y_data, other.y_data)
        # If a non-pandas attribute is False, return False
        if not (x_data_equal and y_data_equal):
            return False

        # Compare pandas object attributes
        try:
            pdtest.assert_frame_equal(self.x_df,other.x_df,check_dtype=False)
            pdtest.assert_frame_equal(self.y_df,other.y_df,check_dtype=False)
            pdtest.assert_frame_equal(self.id_data,other.id_data,check_dtype=False)
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

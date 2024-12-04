"""Creates and exports classes to be used for data handling when predicting NFL stats and Fantasy Football scores.

    Classes:
        StatsDataset : child of torch.utils.data.Dataset. Stores and manipulates data arrays containing 
            pre-game/midgame/final stats for NFL players/games.
"""

import logging
import random
import pandas as pd
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
            concat : append two StatsDatasets into one larger StatsDataset, either in-place or returning a new StatsDataset.
            slice_by_criteria : remove all data from a StatsDataset except the entries that meet a set of criteria
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
                start_index (int, optional): First index to use in DataFrames (if taking consecutive data from the DataFrames). Defaults to 0.
                num_to_use (int, optional): Number of rows to use in DataFrames (if taking consecutive data from the DataFrames). Defaults to -1 (entire DataFrame).
                shuffle (bool, optional): Whether to shuffle the rows of the DataFrames when generating. Defaults to False.
                weeks (list, optional): Week numbers from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                years (list, optional): Year numbers from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                teams (list, optional): Team names from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                players (list, optional): Player names from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
                elapsed_time (list, optional): Game times from the DataFrames to include in the StatsDataset (if slicing Dataset by criteria). If not passed, ignored.
        """
        # Handle optional inputs and assign default values not passed
        start_index = kwargs.get('start_index', 0)
        num_to_use = kwargs.get('num_to_use', -1)
        shuffle = kwargs.get('shuffle', False)
        # Other valid kwargs that are not currently initialized to default
        # values: weeks, years, teams, players, elapsed_time

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
            self.x_data = self.x_data[start_index:start_index + num_to_use]
            self.y_data = self.y_data[start_index:start_index + num_to_use]
            self.id_data = self.id_data.iloc[start_index:start_index + num_to_use]

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

            Returns:
                [None | StatsDataset]: If inplace==True (default), returns None. If inplace==False, returns the concatenated StatsDataset. 
        """

        # Check that data labels match each other
        if self.x_df.columns.to_list() != other.x_df.columns.to_list():
            logging.warning(f'Warning: x data labels do not match for Datasets {self.name} and {other.name}')
        if self.y_df.columns.to_list() != other.y_df.columns.to_list():
            logging.warning(f'Warning: y data labels do not match for Datasets {self.name} and {other.name}')

        # Concatenate data structures
        joined_x_data = torch.cat((self.x_data, other.x_data))
        joined_y_data = torch.cat((self.y_data, other.y_data))
        joined_id_data = pd.concat((self.id_data, other.id_data))

        # Return in place (modify self)
        if inplace:
            self.x_data = joined_x_data
            self.y_data = joined_y_data
            self.id_data = joined_id_data

        # Return new object
        else:
            new_dataset = StatsDataset(
                self.name,
                self.x_df,
                self.y_df,
                self.id_data)
            new_dataset.x_data = joined_x_data
            new_dataset.y_data = joined_y_data
            new_dataset.id_data = joined_id_data
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
        indices = self.id_data.query(df_query).index.values

        if inplace:
            self.x_data = self.x_data[indices]
            self.y_data = self.y_data[indices]
            self.id_data = self.id_data.loc[indices]
        else:
            new_dataset = StatsDataset(
                self.name,
                self.x_df,
                self.y_df,
                self.id_data)
            new_dataset.x_data = new_dataset.x_data[indices]
            new_dataset.y_data = new_dataset.y_data[indices]
            new_dataset.id_data = new_dataset.id_data.loc[indices]
            return new_dataset
        return None


    def __len__(self):
        # Returns the length of the StatsDataset (number of rows in input data x_data)
        return self.x_data.shape[0]


    def __getitem__(self, idx):
        # Returns the input and output data at a given index
        return self.x_data[idx], self.y_data[idx]


    def __getid__(self, idx):
        # Returns the ID data at a given index
        return self.id_data.iloc[idx]


    def __getids__(self):
        # Returns all ID data
        return self.id_data

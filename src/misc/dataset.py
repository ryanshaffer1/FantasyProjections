import random
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    # CONSTRUCTOR

    def __init__(self, pbp_df, boxscore_df,
                 id_df, **kwargs):
        # Handle optional inputs and assign default values not passed
        start_index = kwargs.get('start_index', 0)
        num_to_use = kwargs.get('num_to_use', -1)
        shuffle = kwargs.get('shuffle', False)
        # Other valid kwargs that are not currently initialized to default
        # values: weeks, years, teams, players, elapsed_time

        # Read data from file; convert numeric data (inputs "x" and desired
        # outputs "y") to tensors
        self.x_df = pbp_df
        self.x_data_labels = list(self.x_df.columns)
        self.x_data = torch.tensor(self.x_df.values)
        self.y_df = boxscore_df
        self.y_data_labels = list(self.y_df.columns)
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
        # Check that data labels match each other
        if self.x_data_labels != other.x_data_labels:
            print('Warning: x data labels do not match')
        if self.y_data_labels != other.y_data_labels:
            print('Warning: y data labels do not match')

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
            new_dataset = CustomDataset(
                self.x_df,
                self.y_df,
                self.id_data)
            new_dataset.x_data = joined_x_data
            new_dataset.y_data = joined_y_data
            new_dataset.id_data = joined_id_data
            return new_dataset
        return None


    def slice_by_criteria(self,inplace=True,**kwargs):
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
            new_dataset = CustomDataset(
                self.x_df,
                self.y_df,
                self.id_data)
            new_dataset.x_data = new_dataset.x_data[indices]
            new_dataset.y_data = new_dataset.y_data[indices]
            new_dataset.id_data = new_dataset.id_data.loc[indices]
            return new_dataset
        return None


    def __len__(self):
        return self.x_data.shape[0]


    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


    def __getid__(self, idx):
        return self.id_data.iloc[idx]


    def __getids__(self):
        return self.id_data

"""Creates and exports functions to process gathered NFL stats data into a format usable by a Neural Net Fantasy Predictor.

    Functions:
        preprocess_nn_data : Converts NFL stats data from raw statistics to a Neural Network-readable format.
        add_word_bank_to_df : Converts a column of unique values from a DataFrame into a series of columns in a new DataFrame, each one corresponding to one of the values.
        strip_and_normalize : Trims a DataFrame to only the columns of interest, and normalizes each stat to take a value between 0 and 1.
"""

import pandas as pd
from misc.nn_helper_functions import normalize_stat
from misc.manage_files import create_folders

def preprocess_nn_data(midgame_input, final_stats_input,
                       save_folder=None, save_filenames=None):
    """Converts NFL stats data from raw statistics to a Neural Network-readable format.
    
        Main steps:
            1. Cleans dataframes (fills in blanks/NaNs as 0, converts all True/False to 1/0, removes non-numeric data)
            2. Matches every row in midgame to the corresponding row in final_stats
            3. Normalizing statistics so that all values are between 0 and 1
            4. Encoding player, team, and opponent IDs as vectors of 0's and 1's (1 corresponds to the correct ID, 0 everywhere else)

        Args:
            midgame_input (pandas.DataFrame | str): Stats accrued over the course of an NFL game for a set of players/games, OR path to csv file containing this data.
            final_stats_input (pandas.DataFrame | str): Stats at the end of an NFL game for a set of players/games, OR path to csv file containing this data.
            save_folder (str, optional): folder to save files that can be ingested by a Neural Net Fantasy Predictor. Defaults to None (files will not be saved).
            save_filenames (dict, optional): Filename to use for each neural net input csv. Defaults to:
            {
                "midgame": "midgame_data_to_nn.csv",
                "final": "final_stats_to_nn.csv",
                "id": "data_ids.csv"
            }

        Returns:
            pandas.DataFrame: Midgame input data in Neural Net-readable format
            pandas.DataFrame: Final Stats input data in Neural Net-readable format
            pandas.DataFrame: ID (player/game information) input data in Neural Net-readable format
    """

    # Optional save_filenames input
    if not save_filenames:
        save_filenames = {
            'midgame': 'midgame_data_to_nn.csv',
            'final': 'final_stats_to_nn.csv',
            'id': 'data_ids.csv'}

    # Read files if raw dataframes are not passed in
    if not isinstance(midgame_input,pd.DataFrame):
        midgame_input = pd.read_csv(midgame_input, low_memory=False)
    else:
        midgame_input = midgame_input.reset_index(drop=False)
    if not isinstance(final_stats_input,pd.DataFrame):
        final_stats_input = pd.read_csv(final_stats_input)
    else:
        final_stats_input = final_stats_input.reset_index(drop=False)

    # A few things to prep the data for use in the NN
    with pd.option_context("future.no_silent_downcasting", True):
        midgame_input = midgame_input.fillna(0)  # Fill in blank spaces
        final_stats_input = final_stats_input.fillna(0)  # Fill in blank spaces
    midgame_input["Site"] = pd.to_numeric(midgame_input["Site"] == "Home")  # Convert Site to 1/0
    midgame_input["Possession"] = pd.to_numeric(midgame_input["Possession"])  # Convert Possession to 1/0

    # Only keep numeric data (non-numeric columns will be stripped out later in pre-processing)
    id_columns = ["Player", "Year", "Week", "Team", "Opponent", "Position", "Elapsed Time"]
    midgame_numeric_columns = [
        "Elapsed Time",
        "Team Score",
        "Opp Score",
        "Possession",
        "Field Position",
        "Pass Att",
        "Pass Cmp",
        "Pass Yds",
        "Pass TD",
        "Int",
        "Rush Att",
        "Rush Yds",
        "Rush TD",
        "Rec",
        "Rec Yds",
        "Rec TD",
        "Fmb",
        "Age",
        "Site",
        "Team Wins",
        "Team Losses",
        "Team Ties",
        "Opp Wins",
        "Opp Losses",
        "Opp Ties",
    ]
    final_stats_numeric_columns = [
        "Pass Att",
        "Pass Cmp",
        "Pass Yds",
        "Pass TD",
        "Int",
        "Rush Att",
        "Rush Yds",
        "Rush TD",
        "Rec",
        "Rec Yds",
        "Rec TD",
        "Fmb",
    ]
    # Sort by year/week/team/player
    midgame_input = midgame_input.sort_values(
        by=["Year", "Week", "Team", "Player"], ascending=[True, True, True, True]
    )

    # Match inputs (pbp data) to outputs (boxscore data) by index (give each
    # input and corresponding output the same index in their df)
    final_stats_input = (
        final_stats_input.set_index(["Player", "Year", "Week"])
        .loc[midgame_input.set_index(["Player", "Year", "Week"]).index]
        .reset_index()
    )

    # Keep identifying info in a separate dataframe
    id_df = midgame_input[id_columns]

    # Strip out non-numeric columns, and normalize numeric columns to between 0 and 1
    midgame_input = strip_and_normalize(midgame_input, midgame_numeric_columns)
    final_stats_input = strip_and_normalize(final_stats_input, final_stats_numeric_columns)

    # Encode each non-numeric, relevant pbp field (Player, Team, Position) in a "word bank":
    fields = ["Position", "Player", "Team", "Opponent"]
    for field in fields:
        word_bank_df = add_word_bank_to_df(field, id_df)
        midgame_input = pd.concat((midgame_input, word_bank_df), axis=1)


    print('Data pre-processed for projections')

    # Save data
    if save_folder is not None:
        create_folders(save_folder)
        print('Saving pre-processed NN data...')
        midgame_input.to_csv(f'{save_folder}{save_filenames['midgame']}', index=False)
        final_stats_input.to_csv(f'{save_folder}{save_filenames['final']}', index=False)
        id_df.to_csv(f'{save_folder}{save_filenames['id']}', index=False)
        print(f'Saved pre-processed data to {save_folder}')

    return midgame_input, final_stats_input, id_df


def add_word_bank_to_df(field, id_df):
    """Converts a column of unique values from a DataFrame into a series of columns in a new DataFrame, each one corresponding to one of the values.
        For each row of the DataFrame, a 1 is placed in the column corresponding to its original value, and 0's are placed in every other new column.
        Note that no columns are removed from the original DataFrames, including the column used to generate the word bank.
        
        Args:
            field (str): Name of column in id_df to "enumerate" (convert into distinct columns). Example: "Player"
            id_df (pandas.DataFrame): DataFrame containing the player/game info, including the column of unique values.

        Returns:
            pandas.DataFrame: DataFrame containing the series of columns, each corresponding to a unique value in the input DataFrame column.
    """

    word_bank = id_df[field].unique()
    word_bank.sort()
    print(f"{len(word_bank)} unique {field}s")
    word_bank_df = pd.DataFrame(columns=field + "=" + word_bank)
    for entry in word_bank:
        word_bank_df[field + "=" + entry] = (id_df[field] == entry).astype(int)

    return word_bank_df

def strip_and_normalize(df, cols):
    """Trims a DataFrame to only the columns of interest, and normalizes each stat to take a value between 0 and 1.
    
        The normalization calculation is performed in a separate function, normalize_stat()

        Args:
            df (pandas.DataFrame): DataFrame containing NFL stats data.
            cols (list): List of columns in df to retain.

        Returns:
            pandas.DataFrame: input DataFrame, with only the input columns included, and all numeric values between 0 and 1.
    """

    df = df[cols]
    df = df.apply(normalize_stat,axis=0)
    return df

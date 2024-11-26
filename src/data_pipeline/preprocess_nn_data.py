import pandas as pd
from misc.nn_helper_functions import normalize_stat
from misc.manage_files import create_folders

def preprocess_nn_data(midgame_input=None, final_stats_input=None,
                       save_data=True, save_folder='data/', save_filenames=None):
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
        midgame_input = add_word_bank_to_df(field, id_df, midgame_input)

    print('Data pre-processed for projections')

    # Save data
    if save_data:
        create_folders(save_folder)
        print('Saving pre-processed NN data...')
        midgame_input.to_csv(f'{save_folder}{save_filenames['midgame']}', index=False)
        final_stats_input.to_csv(f'{save_folder}{save_filenames['final']}', index=False)
        id_df.to_csv(f'{save_folder}{save_filenames['id']}', index=False)
        print(f'Saved pre-processed data to {save_folder}')

    return midgame_input, final_stats_input, id_df


def add_word_bank_to_df(field, id_df, midgame_input):
    word_bank = id_df[field].unique()
    word_bank.sort()
    print(f"{len(word_bank)} unique {field}s")
    df = pd.DataFrame(columns=field + "=" + word_bank)
    for entry in word_bank:
        df[field + "=" + entry] = (id_df[field] == entry).astype(int)
    midgame_input = pd.concat((midgame_input, df), axis=1)

    return midgame_input

def strip_and_normalize(df, cols):
    df = df[cols]
    df = df.apply(normalize_stat,axis=0)
    return df

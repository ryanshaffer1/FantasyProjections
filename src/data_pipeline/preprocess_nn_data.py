import pandas as pd
from misc.nn_helper_functions import normalize_stat

def preprocess_nn_data(pbp_df=None, boxscore_df=None, pbp_input_file=None, boxscore_input_file=None, save_data=True):
    # Read files if raw dataframes are not passed in
    if not isinstance(pbp_df,pd.DataFrame):
        pbp_df = pd.read_csv(pbp_input_file, low_memory=False)
    if not isinstance(boxscore_df,pd.DataFrame):
        boxscore_df = pd.read_csv(boxscore_input_file)

    # A few things to prep the data for use in the NN
    pbp_df = pbp_df.fillna(0)  # Fill in blank spaces
    boxscore_df = boxscore_df.fillna(0)  # Fill in blank spaces
    pbp_df["Site"] = pd.to_numeric(pbp_df["Site"] == "Home")  # Convert Site to 1/0
    pbp_df["Possession"] = pd.to_numeric(pbp_df["Possession"])  # Convert Possession to 1/0
    # Only keep numeric data (non-numeric columns will be stripped out later
    # in pre-processing)
    id_columns = ["Player", "Year", "Week", "Team", "Opponent", "Position", "Elapsed Time"]
    pbp_numeric_columns = [
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
    boxscore_numeric_columns = [
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
    pbp_df = pbp_df.sort_values(
        by=["Year", "Week", "Team", "Player"], ascending=[True, True, True, True]
    )

    # Match inputs (pbp data) to outputs (boxscore data) by index (give each
    # input and corresponding output the same index in their df)
    bs_matched_df = (
        boxscore_df.set_index(["Player", "Year", "Week"])
        .loc[pbp_df.set_index(["Player", "Year", "Week"]).index]
        .reset_index()
    )

    # Strip out non-numeric columns
    pbp_data_stripped = pbp_df[pbp_numeric_columns]
    boxscore_data_stripped = bs_matched_df[boxscore_numeric_columns]

    # Keep identifying info in a separate dataframe
    id_df = pbp_df[id_columns]

    # Normalize stats data so that all numbers are between 0 and 1
    pbp_df_norm = pd.DataFrame()
    bs_df_norm = pd.DataFrame()
    for raw_df, norm_df in zip(
        [pbp_data_stripped, boxscore_data_stripped], [pbp_df_norm, bs_df_norm]
    ):
        for col in raw_df.columns:
            norm_df.loc[:, col] = normalize_stat(raw_df.loc[:, col])

    # Encode each non-numeric, relevant field (Player, Team, Position) in a
    # "word bank":
    fields = ["Position", "Player", "Team", "Opponent"]
    for field in fields:
        word_bank = id_df[field].unique()
        word_bank.sort()
        print(f"{len(word_bank)} unique {field}s")
        df = pd.DataFrame(columns=field + "=" + word_bank)
        for entry in word_bank:
            df[field + "=" + entry] = (id_df[field] == entry).astype(int)
        pbp_df_norm = pd.concat((pbp_df_norm, df), axis=1)

    print(id_df)

    # Save data
    if save_data:
        pbp_df_norm.to_csv("data/for_nn/pbp_data_to_nn.csv", index=False)
        bs_df_norm.to_csv("data/for_nn/boxscore_data_to_nn.csv", index=False)
        id_df.to_csv("data/for_nn/data_ids.csv", index=False)

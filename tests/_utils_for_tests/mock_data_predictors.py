import numpy as np
import pandas as pd

# DataFrames used to create a StatsDataset object

id_df = pd.DataFrame(
    data=[
        ["00-0033699", "EkelAu00", 4663.0, "Austin Ekeler", 2024, 1, "WAS", "TB", "RB", 0],
        ["00-0033699", "EkelAu00", 4663.0, "Austin Ekeler", 2024, 2, "WAS", "NYG", "RB", 0],
        ["00-0033699", "EkelAu00", 4663.0, "Austin Ekeler", 2024, 3, "WAS", "CIN", "RB", 0],
        ["00-0033699", "EkelAu00", 4663.0, "Austin Ekeler", 2024, 5, "WAS", "CLE", "RB", 0],
        ["00-0039910", "DaniJa02", 11566.0, "Jayden Daniels", 2024, 3, "WAS", "CIN", "QB", 0],
        ["00-0030061", "ErtzZa00", 1339.0, "Zach Ertz", 2023, 4, "ARI", "SF", "TE", 0],
        ["00-0030061", "ErtzZa00", 1339.0, "Zach Ertz", 2023, 5, "ARI", "CIN", "TE", 0],
        ["00-0030061", "ErtzZa00", 1339.0, "Zach Ertz", 2023, 6, "ARI", "LA", "TE", 0],
        ["00-0030061", "ErtzZa00", 1339.0, "Zach Ertz", 2024, 5, "WAS", "CLE", "TE", 0],
    ],
    columns=["Player ID", "pfr_id", "sleeper_id", "Player Name", "Year", "Week", "Team", "Opponent", "Position", "Elapsed Time"],
)
pbp_df = pd.DataFrame(
    data=[
        [0.0, 0.65, 0.05, 0.05, 0.05],
        [0.0, 0.35, 0.05, 0.05, 0.05],
        [0.0, 0.65, 0.05, 0.05, 0.05],
        [0.0, 0.65, 0.05, 0.05, 0.05],
        [0.0, 0.65, 0.05, 0.05, 0.05],
        [0.0, 0.35, 0.05, 0.05, 0.05],
        [0.0, 0.35, 0.05, 0.05, 0.05],
        [0.0, 0.35, 0.05, 0.05, 0.05],
        [0.0, 0.65, 0.05, 0.05, 0.05],
    ],
    columns=["Elapsed Time", "Field Position", "Pass Yds", "Rush Yds", "Rec Yds"],
)
bs_df = pd.DataFrame(
    data=[
        [0.05, 0.06, 0.1],
        [0.05, 0.08, 0.09],
        [0.05, 0.08, 0.07],
        [0.05, 0.11, 0.08],
        [0.29, 0.08, 0.05],
        [0.05, 0.05, 0.1],
        [0.05, 0.05, 0.06],
        [0.05, 0.05, 0.07],
        [0.05, 0.05, 0.06],
    ],
    columns=["Pass Yds", "Rush Yds", "Rec Yds"],
)

# LastNPredictor variables
# Must be hard-coded by test dev whenever the above data changes
map_to_last_game = pd.Series(
    data=[
        np.nan,
        0,
        1,
        2,
        np.nan,
        np.nan,
        5,
        6,
        7,
    ],
)

# Must be hard-coded by test dev whenever the above data changes
map_to_second_to_last_game = pd.Series(
    data=[
        np.nan,
        np.nan,
        0,
        1,
        np.nan,
        np.nan,
        np.nan,
        5,
        6,
    ],
)

# SleeperPredictor variables
# Must be hard-coded by test dev whenever the above data changes
expected_predicts_sleeper = pd.DataFrame(
    data=[
        [0.000000, 30.490000, 21.670000, 5.2160],
        [0.000000, 23.410000, 19.900000, 4.3310],
        [0.000000, 22.030001, 18.320000, 4.0350],
        [0.000000, 25.260000, 19.260000, 4.4520],
        [187.009995, 48.980000, 0.000000, 12.3784],
        [0.000000, 0.000000, 28.330000, 2.8330],
        [0.000000, 0.000000, 33.000000, 3.3000],
        [0.000000, 0.000000, 37.549999, 3.7550],
        [0.000000, 0.000000, 26.750000, 2.6750],
    ],
    columns=["Pass Yds", "Rush Yds", "Rec Yds", "Fantasy Points"],
)
# NeuralNetPredictor variables
# Updated automatically by script below
pbp_df_neural_net = pd.DataFrame(
    data=[
        [0.0, 0.65, 0.05, 0.05, 0.05, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.35, 0.05, 0.05, 0.05, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.65, 0.05, 0.05, 0.05, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.65, 0.05, 0.05, 0.05, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.65, 0.05, 0.05, 0.05, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.35, 0.05, 0.05, 0.05, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.35, 0.05, 0.05, 0.05, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.35, 0.05, 0.05, 0.05, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.65, 0.05, 0.05, 0.05, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ],
    columns=[
        "Elapsed Time",
        "Field Position",
        "Pass Yds",
        "Rush Yds",
        "Rec Yds",
        "Position_QB",
        "Position_RB",
        "Position_TE",
        "Player ID_00-0030061",
        "Player ID_00-0033699",
        "Player ID_00-0039910",
        "Team_ARI",
        "Team_WAS",
        "Opponent_CIN",
        "Opponent_CLE",
        "Opponent_LA",
        "Opponent_NYG",
        "Opponent_SF",
        "Opponent_TB",
    ],
)

expected_predicts_neural_net = pd.DataFrame(
    data=[
        [381.069216, 524.537582, 460.776528, 113.774180],
        [379.634183, 527.291364, 450.263154, 112.940819],
        [381.788611, 523.347210, 461.025944, 113.708860],
        [385.657046, 520.673186, 459.607309, 113.454331],
        [415.720430, 527.252086, 426.521301, 112.006156],
        [412.847868, 497.221895, 419.486123, 108.184716],
        [416.935874, 493.053142, 414.505640, 107.433313],
        [414.013959, 495.719953, 417.809841, 107.913538],
        [406.506155, 508.339573, 436.559833, 110.750187],
    ],
    columns=["Pass Yds", "Rush Yds", "Rec Yds", "Fantasy Points"],
)


# Hacky way to write a new dataset that can be hardcoded above
if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.getcwd() + "/FantasyProjections")
    from config.player_id_config import PRIMARY_PLAYER_ID
    from misc.dataset import StatsDataset

    data_folder = "data/"
    source_folder = data_folder + "to_nn/"
    misc_folder = data_folder + "misc/"
    id_df = pd.read_csv(source_folder + "data_ids.csv")
    pbp_df = pd.read_csv(
        source_folder + "midgame_data_to_nn.csv",
        usecols=["Elapsed Time", "Field Position", "Pass Yds", "Rush Yds", "Rec Yds"],
    )
    bs_df = pd.read_csv(source_folder + "final_stats_to_nn.csv", usecols=["Pass Yds", "Rush Yds", "Rec Yds"])

    # Convert player names to IDs
    master_id_list = pd.read_csv(misc_folder + "player_ids.csv")

    def name_to_id(name):
        return master_id_list.set_index("Player Name").loc[name, PRIMARY_PLAYER_ID]

    all_data = StatsDataset(name="all data", id_df=id_df, pbp_df=pbp_df, boxscore_df=bs_df)
    slice1 = all_data.slice_by_criteria(
        inplace=False,
        player_ids=[name_to_id("Austin Ekeler")],
        years=[2024],
        weeks=[1, 2, 3, 4, 5],
        elapsed_time=[0],
    )
    slice2 = all_data.slice_by_criteria(
        inplace=False,
        player_ids=[name_to_id("Jayden Daniels")],
        years=[2024],
        weeks=[3],
        elapsed_time=[0],
    )
    slice3 = all_data.slice_by_criteria(
        inplace=False,
        player_ids=[name_to_id("Zach Ertz")],
        years=[2023],
        weeks=[4, 5, 6],
        elapsed_time=[0],
    )
    slice4 = all_data.slice_by_criteria(
        inplace=False,
        player_ids=[name_to_id("Zach Ertz")],
        years=[2024],
        weeks=[5],
        elapsed_time=[0],
    )
    unittest_dataset = slice1.copy()
    unittest_dataset.concat(slice2)
    unittest_dataset.concat(slice3)
    unittest_dataset.concat(slice4)

    def dataframe_to_hardcoded_string(df_to_stringify, name):
        # Round numeric data to 2 digits after decimal
        df_to_stringify = df_to_stringify.round(2)
        # Format data into nested list
        data_string = ""
        for _, row in df_to_stringify.iterrows():
            data_string += f"{row.values.tolist()},\n"
        # Columns
        column_string = f"{df_to_stringify.columns.tolist()}"
        # Full string
        string = f"{name} = pd.DataFrame(data=[{data_string}], columns={column_string})\n"
        return string

    def build_neural_net_input(id_df, pbp_data, pbp_columns):
        pbp_df = pd.DataFrame(data=pbp_data, columns=pbp_columns)
        # One-Hot Encode each non-numeric, relevant pbp field (Player, Team, Position):
        fields = ["Position", "Player ID", "Team", "Opponent"]
        encoded_fields_df = pd.get_dummies(id_df[fields], columns=fields, dtype=int)
        pbp_df = pd.concat((pbp_df, encoded_fields_df), axis=1)
        return pbp_df

    nn_pbp_df = build_neural_net_input(unittest_dataset.id_data, unittest_dataset.x_data, unittest_dataset.x_data_columns)

    with open("test.txt", "w") as file:
        for data, columns, name in zip(
            [unittest_dataset.id_data, unittest_dataset.x_data, unittest_dataset.y_data, nn_pbp_df],
            [[], unittest_dataset.x_data_columns, unittest_dataset.y_data_columns, []],
            ["id_df", "pbp_df", "bs_df", "pbp_df_neural_net"],
        ):
            df_to_write = pd.DataFrame(data=data, columns=columns) if columns else data

            dataframe_string = dataframe_to_hardcoded_string(df_to_write, name)
            print(dataframe_string)
            file.write(dataframe_string)

    print("Done")

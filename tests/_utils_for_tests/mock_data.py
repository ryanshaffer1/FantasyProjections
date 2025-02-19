import pandas as pd

# DataFrames used to create a StatsDataset object

id_df = pd.DataFrame(
    data=[
        ["00-0033699", "EkelAu00", 4663.0, "Austin Ekeler", 2024, 1, "WAS", "TB", "RB", 0],
        ["00-0033699", "EkelAu00", 4663.0, "Austin Ekeler", 2024, 1, "WAS", "TB", "RB", 1],
        ["00-0033699", "EkelAu00", 4663.0, "Austin Ekeler", 2024, 1, "WAS", "TB", "RB", 2],
        ["00-0033699", "EkelAu00", 4663.0, "Austin Ekeler", 2024, 1, "WAS", "TB", "RB", 3],
        ["00-0035208", "ZaccOl01", 6271.0, "Olamide Zaccheaus", 2024, 4, "WAS", "ARI", "WR", 22],
        ["00-0035208", "ZaccOl01", 6271.0, "Olamide Zaccheaus", 2024, 4, "WAS", "ARI", "WR", 23],
        ["00-0035208", "ZaccOl01", 6271.0, "Olamide Zaccheaus", 2024, 4, "WAS", "ARI", "WR", 24],
        ["00-0039910", "DaniJa02", 11566.0, "Jayden Daniels", 2024, 3, "WAS", "CIN", "QB", 18],
        ["00-0030061", "ErtzZa00", 1339.0, "Zach Ertz", 2023, 5, "ARI", "CIN", "TE", 13],
        ["00-0030061", "ErtzZa00", 1339.0, "Zach Ertz", 2024, 5, "WAS", "CLE", "TE", 53],
    ],
    columns=["Player ID", "pfr_id", "sleeper_id", "Player Name", "Year", "Week", "Team", "Opponent", "Position", "Elapsed Time"],
)
pbp_df = pd.DataFrame(
    data=[
        [0.0, 0.65, 0.05, 0.05, 0.05],
        [0.02, 0.34, 0.05, 0.05, 0.05],
        [0.03, 0.55, 0.05, 0.05, 0.05],
        [0.05, 0.6, 0.05, 0.05, 0.05],
        [0.37, 0.71, 0.05, 0.05, 0.08],
        [0.38, 0.55, 0.05, 0.05, 0.08],
        [0.4, 0.18, 0.05, 0.05, 0.08],
        [0.3, 0.35, 0.11, 0.06, 0.05],
        [0.22, 0.78, 0.05, 0.05, 0.05],
        [0.88, 0.9, 0.05, 0.05, 0.06],
    ],
    columns=["Elapsed Time", "Field Position", "Pass Yds", "Rush Yds", "Rec Yds"],
)
bs_df = pd.DataFrame(
    data=[
        [0.05, 0.06, 0.1],
        [0.05, 0.06, 0.1],
        [0.05, 0.06, 0.1],
        [0.05, 0.06, 0.1],
        [0.05, 0.05, 0.13],
        [0.05, 0.05, 0.13],
        [0.05, 0.05, 0.13],
        [0.29, 0.08, 0.05],
        [0.05, 0.05, 0.06],
        [0.05, 0.05, 0.06],
    ],
    columns=["Pass Yds", "Rush Yds", "Rec Yds"],
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
        weeks=[1],
        elapsed_time=[0, 1, 2, 3],
    )
    slice2 = all_data.slice_by_criteria(
        inplace=False,
        player_ids=[name_to_id("Olamide Zaccheaus")],
        years=[2024],
        weeks=[4],
        elapsed_time=[22, 23, 24],
    )
    slice3 = all_data.slice_by_criteria(
        inplace=False,
        player_ids=[name_to_id("Jayden Daniels")],
        years=[2024],
        weeks=[3],
        elapsed_time=[18],
    )
    slice4 = all_data.slice_by_criteria(
        inplace=False,
        player_ids=[name_to_id("Zach Ertz")],
        years=[2023],
        weeks=[5],
        elapsed_time=[13],
    )
    slice5 = all_data.slice_by_criteria(
        inplace=False,
        player_ids=[name_to_id("Zach Ertz")],
        years=[2024],
        weeks=[5],
        elapsed_time=[53],
    )
    unittest_dataset = slice1.copy()
    unittest_dataset.concat(slice2)
    unittest_dataset.concat(slice3)
    unittest_dataset.concat(slice4)
    unittest_dataset.concat(slice5)

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

    with open("test.txt", "w") as file:
        for data, columns, name in zip(
            [unittest_dataset.id_data, unittest_dataset.x_data, unittest_dataset.y_data],
            [[], unittest_dataset.x_data_columns, unittest_dataset.y_data_columns],
            ["id_df", "pbp_df", "bs_df"],
        ):
            df_to_write = pd.DataFrame(data=data, columns=columns) if columns else data

            dataframe_string = dataframe_to_hardcoded_string(df_to_write, name)
            print(dataframe_string)
            file.write(dataframe_string)

    print("Done")

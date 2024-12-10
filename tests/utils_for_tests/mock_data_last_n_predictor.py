import numpy as np
import pandas as pd

# DataFrames used to create a StatsDataset object

id_df = pd.DataFrame(data=[['Austin Ekeler', 2024, 1, 'WAS', 'TB', 'RB', 0],
                            ['Austin Ekeler', 2024, 2, 'WAS', 'NYG', 'RB', 0],
                            ['Austin Ekeler', 2024, 3, 'WAS', 'CIN', 'RB', 0],
                            ['Austin Ekeler', 2024, 5, 'WAS', 'CLE', 'RB', 0],
                            ['Jayden Daniels', 2024, 3, 'WAS', 'CIN', 'QB', 0],
                            ['Zach Ertz', 2023, 4, 'ARI', 'SF', 'TE', 0],
                            ['Zach Ertz', 2023, 5, 'ARI', 'CIN', 'TE', 0],
                            ['Zach Ertz', 2023, 6, 'ARI', 'LA', 'TE', 0],
                            ['Zach Ertz', 2024, 5, 'WAS', 'CLE', 'TE', 0],
                            ], columns=['Player', 'Year', 'Week', 'Team', 'Opponent', 'Position', 'Elapsed Time'])
pbp_df = pd.DataFrame(data=[[0.0, 0.65, 0.05, 0.05, 0.05],
                            [0.0, 0.35, 0.05, 0.05, 0.05],
                            [0.0, 0.65, 0.05, 0.05, 0.05],
                            [0.0, 0.65, 0.05, 0.05, 0.05],
                            [0.0, 0.65, 0.05, 0.05, 0.05],
                            [0.0, 0.35, 0.05, 0.05, 0.05],
                            [0.0, 0.35, 0.05, 0.05, 0.05],
                            [0.0, 0.35, 0.05, 0.05, 0.05],
                            [0.0, 0.65, 0.05, 0.05, 0.05],
                            ], columns=['Elapsed Time', 'Field Position', 'Pass Yds', 'Rush Yds', 'Rec Yds'])
bs_df = pd.DataFrame(data=[[0.05, 0.06, 0.1],
                            [0.05, 0.08, 0.09],
                            [0.05, 0.08, 0.07],
                            [0.05, 0.11, 0.08],
                            [0.29, 0.08, 0.05],
                            [0.05, 0.05, 0.1],
                            [0.05, 0.05, 0.06],
                            [0.05, 0.05, 0.07],
                            [0.05, 0.05, 0.06],
                            ], columns=['Pass Yds', 'Rush Yds', 'Rec Yds'])

map_to_last_game = pd.Series(data=[
    np.nan,
    0,
    1,
    2,
    np.nan,
    np.nan,
    5,
    6,
    7
])

map_to_second_to_last_game = pd.Series(data=[
    np.nan,
    np.nan,
    0,
    1,
    np.nan,
    np.nan,
    np.nan,
    5,
    6
])


# Hacky way to write a new dataset that can be hardcoded above
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd()+'/FantasyProjections')
    from misc.dataset import StatsDataset
    
    source_folder = 'C:/Users/rshaf/OneDrive/Documents/Projects/FantasyProjections/data2/to_nn/'
    id_df = pd.read_csv(source_folder+'data_ids.csv')
    pbp_df = pd.read_csv(source_folder+'midgame_data_to_nn.csv', usecols=['Elapsed Time','Field Position', 'Pass Yds', 'Rush Yds','Rec Yds'])
    bs_df = pd.read_csv(source_folder+'final_stats_to_nn.csv', usecols=['Pass Yds', 'Rush Yds','Rec Yds'])

    all_data = StatsDataset(name='all data', pbp_df=pbp_df, boxscore_df=bs_df, id_df=id_df)
    slice1 = all_data.slice_by_criteria(inplace=False, players=['Austin Ekeler'], years=[2024], weeks=[1,2,3,4,5], elapsed_time=[0])
    slice2 = all_data.slice_by_criteria(inplace=False, players=['Jayden Daniels'], years=[2024], weeks=[3], elapsed_time=[0])
    slice3 = all_data.slice_by_criteria(inplace=False, players=['Zach Ertz'], years=[2023], weeks=[4,5,6], elapsed_time=[0])
    slice4 = all_data.slice_by_criteria(inplace=False, players=['Zach Ertz'], years=[2024], weeks=[5], elapsed_time=[0])
    unittest_dataset = slice1.copy()
    unittest_dataset.concat(slice2)
    unittest_dataset.concat(slice3)
    unittest_dataset.concat(slice4)
    
    def dataframe_to_hardcoded_string(df, name):
        # Round numeric data to 2 digits after decimal
        df = df.round(2)
        # Format data into nested list
        data_string = ''
        for _, row in df.iterrows():
            data_string += f'{row.values.tolist()},\n'
        # Columns
        column_string = f'{df.columns.tolist()}'         
        # Full string
        string = f'{name} = pd.DataFrame(data=[{data_string}], columns={column_string})\n'
        return string

    with open('test.txt','w') as file:
        for df, name in zip([unittest_dataset.id_data, unittest_dataset.x_df, unittest_dataset.y_df],['id_df','pbp_df','bs_df']):
            dataframe_string = dataframe_to_hardcoded_string(df, name)
            print(dataframe_string)
            file.write(dataframe_string)
    
    print('Done')

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

# LastNPredictor variables
# Must be hard-coded by test dev whenever the above data changes
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

# Must be hard-coded by test dev whenever the above data changes
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

# SleeperPredictor variables
# Must be hard-coded by test dev whenever the above data changes
expected_predicts_sleeper = pd.DataFrame(data=[
                                    [0.000000, 30.490000, 21.670000, 5.2160],
                                    [0.000000, 23.410000, 19.900000, 4.3310],
                                    [0.000000, 22.030001, 18.320000, 4.0350],
                                    [0.000000, 25.260000, 19.260000, 4.4520],
                                    [187.009995, 48.980000, 0.000000, 12.3784],
                                    [0.000000, 0.000000, 28.330000, 2.8330],
                                    [0.000000, 0.000000, 33.000000, 3.3000],
                                    [0.000000, 0.000000, 37.549999, 3.7550],
                                    [0.000000, 0.000000, 26.750000, 2.6750]],
                                 columns=['Pass Yds', 'Rush Yds', 'Rec Yds', 'Fantasy Points'])
# NeuralNetPredictor variables
# Updated automatically by script below
pbp_df_neural_net = pd.DataFrame(data=[[0.0, 0.65, 0.05, 0.05, 0.05, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                        [0.0, 0.35, 0.05, 0.05, 0.05, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.65, 0.05, 0.05, 0.05, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.65, 0.05, 0.05, 0.05, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.65, 0.05, 0.05, 0.05, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.35, 0.05, 0.05, 0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.35, 0.05, 0.05, 0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.35, 0.05, 0.05, 0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.65, 0.05, 0.05, 0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                        ], columns=['Elapsed Time', 'Field Position', 'Pass Yds', 'Rush Yds', 'Rec Yds', 
                                                    'Position=QB', 'Position=RB', 'Position=TE', 
                                                    'Player=Austin Ekeler', 'Player=Jayden Daniels', 'Player=Zach Ertz', 
                                                    'Team=ARI', 'Team=WAS', 
                                                    'Opponent=CIN', 'Opponent=CLE', 'Opponent=LA', 'Opponent=NYG', 'Opponent=SF', 'Opponent=TB'])

expected_predicts_neural_net = pd.DataFrame(data=[
                                                [376.053169, 518.768640, 452.056274, 112.124618],
                                                [375.168535, 521.017497, 442.399019, 111.348393],
                                                [377.565150, 518.032830, 453.077894, 112.213678],
                                                [383.223504, 516.409539, 453.586681, 112.328562],
                                                [417.155152, 529.886314, 427.540419, 112.428879],
                                                [414.464179, 500.293094, 425.109524, 109.118829],
                                                [420.885235, 496.811786, 420.228070, 108.539395],
                                                [416.170914, 499.182728, 423.181051, 108.883214],
                                                [408.103555, 510.734709, 439.930292, 111.390642]], 
                                            columns=['Pass Yds', 'Rush Yds', 'Rec Yds', 'Fantasy Points'])

# Hacky way to write a new dataset that can be hardcoded above
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd()+'/FantasyProjections')
    from misc.dataset import StatsDataset
    from data_pipeline.preprocess_nn_data import add_word_bank_to_df
    
    source_folder = 'C:/Users/rshaf/OneDrive/Documents/Projects/FantasyProjections/data2/to_nn/'
    id_df = pd.read_csv(source_folder+'data_ids.csv')
    pbp_df = pd.read_csv(source_folder+'midgame_data_to_nn.csv', usecols=['Elapsed Time','Field Position', 'Pass Yds', 'Rush Yds','Rec Yds'])
    bs_df = pd.read_csv(source_folder+'final_stats_to_nn.csv', usecols=['Pass Yds', 'Rush Yds','Rec Yds'])

    all_data = StatsDataset(name='all data', id_df=id_df, pbp_df=pbp_df, boxscore_df=bs_df)
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

    def build_neural_net_input(id_df, pbp_data, pbp_columns):
        pbp_df = pd.DataFrame(data=pbp_data, columns=pbp_columns)
        # Encode each non-numeric, relevant pbp field (Player, Team, Position) in a "word bank":
        fields = ["Position", "Player", "Team", "Opponent"]
        for field in fields:
            word_bank_df = add_word_bank_to_df(field, id_df)
            pbp_df = pd.concat((pbp_df, word_bank_df), axis=1)
        return pbp_df

    nn_pbp_df = build_neural_net_input(unittest_dataset.id_data, unittest_dataset.x_data, unittest_dataset.x_data_columns)

    with open('test.txt','w') as file:
        for data, columns, name in zip([unittest_dataset.id_data, unittest_dataset.x_data, unittest_dataset.y_data, nn_pbp_df],
                                       [[],unittest_dataset.x_data_columns,unittest_dataset.y_data_columns,[]],
                                       ['id_df','pbp_df','bs_df', 'pbp_df_neural_net']):
            if columns:
                df = pd.DataFrame(data=data,columns=columns)
            else:
                df = data

            dataframe_string = dataframe_to_hardcoded_string(df, name)
            print(dataframe_string)
            file.write(dataframe_string)
    
    
    print('Done')

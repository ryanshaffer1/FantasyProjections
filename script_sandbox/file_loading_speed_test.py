import time
import pandas as pd
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal

def wrapper(func, csv_file, parquet_file):
    df = func(csv_file, parquet_file)
    return df

def compare_dfs(df1,df2):
    try:
        assert_frame_equal(df1,df2, check_names=True)
        return True
    except AssertionError:
        return False

def print_comparison_table(dfs):
    dfs = clean_dfs(dfs)
    # Build up nested list
    table = []
    num_dfs = len(dfs)
    for i in range(num_dfs):
        row = []
        for j in range(i+1):
            row.append(compare_dfs(dfs[i],dfs[j]))
        table.append(row)

    # Print table
    print('Equality Comparisons:')
    num_rows = len(table[-1])
    print(('\t'+'{}\t'*num_rows).format(*range(1,num_rows+1)))
    for i, row in enumerate(table):
        num_cols = len(row)
        print(('{}\t'*(num_cols+1)).format(i+1,*row))

def clean_dfs(dfs):
    for i,df in enumerate(dfs):
        dfs[i] = df.rename(columns={'Unnamed: 0':''})
    return dfs

def function1(csv_file, parquet_file):
    df = pd.read_csv(csv_file)
    return df

def function2(csv_file, parquet_file):
    df = pd.read_csv(csv_file,engine='pyarrow')
    return df

def function3(csv_file, parquet_file):
    df = pd.read_parquet(parquet_file,engine='pyarrow')
    return df

def function4(csv_file, parquet_file):
    df = pq.read_table(parquet_file).to_pandas()
    return df

funcs = [function1,function2,function3,function4]

YEARS = []
# YEARS = range(2021, 2025)

ONLINE_DATA_SOURCE = 'https://github.com/nflverse/nflverse-data/releases/download/'
online_file_paths = {'pbp': ONLINE_DATA_SOURCE + 'pbp/play_by_play_{0}',
                     'roster': ONLINE_DATA_SOURCE + 'weekly_rosters/roster_weekly_{0}'}

csv_filepath = 'C:/Users/rshaf/OneDrive/Documents/Projects/FantasyProjections/data2/inputs/play_by_play/'
parquet_filepath = 'C:/Users/rshaf/OneDrive/Documents/Projects/FantasyProjections/data2/inputs/test/'
filename = 'play_by_play_{0}'


for year in YEARS:
    print(f'------ {year} ------')

    # Filenames
    csv_file = csv_filepath + filename.format(year) + '.csv'
    parquet_file = parquet_filepath + filename.format(year) + '.parquet'

    dfs = []
    times = [time.perf_counter()]
    for func in funcs:
        # Load files and time each method
        df = wrapper(func,csv_file, parquet_file)
        dfs.append(df)
        times.append(time.perf_counter())

    # Equality check
    # identical = (csv_df == pqt_df)

    # Print results
    for i in range(len(funcs)):
        print(f'function{i+1}: {times[i+1]-times[i]}')

print('---- Pre-processed data ----')
PBP_DATAFILE = 'data2/to_nn/midgame_data_to_nn.csv'
BOXSCORE_DATAFILE = 'data2/to_nn/final_stats_to_nn.csv'
ID_DATAFILE = 'data2/to_nn/data_ids.csv'
PBP_PARQUET = 'data2/inputs/test/midgame_data_to_nn.parquet'
BS_PARQUET = 'data2/inputs/test/final_stats_to_nn.parquet'
ID_PARQUET = 'data2/inputs/test/data_ids.parquet'

for csv_file, parquet_file in zip([PBP_DATAFILE, BOXSCORE_DATAFILE, ID_DATAFILE],[PBP_PARQUET, BS_PARQUET, ID_PARQUET]):
    print(csv_file)
    dfs = []
    times = [time.perf_counter()]
    for func in funcs:
        # Load files and time each method
        df = wrapper(func,csv_file, parquet_file)
        dfs.append(df)
        times.append(time.perf_counter())

    # Print results
    for i in range(len(funcs)):
        print(f'function{i+1}: {times[i+1]-times[i]}')
    # print(f'{year}, Identical = /n{identical}')
    print_comparison_table(dfs)
    
    
# dfs[0].to_parquet(parquet_file, engine='pyarrow')

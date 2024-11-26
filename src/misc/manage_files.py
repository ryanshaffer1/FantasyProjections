import os
import pandas as pd

def create_folders(folders):
    # Handle case of single folder being passed
    if isinstance(folders,str):
        folders = [folders]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f'Created folder {folder}')


def collect_input_dfs(years, weeks, local_file_paths, online_file_paths, online_avail=False):
    # Handle single year being input
    if not hasattr(years,'__iter__'):
        years = [years]

    # Create local folders if they do not exist
    local_folders = ['/'.join(val.split('/')[:-1])+'/' for val in local_file_paths.values()]
    create_folders(local_folders)

    # Load files one year at a time
    all_dfs = []
    for year in years:
        # Load local files
        try:
            yearly_dfs = tuple(pd.read_csv(local_file_paths[key].format(year), low_memory=False) for key in local_file_paths)
            # Check if local files contain all weeks (checks all df's together)
            weeks_present = [all((any(df['week']==week) for df in yearly_dfs)) for week in weeks]
        except FileNotFoundError:
            weeks_present = [False]

        # Download files from online and save locally (updates all df's together)
        if not all(weeks_present):
            yearly_dfs = ()
            if online_avail:
                for name in local_file_paths:
                    # Read from online filepath
                    print(f'Downloading {name} from {online_file_paths[name].format(year)}')
                    df = pd.read_csv(online_file_paths[name].format(year), low_memory=False)
                    yearly_dfs = yearly_dfs + (df,)
                    # Save locally
                    df.to_csv(local_file_paths[name].format(year))
                    print(f'Saved {name} to {local_file_paths[name].format(year)}')
            else:
                print('Warning! Not all weeks are present in the dfs, and could not download from online')

        all_dfs.append(yearly_dfs)

    return all_dfs

def collect_roster_filter(filter_roster, update_filter, roster_filter_file):
    if filter_roster and not update_filter:
        try:
            filter_df = pd.read_csv(roster_filter_file)
            load_success = True
        except FileNotFoundError:
            filter_df = None
            load_success = False
    else:
        filter_df = None
        load_success = False

    return filter_df, load_success

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


def collect_input_dfs(year, weeks, local_file_paths, online_file_paths, online_avail=False):
    # Create local folders if they do not exist
    local_folders = ['/'.join(val.split('/')[:-1])+'/' for val in local_file_paths.values()]
    create_folders(local_folders)

    # Load local files
    try:
        dfs = tuple(pd.read_csv(local_file_paths[key].format(year), low_memory=False) for key in local_file_paths)
        # Check if local files contain all weeks (checks all df's together)
        weeks_present = [all((any(df['week']==week) for df in dfs)) for week in weeks]
    except FileNotFoundError:
        weeks_present = [False]

    # Download files from online and save locally (updates all df's together)
    if not all(weeks_present):
        dfs = ()
        if online_avail:
            for name in local_file_paths:
                # Read from online filepath
                print(f'Downloading {name} from {online_file_paths[name].format(year)}')
                df = pd.read_csv(online_file_paths[name].format(year), low_memory=False)
                dfs = dfs + (df,)
                # Save locally
                df.to_csv(local_file_paths[name].format(year))
                print(f'Saved {name} to {local_file_paths[name].format(year)}')
        else:
            print('Warning! Not all weeks are present in the dfs, and could not download from online')

    return dfs

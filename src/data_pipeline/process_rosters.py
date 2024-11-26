import dateutil.parser as dateparse


def process_rosters(all_rosters_df, weeks, filter_df=None):
    # Filter to only the desired weeks
    all_rosters_df = all_rosters_df[all_rosters_df.apply(
        lambda x: x['week'] in weeks, axis=1)]

    # Optionally filter based on subset of desired players
    if filter_df:
        all_rosters_df = all_rosters_df[all_rosters_df.apply(
            lambda x: x['full_name'] in filter_df['Name'].to_list(), axis=1)]

    # Filter to only skill positions
    # Positions currently being tracked for stats
    skill_positions = ['QB', 'RB', 'FB', 'HB', 'WR', 'TE']
    all_rosters_df = all_rosters_df[all_rosters_df.apply(
        lambda x: x['position'] in skill_positions, axis=1)]

    # Filter to only active players
    valid_statuses = ['ACT']
    all_rosters_df = all_rosters_df[all_rosters_df.apply(
        lambda x: x['status'] in valid_statuses, axis=1)]


    # Compute age based on birth date
    all_rosters_df['Age'] = all_rosters_df['season'] - all_rosters_df['birth_date'].apply(
                                parse_birthdate)

    # Trim to just the fields that are useful
    all_rosters_df=all_rosters_df[['team',
    'week',
    'position',
    'jersey_number',
    'full_name',
    'gsis_id',
     'Age']]
    # Reformat
    all_rosters_df=all_rosters_df.rename(columns=
        {
        'team': 'Team',
        'week': 'Week',
        'position': 'Position',
        'jersey_number': 'Number',
        'full_name': 'Name',
        'gsis_id': 'Player ID'
        }).set_index(['Team','Week']).sort_index()

    return all_rosters_df

def parse_birthdate(x):
    try:
        output = dateparse.parse(x).year
    except TypeError:
        output = 2000 # whatever man
    return output

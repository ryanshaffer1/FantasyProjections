"""Contains variables defining the configuration of data used in Fantasy Projections.

    Variables:
"""  # fmt: skip

from config.player_id_config import ALT_PLAYER_IDS

baseline_data_outputs = {
    "id": ["Player ID", *ALT_PLAYER_IDS, "Player Name", "Year", "Week", "Elapsed Time"],
    "midgame": ["Elapsed Time"],
    "final": [],
}

baseline_data_thresholds = {"Elapsed Time": [0, 60]}

baseline_one_hot_columns = ["Player ID"]

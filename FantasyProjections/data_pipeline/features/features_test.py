from config.data_files_config import INPUT_FOLDER, ONLINE_URL_NFLVERSE
from data_pipeline.features import BasicFeatureSet, InjuryFeatureSet

features = [
    BasicFeatureSet(
        "basic",
        sources={
            "online": {
                "pbp": ONLINE_URL_NFLVERSE + "pbp/play_by_play_{0}.csv",
                "roster": ONLINE_URL_NFLVERSE + "weekly_rosters/roster_weekly_{0}.csv",
            },
            "local": {
                "pbp": INPUT_FOLDER + "play_by_play/play_by_play_{0}.csv",
                "roster": INPUT_FOLDER + "rosters/roster_weekly_{0}.csv",
            },
        },
        thresholds={
            "Team Score": [0, 100],
            "Opp Score": [0, 100],
            "Possession": [0, 1],
            "Field Position": [0, 100],
            "Pass Att": [0, 100],
            "Pass Cmp": [0, 100],
            "Pass Yds": [-50, 1000],
            "Pass TD": [0, 8],
            "Int": [0, 8],
            "Rush Att": [0, 100],
            "Rush Yds": [-50, 1000],
            "Rush TD": [0, 8],
            "Rec": [0, 100],
            "Rec Yds": [-50, 1000],
            "Rec TD": [0, 8],
            "Fmb": [0, 8],
            "Age": [0, 60],
            "Site": [0, 1],
            "Team Wins": [0, 18],
            "Team Losses": [0, 18],
            "Team Ties": [0, 18],
            "Opp Wins": [0, 18],
            "Opp Losses": [0, 18],
            "Opp Ties": [0, 18],
        },
    ),
    InjuryFeatureSet(
        "injuries",
        sources={
            "online": ONLINE_URL_NFLVERSE + "injuries/injuries_{0}.csv",
            "local": INPUT_FOLDER + "injuries/injuries_{0}.csv",
        },
        thresholds={
            "injury_status": [0, 1],
        },
    ),
]

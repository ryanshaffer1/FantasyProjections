import pandas as pd
import pytest

# Module under test
from data_pipeline.utils.data_helper_functions import calc_game_time_elapsed


# Test calc_game_time_elapsed
@pytest.mark.skip(reason="Feature not yet implemented")
def test_calc_game_time_elapsed_df_input():
    df_data = pd.DataFrame(
        columns=["qtr", "time"],
        data=[
            ["1", "15:00"],
            ["2", "15:00"],
            ["3", "15:00"],
            ["4", "15:00"],
            ["4", "0:00"],
            ["2", "0:00"],
            ["3", "7:45"],
            ["1", "14:54"],
        ],
    )
    elapsed_times_expected = pd.Series(
        data=[
            0,
            15,
            30,
            45,
            60,
            30,
            37.25,
            0.1,
        ],
    )
    assert calc_game_time_elapsed(df_data) == elapsed_times_expected

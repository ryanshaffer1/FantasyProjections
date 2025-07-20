"""Creates and exports helper functions related to date/time manipulation for NFL stats/odds data.

    Functions:
        week_to_date_range : Returns the start and end dates of a given NFL week.
        date_to_nfl_week : Returns the NFL season/week that contains a given date.
        find_prev_time_index : Finds the most recent time before a given time in a series of times (i.e. the nearest time in the past.)
"""  # fmt: skip

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import dateutil.parser as dateparse
import dateutil.relativedelta as datedelta
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Week starts on WEDNESDAY
WEEK_1_DATESTRS: dict[int, str] = {
    2018: "2018-09-05 00:00:00Z",
    2019: "2019-09-04 00:00:00Z",
    2020: "2020-09-09 00:00:00Z",
    2021: "2021-09-08 00:00:00Z",
    2022: "2022-09-07 00:00:00Z",
    2023: "2023-09-06 00:00:00Z",
    2024: "2024-09-04 00:00:00Z",
}
WEEK_1_DATES: dict[int, dt.datetime] = {key: dateparse.parse(val) for key, val in WEEK_1_DATESTRS.items()}


def week_to_date_range(year: int, week: int) -> list[dt.datetime]:
    """Returns the start and end dates of a given NFL week.

        For convention, it is assumed that the week starts on a Wednesday and ends on a Tuesday.

        Args:
            year (int): NFL season
            week (int): NFL week

        Returns:
            list: List of two datetime objects, corresponding to the start and end dates of the week.

    """  # fmt: skip

    week_1_date = WEEK_1_DATES[year]
    days_from_week_1 = (week - 1) * 7
    week_start_date = week_1_date + datedelta.relativedelta(days=days_from_week_1)
    week_end_date = week_start_date + datedelta.relativedelta(days=6)
    date_range = [week_start_date, week_end_date]

    return date_range


def date_to_nfl_week(date: dt.datetime | str) -> tuple[int, int]:
    """Returns the NFL season/week that contains a given date.

        For convention, it is assumed that the week starts on a Wednesday and ends on a Tuesday.

        Args:
            date (datetime | str): Date

        Returns:
            int: NFL season (year)
            int: NFL week

    """  # fmt: skip

    if isinstance(date, str):
        date = dateparse.parse(date)

    days_delta = np.array([(date - start_date).days for start_date in WEEK_1_DATES.values()])
    year = list(WEEK_1_DATES.keys())[np.argmin(np.where(days_delta < 0, np.inf, days_delta))]
    year_start_date = WEEK_1_DATES[year]
    time_from_start_date = date - year_start_date
    week = (time_from_start_date.days // 7) + 1

    return year, week


def find_prev_time_index(time: float | str, other_times_series: pd.Series) -> int:
    """Finds the most recent time before a given time in a series of times (i.e. the nearest time in the past).

        Args:
            time (float | int | str): Float/int representing an elapsed time (like since a game starting),
                or string in a parseable datetime format representing a UTC time.
            other_times_series (pandas.Series): Series including other times in the same format as time input.
                Note that if time is a float or int, it is assumed that the times in other_time_series are in the index.
                If time is a string, it is assumed that times in other_time_series are the values.

        Returns:
            int: Index of other_times_series that is the most recent time before the input time.

    """  # fmt: skip

    # Time input as an elapsed time float
    if isinstance(time, (float, int)):
        earlier_times = other_times_series.index <= time
        index = np.max(np.where(earlier_times))

    # Time input as a UTC time string
    else:
        time_dt = dateparse.parse(time)

        time_deltas = np.array([(time_dt - dateparse.parse(other_time)) for other_time in other_times_series])
        big_timedelta = dt.timedelta(days=1000)
        index = np.argmin(np.where(time_deltas < dt.timedelta(0), big_timedelta, time_deltas))  # type: ignore[reportArgumentType]

    return int(index)

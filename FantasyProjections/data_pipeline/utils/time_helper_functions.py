import datetime as dt
import dateutil.parser as dateparse
import dateutil.relativedelta as datedelta
import numpy as np

# Week starts on WEDNESDAY
WEEK_1_DATES = {2018: '2018-09-05 00:00:00Z',
                2019: '2019-09-04 00:00:00Z',
                2020: '2020-09-09 00:00:00Z',
                2021: '2021-09-08 00:00:00Z',
                2022: '2022-09-07 00:00:00Z',
                2023: '2023-09-06 00:00:00Z',
                2024: '2024-09-04 00:00:00Z'}
WEEK_1_DATES = {key:dateparse.parse(val) for key, val in WEEK_1_DATES.items()}

def week_to_date_range(year, week):

    week_1_date = WEEK_1_DATES[year]
    days_from_week_1 = (week-1)*7
    week_start_date = week_1_date + datedelta.relativedelta(days=days_from_week_1)
    week_end_date = week_start_date + datedelta.relativedelta(days=6)
    date_range = [week_start_date, week_end_date]

    return date_range


def date_to_nfl_week(date):
    try:
        date = dateparse.parse(date)
    except TypeError:
        # Assume date is already a datetime
        pass

    days_delta = np.array([(date - start_date).days for start_date in WEEK_1_DATES.values()])
    year = list(WEEK_1_DATES.keys())[np.argmin(np.where(days_delta < 0, np.inf, days_delta))]
    year_start_date = WEEK_1_DATES[year]
    time_from_start_date = date - year_start_date
    week = (time_from_start_date.days // 7) + 1

    return year, week

def find_prev_time_index(time, other_times_series):
    # Time input as an elapsed time float
    if isinstance(time, (float, int)):
        earlier_times = other_times_series.index <= time
        index = np.max(np.where(earlier_times))

    # Time input as a UTC time string
    else:
        time = dateparse.parse(time)

        time_deltas = np.array([(time - dateparse.parse(other_time)) for other_time in other_times_series])
        big_timedelta = dt.timedelta(days=1000)
        index = np.argmin(np.where(time_deltas < dt.timedelta(0), big_timedelta, time_deltas))

    return index

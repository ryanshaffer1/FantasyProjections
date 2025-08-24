from __future__ import annotations

from typing import TYPE_CHECKING

from misc.time_helper_functions import calc_weeks_from_epoch

if TYPE_CHECKING:
    from misc.dataset import StatsDataset


class MarchingDataset:
    def __init__(
        self,
        name: str,
        all_data: StatsDataset,
        subsets: list[dict],
        config: list[dict] | None = None,
        march: dict | None = None,
        **_kwargs,
    ):
        self.name = name
        self.all_data = all_data
        self.subsets = subsets
        self.config = config if config is not None else [{}]
        self.march = march if march is not None else {}

        # Convert march start/end to epweeks
        if self.march == {}:
            self.march = {
                "start": min(self.all_data.id_data["EpWeek"]),
                "end": max(self.all_data.id_data["EpWeek"]),
            }
        else:
            self.march["start"] = calc_weeks_from_epoch(self.march["start"]["year"], self.march["start"]["week"])
            self.march["end"] = calc_weeks_from_epoch(self.march["end"]["year"], self.march["end"]["week"])

        # Set dataset state to the start of the march
        self.current_week = self.march["start"]

        # Convert the window of each subset to weeks before the end of the march state
        full_window_min = min(min(subset["window"]) for subset in self.subsets)
        full_window_max = max(max(subset["window"]) for subset in self.subsets)
        window_offset = full_window_max - full_window_min
        for subset in self.subsets:
            # Convert window from a min, max to an enumerated range
            if len(subset["window"]) == 2 and subset["window"][1] - subset["window"][0] > 1:
                subset["window"] = list(range(subset["window"][0], subset["window"][1] + 1))
            # Convert window to be relative to the end of the march
            subset["window"] = [week - window_offset for week in subset["window"]]

        # Create datasets for the start of the march
        self.datasets = self.create_datasets()

        # Track the initial epweek for any subsets with accumulation
        for subset in self.subsets:
            if subset.get("accumulate", False):
                subset["initial_epweek"] = min(subset["config"]["epweeks"])

    def create_datasets(self):
        # Create config listing epweeks for each subset to use at the current march week
        for subset in self.subsets:
            subset_epweeks = [self.current_week + week for week in subset["window"]]
            if subset.get("initial_epweek"):
                # If accumulating, set the epweeks to start from the initial epweek (if initial epweek is before current subset epweeks)
                subset_epweeks = (
                    list(
                        range(subset["initial_epweek"], subset_epweeks[0]),
                    )
                    + subset_epweeks
                )
            subset["config"] = {"epweeks": subset_epweeks}

        # Create StatsDataset objects for each subset
        datasets = []
        for j, subset in enumerate(self.subsets):
            # Slice dataset rows based on input configuration criteria
            for i, configuration in enumerate(self.config):
                configuration.update(subset["config"])
                if i == 0:
                    dataset = self.all_data.slice_by_criteria(inplace=False, **configuration)
                else:
                    dataset.concat(self.all_data.slice_by_criteria(inplace=False, **configuration))
            dataset.name = subset.get("name", f"subset_{j}")
            datasets.append(dataset)

        return datasets

    def advance_week(self, num_weeks: int = 1):
        """Advances the current week of the marching dataset by the specified number of weeks, and updates the datasets accordingly.

            Args:
                num_weeks (int, optional): Number of weeks to advance. Defaults to 1.

        """  # fmt: skip
        # Set the current week based on the advancement, and check it is within the march range
        self.current_week += num_weeks
        if self.current_week > self.march["end"]:
            msg = "Cannot advance week beyond the end of the march."
            raise ValueError(msg)

        # Set dataset windows based on accumulation

        # Update datasets based on current week
        self.datasets = self.create_datasets()

    def set_to_week(self, epweek: int | None = None, year: int | None = None, week: int | None = None):
        """Sets the current week of the marching dataset to the specified epweek or year/week, and updates the datasets accordingly.

            Either epweek or both year and week must be specified.

            Args:
                epweek (int|None, optional): Epweek to set the current week to. Defaults to None.
                year (int|None, optional): Year of the week to set the current week to. Defaults to None.
                week (int|None, optional): Week number within the year to set the current week to. Defaults to None.

        """  # fmt: skip

        # Handle optional inputs and set the new current week
        if epweek is not None:
            new_week = epweek
        elif year is not None and week is not None:
            new_week = calc_weeks_from_epoch(year, week)
        else:
            msg = "Must specify either epweek or both year and week."
            raise ValueError(msg)

        # Check new week is within the march range
        if new_week < self.march["start"] or new_week > self.march["end"]:
            msg = "Cannot set current week outside of the march range."
            raise ValueError(msg)

        self.current_week = new_week
        self.datasets = self.create_datasets()

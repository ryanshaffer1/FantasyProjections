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
        self.all_epweeks = list(range(self.march["start"], self.march["end"] + 1))

        # Set dataset state to the start of the march
        self.current_week = self.march["start"]

        # Convert the window of each subset to weeks before the end of the march state
        max_window_week = max(max(subset["window"]) for subset in self.subsets)
        for subset in self.subsets:
            # Convert window from a min, max to an enumerated range
            if len(subset["window"]) == 2 and subset["window"][1] - subset["window"][0] > 1:  # noqa: PLR2004
                subset["window"] = list(range(subset["window"][0], subset["window"][1] + 1))
            # Convert window to be relative to the end of the march
            subset["window"] = [week - max_window_week for week in subset["window"]]

        self.generate_epweek_lists()

        # Track the initial epweek for any subsets with accumulation
        for subset in self.subsets:
            if subset.get("accumulate", False):
                subset["initial_epweek"] = min(subset["config"]["epweeks"])

    def generate_epweek_lists(self):
        """Generates the list of epweeks for each subset based on the current week and the subset's window.

            This is called during initialization and whenever the current week is updated.
        """  # fmt: skip

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
            if subset.get("config"):
                subset["config"].update({"epweeks": subset_epweeks})
            else:
                subset["config"] = {"epweeks": subset_epweeks}

    def create_datasets(self, pregenerated_config: bool = False):
        if not pregenerated_config:
            # Generate list of weeks to use for each subset
            self.generate_epweek_lists()

        # Create StatsDataset objects for each subset
        datasets = {}
        for j, subset in enumerate(self.subsets):
            # Slice dataset rows based on input configuration criteria
            for i, configuration in enumerate(self.config):
                configuration = configuration.copy()
                configuration.update(subset["config"])
                if i == 0:
                    dataset = self.all_data.slice_by_criteria(inplace=False, **configuration)
                else:
                    dataset.concat(self.all_data.slice_by_criteria(inplace=False, **configuration))
            dataset.name = subset.get("name", f"subset_{j}")
            dataset.manager = self
            datasets[dataset.name] = dataset

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

        # Update datasets based on current week
        datasets = self.create_datasets()

        return datasets

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
        datasets = self.create_datasets()

        return datasets

    def show_full_dataset(self):
        """Generates the full dataset containing all data points within the march range.

            Returns:
                dict: Maps name of each dataset to the corresponding StatsDataset object containing all data points within the march range.

        """  # fmt: skip

        # Create config listing all epweeks for each subset
        for subset in self.subsets:
            all_subset_epweeks = list(
                range(min(subset["window"]) + self.march["start"], max(subset["window"]) + self.march["end"] + 1),
            )
            subset["config"] = {"epweeks": all_subset_epweeks}

        # Generate the full datasets
        datasets = self.create_datasets(pregenerated_config=True)

        # Reset the subset configs
        self.generate_epweek_lists()

        return datasets

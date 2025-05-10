from misc.manage_files import collect_input_dfs


class FeatureSet:
    def __init__(self, name, sources, thresholds):
        self.name = name
        self.sources = sources
        self.columns = thresholds.keys()
        self.thresholds = thresholds
        self.df_dict = {}

    def collect_data(self, year, weeks):
        # Collect data from local or online source

        if isinstance(self.sources["local"], dict):
            # Check if multiple sources are already specified in a dict
            local_file_path = self.sources["local"]
            online_file_path = self.sources["online"]
        else:
            # If one one source, format it into a dict
            local_file_path = {self.name: self.sources["local"]}
            online_file_path = {self.name: self.sources["online"]}
        self.df_dict = collect_input_dfs(year, weeks, local_file_path, online_file_path, online_avail=True)

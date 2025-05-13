from misc.manage_files import collect_input_dfs


class FeatureSet:
    def __init__(self, features, sources):
        self.sources = sources
        self.features = features
        self.thresholds = {feat.name: feat.thresholds for feat in self.features}
        self.df_dict = {}

    def collect_data(self, year, weeks):
        # Collect data from local or online source

        if isinstance(self.sources["local"], dict):
            # Check if multiple sources are already specified in a dict
            local_file_path = self.sources["local"]
            online_file_path = self.sources["online"]
        else:
            # If only one source, format it into a dict
            local_file_path = {"feature_set": self.sources["local"]}
            online_file_path = {"feature_set": self.sources["online"]}
        self.df_dict = collect_input_dfs(year, weeks, local_file_path, online_file_path, online_avail=True)

    def get_features_by_name(self, name):
        """Returns the feature object with the given name."""
        for feature in self.features:
            if feature.name == name:
                return feature
        msg = f"Feature {name} not found in feature set {self.name}."
        raise ValueError(msg)

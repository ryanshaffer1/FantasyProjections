import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sleeper_wrapper import Stats, Players
from nn_helper_functions import end_learning, stats_to_fantasy_points, assign_device, print_model, reformat_sleeper_stats
from nn_plot_functions import plot_grid_search_results
from neural_network import NeuralNetwork
from prediction_result import PredictionResult

class FantasyPredictor():
    def __init__(self,name):
        # # Assign training data, validation data, and test data
        # # If fewer than three datasets are in the list "datasets", the variables within
        # # "datasets" will be assigned to training_data, validation_data, then test_data in
        # # that order. All unassigned variables will be assigned None.
        # (self.training_data,self.validation_data,self.test_data) = datasets + type(datasets)([None])*(3-len(datasets))
        self.name = name

    def gen_prediction_result(self, stat_predicts, stat_truths, eval_data):
        # Generate PredictorResult object
        result = PredictionResult(self, stat_predicts, stat_truths, eval_data)

        # Compute/display average absolute error
        print(f"{self.name} Test Error: Avg. Abs. Fantasy Points Different = {(np.mean(result.avg_diff(absolute=True))):>0.2f}")

        return result

    def eval_truth(self, eval_data):
        stat_truths = stats_to_fantasy_points(
            eval_data.y_data, stat_indices='default', normalized=True)
        return stat_truths

class NeuralNetPredictor(FantasyPredictor):
    def __init__(self,name,nn_settings=None,**kwargs):
        # Initialize FantasyPredictor
        super().__init__(name)

        # Handle optional inputs
        if not nn_settings:
            nn_settings = {}
        self.save_file = kwargs.get('save_file', None)
        self.load_file = kwargs.get('load_file', None)
        self.max_epochs = nn_settings.get('max_epochs',100)
        self.n_epochs_to_stop = nn_settings.get('n_epochs_to_stop',5)

        # Assign attributes
        self.device = assign_device()

        # Initialize model
        if self.load_file:
            self.model = self.load_model(self.load_file,print_loaded_model=True)
            self.optimizer = torch.optim.SGD(self.model.parameters()) # Can modify this to read a saved optimizer
        else:
            self.model = NeuralNetwork().to(self.device)
            self.optimizer = torch.optim.SGD(self.model.parameters())

    def train_and_tune(self,param_tuner,training_data,validation_data):
        param_set = param_tuner.param_set

        # ---------------------
        # Iterate through HyperParameter tuning loop
        # ---------------------
        for tune_layer in range(param_tuner.hyper_tuner_layers):
            print(f'\nOptimization Round {tune_layer+1} of {param_tuner.hyper_tuner_layers}\n-------------------------------')
            # Iterate through all combinations of hyperparameters
            for grid_ind in range(param_set.total_gridpoints):
                # Set and display hyperparameters for current run
                param_set.set_values(grid_ind)
                print(f'\nHP Grid Point {grid_ind+1} of {param_set.total_gridpoints}: -------------------- ')
                for hp in param_set.hyper_parameters:
                    print(f"\t{hp.name} = {hp.value}")

                # Configure data loaders
                train_dataloader = DataLoader(training_data, batch_size=int(param_set.get('mini_batch_size').value), shuffle=False)
                validation_dataloader = DataLoader(validation_data, batch_size=int(validation_data.x_data.shape[0]), shuffle=False)

                # Initialize Neural Network object, configure optimizer, optionally print info on model
                self.model = NeuralNetwork().to(self.device)
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=param_set.get('learning_rate').value,weight_decay=param_set.get('lmbda').value)
                if not param_tuner.optimize_hypers:
                    print_model(self.model)

                # ---------------------
                # Model Training and Validation Testing
                # ---------------------
                # Training/validation loop
                val_perfs = []
                for t in range(self.max_epochs):
                    print(f'Training Epoch {t+1} ------------- ')
                    # Train
                    self.train(train_dataloader, param_set.get('loss_fn').value)

                    # Validation
                    val_result = self.eval_model(eval_dataloader=validation_dataloader)
                    val_perfs.append(np.mean(val_result.avg_diff(absolute=True)))

                    # Check stopping condition
                    stop_training = end_learning(val_perfs,self.n_epochs_to_stop) # Check stopping condition
                    if stop_training:
                        print('Learning has stopped, terminating training process')
                        break

                # Track validation performance for the set of hyperparameters used
                param_tuner.gridpoint_model_perf.append(val_perfs[-1])
                # Save the model if it is the best performing so far
                if grid_ind == np.nanargmin(param_tuner.gridpoint_model_perf):
                    self.save_model()

            # Print some results, determine whether to perform another layer of grid search, and if so, refine the mesh
            param_tuner.next_hp_layer(self,tune_layer)

        # After grid search finishes, plot results
        if len(param_tuner.hyper_tuning_table) > 0 and param_tuner.plot_grid_results:
            plot_grid_search_results(param_tuner.save_file,param_tuner.param_set,variables=('learning_rate','lmbda'))

        # Set the model back to the highest performing config
        self.model = self.load_model(self.save_file,print_loaded_model=False)


    def train(self, dataloader, loss_fn, print_losses=True):
        size = len(dataloader.dataset)
        num_batches = int(np.ceil(size / dataloader.batch_size))
        self.model.train()
        for batch, (x_matrix, y_matrix) in enumerate(dataloader):
            x_matrix, y_matrix = x_matrix.to(self.device), y_matrix.to(self.device)

            # Compute prediction error
            pred = self.model(x_matrix)
            loss = loss_fn(pred, y_matrix)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if print_losses and batch % int(num_batches / 10) == 0:
                loss, current = loss.item(), (batch + 1) * len(x_matrix)
                print(f"\tloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def eval_model(self,eval_data=None,eval_dataloader=None):
        # Create dataloader if only a dataset is passed as input
        if not eval_dataloader:
            eval_dataloader = DataLoader(eval_data, batch_size=int(eval_data.x_data.shape[0]), shuffle=False)

        # Gather all predicted/true outputs for the input dataset
        self.model.eval()
        pred = torch.empty([0, 12])
        y_matrix = torch.empty([0, 12])
        with torch.no_grad():
            for (x_matrix, y_vec) in eval_dataloader:
                pred = torch.cat((pred, self.model(x_matrix)))
                y_matrix = torch.cat((y_matrix, y_vec))

        # Convert outputs into un-normalized statistics/fantasy points
        stat_predicts = stats_to_fantasy_points(pred, stat_indices='default', normalized=True)
        stat_truths = stats_to_fantasy_points(y_matrix, stat_indices='default', normalized=True)
        result = self.gen_prediction_result(stat_predicts, stat_truths, eval_data)

        return result

    def load_model(self, model_file,print_loaded_model=False):
        # Establish shape of the model based on data within file
        state_dict = torch.load(model_file, weights_only=True)
        shape = {
        'players_input': state_dict['embedding_player.0.weight'].shape[1],
        'teams_input': state_dict['embedding_team.0.weight'].shape[1],
        'positions_input': 4,
        'embedding_player': state_dict['embedding_player.0.weight'].shape[0],
        'embedding_team': state_dict['embedding_team.0.weight'].shape[0],
        'embedding_opp': state_dict['embedding_opp.0.weight'].shape[0],
        'linear_stack': state_dict['linear_stack.0.weight'].shape[0],
        'stats_output': state_dict['linear_stack.2.weight'].shape[0],
        }
        shape['stats_input'] = (state_dict['linear_stack.0.weight'].shape[1] -
                            shape['embedding_player'] -
                            shape['embedding_team'] -
                            shape['embedding_opp'] -
                            shape['positions_input']
                            )

        # Create Neural Network model with shape determined above
        model = NeuralNetwork(shape=shape).to(self.device)

        # Load weights/biases into model
        model.load_state_dict(state_dict)

        # Print model
        if print_loaded_model:
            print(f'Loaded model from file {model_file}')
            print_model(model)

        return model

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_file)
        # torch.save(self.optimizer,self.save_file)
        print(f'Saved PyTorch Model State to {self.save_file}')


class SleeperPredictor(FantasyPredictor):
    # Prediction Algorithm: Pulls projections from Sleeper Fantasy. Not sure how they compute their projections!
    # Sleeper API calls documented here:
    # https://github.com/SwapnikKatkoori/sleeper-api-wrapper
    def __init__(self,name,player_dict_file,proj_dict_file,update_players=False):
        # Initialize FantasyPredictor
        super().__init__(name)
        # Files with data from Sleeper
        self.player_dict_file = player_dict_file
        self.proj_dict_file = proj_dict_file
        # Generate dictionary mapping player names to IDs
        if update_players:
            self.player_to_sleeper_id = self.refresh_players()
        else:
            self.player_to_sleeper_id = self.load_players()
        # Initialize attributes defined later (dependent on eval data used)
        self.all_proj_dict = {}


    def refresh_players(self):
        players = Players()
        player_dict = players.get_all_players()

        # Re-organize player dict into dictionary mapping full names to Sleeper player IDs
        # THIS DOESN'T WORK --- MULTIPLE PLAYERS WITH SAME NAME AND DIFFERENT IDS.
        # EX: MIKE WILLIAMS
        player_to_sleeper_id = {}
        for player in player_dict:
            sleeper_id = player
            player_name = player_dict[player].get('full_name', None)
            if player_name:
                player_to_sleeper_id[player_name] = sleeper_id
            else:
                print(f'Warning: {player} not added to player dictionary')

        # Save player dictionary to JSON file for use next time
        with open(self.player_dict_file, 'w', encoding='utf-8') as file:
            json.dump(player_to_sleeper_id, file)

        return player_to_sleeper_id

    def load_players(self):
        with open(self.player_dict_file, 'r', encoding='utf-8') as file:
            player_to_sleeper_id = json.load(file)

        return player_to_sleeper_id

    def remove_game_duplicates(self,eval_data):
        duplicated_rows_eval_data = eval_data.id_data.reset_index().duplicated(subset=[
            'Player', 'Year', 'Week'])
        eval_data.y_data = eval_data.y_data[np.logical_not(duplicated_rows_eval_data)]
        eval_data.id_data = eval_data.id_data.reset_index(
            drop=True).loc[np.logical_not(duplicated_rows_eval_data)].reset_index(drop=True)
        return eval_data


    def gather_sleeper_proj(self, eval_data):
        # Unique year-week combinations in evaluation dataset
        eval_data.id_data['Year-Week'] = eval_data.id_data[['Year',
                                                    'Week']].astype(str).agg('-'.join, axis=1)
        unique_year_weeks = list(eval_data.id_data['Year-Week'].unique())

        # Gather all stats from Sleeper
        with open(self.proj_dict_file, 'r', encoding='utf-8') as file:
            all_proj_dict = json.load(file)
        if not all(year_week in all_proj_dict for year_week in unique_year_weeks):
            # Gather any unsaved stats from Sleeper
            stats = Stats()
            for year_week in unique_year_weeks:
                if year_week not in all_proj_dict:
                    [year, week] = year_week.split('-')
                    week_proj = stats.get_week_projections(
                        'regular', int(year), int(week))
                    all_proj_dict[year_week] = week_proj
                    print(
                        f'Adding Year-Week {year_week} to Sleeper projections dictionary: {self.proj_dict_file}')
            # Save player dictionary to JSON file for use next time
            with open(self.proj_dict_file, 'w', encoding='utf-8') as file:
                json.dump(all_proj_dict, file)

        return all_proj_dict


    def eval_model(self, eval_data):
        # Remove duplicated games from eval data (only one projection per game from Sleeper)
        eval_data = self.remove_game_duplicates(eval_data)

        # Gather projections data from Sleeper API
        self.all_proj_dict = self.gather_sleeper_proj(eval_data)

        # Build up array of predicted stats for all eval_data cases based on
        # sleeper projections dictionary
        stat_predicts = torch.empty(0)
        for row in range(eval_data.id_data.shape[0]):
            id_row = eval_data.id_data.iloc[row]
            year_week = id_row['Year-Week']
            player = id_row['Player']
            if player in self.player_to_sleeper_id:
                proj_stats = self.all_proj_dict[year_week][self.player_to_sleeper_id[player]]
                stat_line = torch.tensor(reformat_sleeper_stats(proj_stats))
            else:
                stat_line = torch.zeros([12])
            stat_predicts = torch.cat((stat_predicts, stat_line))

        # Compute fantasy points using stat lines (note that this ignores the
        # built-in fantasy points projection in the Sleeper API, which differs
        # from the sum of the stats)
        stat_predicts = stats_to_fantasy_points(torch.reshape(
            stat_predicts, [-1, 12]), stat_indices='default', normalized=False)

        # True stats from eval data
        stat_truths = self.eval_truth(eval_data)

        # Create result object
        result = self.gen_prediction_result(stat_predicts, stat_truths, eval_data)

        return result


class NaivePredictor(FantasyPredictor):
    def __init__(self, name, n=1):
        # Initialize FantasyPredictor
        super().__init__(name)
        self.n = n

    def eval_model(self, eval_data, all_data):
        # Drop all the duplicated rows that are for the same game, and only
        # dependent on elapsed game time - that variable is irrelevant here, so we
        # can greatly simplify
        duplicated_rows_eval_data = eval_data.id_data.reset_index().duplicated(
            subset=['Player', 'Year', 'Week'])
        eval_data.y_data = eval_data.y_data[np.logical_not(duplicated_rows_eval_data)]
        eval_data.id_data = eval_data.id_data.reset_index(
            drop=True).loc[np.logical_not(duplicated_rows_eval_data)].reset_index(drop=True)
        duplicated_rows_all_data = all_data.id_data.duplicated(
            subset=['Player', 'Year', 'Week'])
        all_data.y_data = all_data.y_data[np.logical_not(duplicated_rows_all_data)]
        all_data.id_data = all_data.id_data.loc[np.logical_not(duplicated_rows_all_data)].reset_index(
            drop=True)

        # Variables needed to search for previous games and convert stats to
        # fantasy points
        first_year_in_dataset = min(all_data.id_data['Year'])
        # Set up dataframe to use to search for previous games
        all_ids = all_data.id_data.copy()[['Player', 'Year', 'Week']]
        all_ids = all_ids.reset_index().set_index(
            ['Player', 'Year', 'Week']).sort_index()
        indices = all_ids.index
        all_ids['Player Copy'] = all_ids.index.get_level_values(0).to_list()
        all_ids['Prev Year'] = all_ids.index.get_level_values(1).to_list()
        all_ids['Prev Week'] = all_ids.index.get_level_values(2).to_list()
        all_ids['Prev Game Found'] = False
        all_ids['Continue Prev Search'] = True
        while any(all_ids['Continue Prev Search']):
            # Only make changes to rows of the dataframe that are continuing the search
            search_ids = all_ids['Continue Prev Search']
            # Find the next (previous) week to look for
            all_ids.loc[search_ids,
                        'Prev Week'] = all_ids.loc[search_ids,
                                                'Prev Week'] - 1
            # If previous week is set to Week 0, fix this by going to Week 18 of the
            # previous year
            all_ids.loc[search_ids, 'Prev Year'] = all_ids.loc[search_ids].apply(
                lambda x: x['Prev Year'] - 1 if x['Prev Week'] == 0 else x['Prev Year'], axis=1)
            all_ids.loc[search_ids, 'Prev Week'] = all_ids.loc[search_ids].apply(
                lambda x: 18 if x['Prev Week'] == 0 else x['Prev Week'], axis=1)

            # Check if previous game is in the dataframe
            all_ids.loc[search_ids, 'Prev Game Found'] = all_ids.loc[search_ids].apply(
                lambda x: (x['Player Copy'], x['Prev Year'], x['Prev Week']) in indices, axis=1)

            # Check whether to continue searching for a previous game for this player
            all_ids.loc[search_ids,'Continue Prev Search'] = np.logical_not(
                all_ids.loc[search_ids,'Prev Game Found']) & (
                    all_ids.loc[search_ids,'Prev Year'] >= first_year_in_dataset)

        # Assign the index (row) of the previous game
        all_ids['Prev Game Index'] = np.nan
        valid_rows = all_ids['Prev Game Found']
        all_ids.loc[valid_rows, 'Prev Game Index'] = all_ids.loc[valid_rows].apply(
            lambda x: all_ids.loc[(x['Player Copy'], x['Prev Year'], x['Prev Week']), 'index'], axis=1)

        # Get the stats for each previous game, in order
        all_ids['tensor'] = all_ids.apply(
            lambda x: list(all_data.y_data[int(x['Prev Game Index'])]) if x['Prev Game Index'] >= 0
            else [np.nan] * all_data.y_data.shape[1], axis=1)

        # Grab stats for each game in the evaluation data
        prev_game_stats_df = eval_data.id_data.apply(
            lambda x: all_ids.loc[(x['Player'], x['Year'], x['Week']), 'tensor'], axis=1)
        prev_game_stats = torch.tensor(prev_game_stats_df.to_list())

        # Un-normalize and compute Fantasy score
        stat_predicts = stats_to_fantasy_points(
            prev_game_stats, stat_indices='default', normalized=True)
        # True stats from eval data
        stat_truths = self.eval_truth(eval_data)

        # Create result object
        result = self.gen_prediction_result(stat_predicts, stat_truths, eval_data)

        return result

class PerfectPredictor(FantasyPredictor):
    def __init__(self, name):
        # Initialize FantasyPredictor
        super().__init__(name)

    def eval_model(self, eval_data):
        # True stats from eval data
        stat_truths = self.eval_truth(eval_data)
        # Predicts equal truth
        stat_predicts = stat_truths

        # Create result object
        result = self.gen_prediction_result(stat_predicts, stat_truths, eval_data)

        return result

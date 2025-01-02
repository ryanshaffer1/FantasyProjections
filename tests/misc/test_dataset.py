import unittest
import random
import numpy as np
import pandas as pd
import pandas.testing as pdtest
import torch
# Module under test
from misc.dataset import StatsDataset
# Modules needed for test setup
import tests._utils_for_tests.mock_data as mock_data
import logging
import logging.config
from config.log_config import LOGGING_CONFIG

# Set up same logger as project code
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('log')

class TestConstructor_StatsDataset(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.id_df = mock_data.id_df
        self.pbp_df = mock_data.pbp_df
        self.bs_df = mock_data.bs_df
        self.name = 'dataset'

    def test_basic_attributes_df_input(self):
        dataset = StatsDataset(name=self.name,id_df=self.id_df,
                               pbp_df=self.pbp_df,boxscore_df=self.bs_df)
        inputs = ['dataset', self.id_df]
        inputs.append(torch.tensor(self.pbp_df.values))
        inputs.append(torch.tensor(self.bs_df.values))
        inputs.append(self.pbp_df.columns.to_list())
        inputs.append(self.bs_df.columns.to_list())
        inputs.append(['weeks', 'years', 'teams', 'players', 'elapsed_time'])

        # Multiple assertions, oh well
        self.assertEqual(dataset.name,self.name)
        pdtest.assert_frame_equal(dataset.id_data,inputs[1])
        self.assertTrue(torch.equal(dataset.x_data,inputs[2]))
        self.assertTrue(torch.equal(dataset.y_data,inputs[3]))
        self.assertEqual(dataset.x_data_columns,inputs[4])
        self.assertEqual(dataset.y_data_columns,inputs[5])
        self.assertEqual(dataset.valid_criteria,inputs[6])

    def test_basic_attributes_data_input(self):
        dataset = StatsDataset(name=self.name,id_df=self.id_df,
                               x_data=torch.tensor(self.pbp_df.values), x_data_columns=self.pbp_df.columns.to_list(),
                               y_data=torch.tensor(self.bs_df.values), y_data_columns=self.bs_df.columns.to_list(),)
        inputs = ['dataset', self.id_df]
        inputs.append(torch.tensor(self.pbp_df.values))
        inputs.append(torch.tensor(self.bs_df.values))
        inputs.append(self.pbp_df.columns.to_list())
        inputs.append(self.bs_df.columns.to_list())
        inputs.append(['weeks', 'years', 'teams', 'players', 'elapsed_time'])

        # Multiple assertions, oh well
        self.assertEqual(dataset.name,self.name)
        pdtest.assert_frame_equal(dataset.id_data,inputs[1])
        self.assertTrue(torch.equal(dataset.x_data,inputs[2]))
        self.assertTrue(torch.equal(dataset.y_data,inputs[3]))
        self.assertEqual(dataset.x_data_columns,inputs[4])
        self.assertEqual(dataset.y_data_columns,inputs[5])
        self.assertEqual(dataset.valid_criteria,inputs[6])
    
    def test_missing_inputs_raises_error(self):
        with self.assertRaises(TypeError):
            StatsDataset(name=self.name, pbp_df=self.pbp_df, boxscore_df=self.bs_df)
        with self.assertRaises(TypeError):
            StatsDataset(name=self.name, id_df=self.id_df, pbp_df=self.pbp_df)
        with self.assertRaises(TypeError):
            StatsDataset(name=self.name, id_df=self.id_df, boxscore_df=self.bs_df)

    def test_non_pandas_df_inputs_raises_error(self):
        with self.assertRaises(TypeError):
            StatsDataset(name=self.name,id_df=self.id_df, 
                         pbp_df=torch.tensor(self.pbp_df.values),
                         boxscore_df=self.bs_df)

    def test_non_tensor_data_inputs_raises_error(self):
        with self.assertRaises(TypeError):
            StatsDataset(name=self.name,id_df=self.id_df, 
                         x_data=torch.tensor(self.pbp_df.values), x_data_columns=self.pbp_df.columns.to_list(),
                         y_data=self.bs_df, y_data_columns=self.bs_df.columns.to_list())

    def test_mixed_df_and_tensor_inputs_gives_correct_result(self):
        dataset = StatsDataset(name=self.name,id_df=self.id_df, 
                                x_data=torch.tensor(self.pbp_df.values), x_data_columns=self.pbp_df.columns.to_list(),
                                boxscore_df=self.bs_df)
        dataset_expected = StatsDataset(name=self.name,id_df=self.id_df,
                                        pbp_df=self.pbp_df,boxscore_df=self.bs_df)

        self.assertTrue(dataset.equals(dataset_expected))
        
    def test_slicing_data_by_indices(self):
        i_start = 3
        i_end = 6
        dataset_sliced_by_index = StatsDataset(name=self.name, id_df=self.id_df, pbp_df=self.pbp_df, boxscore_df=self.bs_df,
                                               start_index=i_start, end_index=i_end)
        dataset_subset_of_data = StatsDataset(name=self.name,id_df=self.id_df[i_start:i_end], 
                                              pbp_df=self.pbp_df.iloc[i_start:i_end],
                                              boxscore_df=self.bs_df.iloc[i_start:i_end])
                                              
        self.assertTrue(dataset_sliced_by_index.equals(dataset_subset_of_data))

    def test_slicing_data_by_start_index(self):
        i_start = 3
        dataset_sliced_by_index = StatsDataset(name=self.name, id_df=self.id_df, 
                                               pbp_df=self.pbp_df, boxscore_df=self.bs_df,
                                               start_index=i_start)
        dataset_subset_of_data = StatsDataset(name=self.name, id_df=self.id_df[i_start:],
                                              pbp_df=self.pbp_df.iloc[i_start:],
                                              boxscore_df=self.bs_df.iloc[i_start:])
        self.assertTrue(dataset_sliced_by_index.equals(dataset_subset_of_data))

    def test_slicing_data_by_end_index(self):
        i_end = 6
        dataset_sliced_by_index = StatsDataset(name=self.name, id_df=self.id_df,
                                               pbp_df=self.pbp_df, boxscore_df=self.bs_df,
                                               end_index=i_end)
        dataset_subset_of_data = StatsDataset(name=self.name, id_df=self.id_df[:i_end],
                                              pbp_df=self.pbp_df.iloc[:i_end],
                                              boxscore_df=self.bs_df.iloc[:i_end])
        self.assertTrue(dataset_sliced_by_index.equals(dataset_subset_of_data))

    def test_slicing_data_by_criteria_simple(self):
        player = 'Austin Ekeler'
        df_indices = self.id_df['Player'] == player
        dataset_sliced_by_criteria = StatsDataset(name=self.name, id_df=self.id_df,
                                                  pbp_df=self.pbp_df, boxscore_df=self.bs_df,
                                                  players=[player])
        dataset_subset_of_data = StatsDataset(name=self.name, id_df=self.id_df[df_indices],
                                              pbp_df=self.pbp_df[df_indices],
                                              boxscore_df=self.bs_df[df_indices])
        self.assertTrue(dataset_sliced_by_criteria.equals(dataset_subset_of_data))

    def test_shuffle(self):
        dataset_shuffled = StatsDataset(name=self.name, id_df=self.id_df,
                                        pbp_df=self.pbp_df, boxscore_df=self.bs_df,
                                        shuffle=True)

        dataset_manually_shuffled = StatsDataset(name=self.name, id_df=self.id_df,
                                                 pbp_df=self.pbp_df, boxscore_df=self.bs_df,
                                                 shuffle=False)
        # Manually shuffle the unshuffled dataset
        random.seed(10)
        shuffled_indices = list(range(dataset_manually_shuffled.x_data.shape[0]))
        random.shuffle(shuffled_indices)
        dataset_manually_shuffled.x_data = dataset_manually_shuffled.x_data[shuffled_indices]
        dataset_manually_shuffled.y_data = dataset_manually_shuffled.y_data[shuffled_indices]
        dataset_manually_shuffled.id_data = dataset_manually_shuffled.id_data.iloc[shuffled_indices]
        self.assertTrue(dataset_shuffled.equals(dataset_manually_shuffled))
    
    # Tear Down
    def tearDown(self):
        pass

class TestEquals_StatsDataset(unittest.TestCase):
    # Set Up
    def setUp(self):
        # Dataset 1
        self.dataset = StatsDataset(name='dataset',
                                    id_df=mock_data.id_df,
                                    pbp_df=mock_data.pbp_df,
                                    boxscore_df=mock_data.bs_df)     
        # Dataset 2 using same inputs
        self.identical_dataset = StatsDataset(name='dataset',
                                              id_df=mock_data.id_df.copy(),
                                              pbp_df=mock_data.pbp_df.copy(),
                                              boxscore_df=mock_data.bs_df.copy())
        # Dataset 3 with new name
        self.dataset_new_name = StatsDataset(name='foo',
                                             id_df=mock_data.id_df.copy(),
                                             pbp_df=mock_data.pbp_df.copy(),
                                             boxscore_df=mock_data.bs_df.copy())
    
    def test_dataset_equals_self(self):
        self.assertTrue(self.dataset.equals(self.dataset))
    
    def test_identical_datasets_are_equal(self):

        self.assertTrue(self.dataset.equals(self.identical_dataset))
    
    def test_different_dataset_names_are_not_equal_when_flag_true(self):
        self.assertFalse(self.dataset.equals(self.dataset_new_name, check_non_data_attributes=True))

    def test_different_dataset_names_are_equal_when_flag_false(self):
        self.assertTrue(self.dataset.equals(self.dataset_new_name, check_non_data_attributes=False))
    
    def test_different_slice_criteria_are_not_equal_when_flag_true(self):
        self.identical_dataset.valid_criteria = []
        self.assertFalse(self.dataset.equals(self.identical_dataset, check_non_data_attributes=True))
    
    def test_different_x_data_are_not_equal(self):
        self.identical_dataset.x_data[0] = self.identical_dataset.x_data[0] + 1
        self.assertFalse(self.dataset.equals(self.identical_dataset))
    
    def test_different_y_data_are_not_equal(self):
        self.identical_dataset.y_data[1] = self.identical_dataset.y_data[1] + 1
        self.assertFalse(self.dataset.equals(self.identical_dataset))

    def test_different_x_data_columns_are_not_equal(self):
        self.identical_dataset.x_data_columns[-1] = 'blah'
        self.assertFalse(self.dataset.equals(self.identical_dataset))

    def test_different_y_data_columns_are_not_equal(self):
        self.identical_dataset.y_data_columns[-1] = 'blah'
        self.assertFalse(self.dataset.equals(self.identical_dataset))
    
    def test_different_id_data_are_not_equal(self):
        self.identical_dataset.id_data = self.identical_dataset.id_data.rename(columns={'Week':'XYZ'})
        self.assertFalse(self.dataset.equals(self.identical_dataset))
    
    # Tear Down
    def tearDown(self):
        pass

class TestConcat_StatsDataset(unittest.TestCase):
    # Set Up
    def setUp(self):
        # Partial dataset 1
        id_df1 = pd.DataFrame(data=[['Austin Ekeler',        2024, 1, 'WAS', 'TB', 'RB', 0],
                                    ['Austin Ekeler',       2024, 1, 'WAS', 'TB', 'RB', 1],
                                    ['Austin Ekeler',       2024, 1, 'WAS', 'TB', 'RB', 2],
                                    ['Austin Ekeler',       2024, 1, 'WAS', 'TB', 'RB', 3],],
                             columns=['Player', 'Year', 'Week', 'Team', 'Opponent', 'Position', 'Elapsed Time'])
        pbp_df1 = pd.DataFrame(data=[[0, 0.65],[0.016666667, 0.34],[0.033333333, 0.55],[0.05, 0.6]],
                              columns=['Elapsed Time','Field Position'])
        bs_df1 = pd.DataFrame(data=[[0, 0, 0.047619048],[0, 0, 0.047619048],[0, 0, 0.047619048],[0, 0, 0.047619048]],
                             columns=['Pass Att', 'Pass Cmp', 'Pass Yds'])
        self.dataset1 = StatsDataset(name='dataset', id_df=id_df1, pbp_df=pbp_df1, boxscore_df=bs_df1)
        
        # Partial dataset 2
        id_df2 = pd.DataFrame(data=[
                                    ['Olamide Zaccheaus',   2024, 4, 'WAS', 'ARI', 'WR', 22],
                                    ['Olamide Zaccheaus',   2024, 4, 'WAS', 'ARI', 'WR', 23],
                                    ['Olamide Zaccheaus',   2024, 4, 'WAS', 'ARI', 'WR', 24],],
                             columns=['Player', 'Year', 'Week', 'Team', 'Opponent', 'Position', 'Elapsed Time'])
        pbp_df2 = pd.DataFrame(data=[
                                    [0.366666667, 0.71],[0.383333333, 0.55],[0.4, 0.18]],
                              columns=['Elapsed Time','Field Position'])
        bs_df2 = pd.DataFrame(data=[[0, 0, 0.047619048],[0, 0, 0.047619048],[0, 0, 0.047619048]],
                             columns=['Pass Att', 'Pass Cmp', 'Pass Yds'])
        self.dataset2 = StatsDataset(name='dataset', id_df=id_df2, pbp_df=pbp_df2, boxscore_df=bs_df2)
        
        # Combined dataset
        id_df = pd.DataFrame(data=[['Austin Ekeler',        2024, 1, 'WAS', 'TB', 'RB', 0],
                                    ['Austin Ekeler',       2024, 1, 'WAS', 'TB', 'RB', 1],
                                    ['Austin Ekeler',       2024, 1, 'WAS', 'TB', 'RB', 2],
                                    ['Austin Ekeler',       2024, 1, 'WAS', 'TB', 'RB', 3],
                                    ['Olamide Zaccheaus',   2024, 4, 'WAS', 'ARI', 'WR', 22],
                                    ['Olamide Zaccheaus',   2024, 4, 'WAS', 'ARI', 'WR', 23],
                                    ['Olamide Zaccheaus',   2024, 4, 'WAS', 'ARI', 'WR', 24]],
                             columns=['Player', 'Year', 'Week', 'Team', 'Opponent', 'Position', 'Elapsed Time'])
        pbp_df = pd.DataFrame(data=[[0, 0.65],[0.016666667, 0.34],[0.033333333, 0.55],[0.05, 0.6],
                                    [0.366666667, 0.71],[0.383333333, 0.55],[0.4, 0.18]],
                              columns=['Elapsed Time','Field Position'])
        bs_df = pd.DataFrame(data=[[0, 0, 0.047619048],[0, 0, 0.047619048],[0, 0, 0.047619048],
                                   [0, 0, 0.047619048],[0, 0, 0.047619048],[0, 0, 0.047619048],
                                   [0, 0, 0.047619048]],
                             columns=['Pass Att', 'Pass Cmp', 'Pass Yds'])
        self.comb_dataset = StatsDataset(name='dataset',pbp_df=pbp_df,boxscore_df=bs_df,id_df=id_df)        
        
    def test_concat_inplace_gives_correct_result(self):
        self.dataset1.concat(self.dataset2, inplace=True)
        self.assertTrue(self.dataset1.equals(self.comb_dataset))
    
    def test_concat_inplace_returns_none(self):
        result = self.dataset1.concat(self.dataset2, inplace=True)
        self.assertIsNone(result)
    
    def test_concat_not_inplace_returns_correct_result(self):
        self.assertTrue(self.comb_dataset.equals(
            self.dataset1.concat(self.dataset2, inplace=False)))
    
    def test_concat_different_x_columns_gives_error(self):
        self.dataset2.x_data_columns = ['XYZ' if x=='Field Position' else x for x in self.dataset2.x_data_columns]
        
        with self.assertRaises(NameError):
            self.dataset1.concat(self.dataset2)

    def test_concat_different_y_columns_gives_error(self):
        self.dataset2.y_data_columns = ['XYZ' if x=='Pass Att' else x for x in self.dataset2.y_data_columns]
        with self.assertRaises(NameError):
            self.dataset1.concat(self.dataset2)

    def test_concat_different_id_columns_gives_error(self):
        self.dataset2.id_data = self.dataset2.id_data.rename(columns={'Team':'XYZ'})
        with self.assertRaises(NameError):
            self.dataset1.concat(self.dataset2)

    # Tear Down
    def tearDown(self):
        pass

class TestCopy_StatsDataset(unittest.TestCase):
    # Set Up
    def setUp(self):
        self.name = 'dataset'
        self.dataset = StatsDataset(name=self.name,
                                    id_df=mock_data.id_df,
                                    pbp_df=mock_data.pbp_df,
                                    boxscore_df=mock_data.bs_df)   
        
    def test_copy_returns_identical_dataset(self):
        dataset_copy = self.dataset.copy()
        self.assertTrue(self.dataset.equals(dataset_copy))

    def test_copy_returns_new_address(self):
        dataset_copy = self.dataset.copy()
        self.assertNotEqual(id(self.dataset), id(dataset_copy))

    # Tear Down
    def tearDown(self):
        pass

class TestSliceByCriteria_StatsDataset(unittest.TestCase):
    # Set Up
    def setUp(self):
       self.id_df = mock_data.id_df
       self.pbp_df = mock_data.pbp_df
       self.bs_df = mock_data.bs_df
       self.name = 'dataset'
       self.dataset = StatsDataset(name=self.name,pbp_df=self.pbp_df,boxscore_df=self.bs_df,id_df=self.id_df)

    def test_no_input_gives_unmodified_dataset(self):
        dataset_sliced = self.dataset.copy()
        dataset_sliced.slice_by_criteria()
        self.assertTrue(self.dataset.equals(dataset_sliced))
        
    def test_inplace_true_returns_none(self):
        result = self.dataset.slice_by_criteria(inplace=True)
        self.assertIsNone(result)
        
    def test_inplace_false_returns_new_object(self):
        result = self.dataset.slice_by_criteria(inplace=False)
        self.assertTrue(self.dataset.equals(result))
        
    def test_slice_by_week(self):
        weeks = [1]
        indices_with_slice = [0,1,2,3] # Set manually
        dataset_expected = StatsDataset(name=self.name,
                                        pbp_df=self.pbp_df.iloc[indices_with_slice],
                                        boxscore_df=self.bs_df.iloc[indices_with_slice],
                                        id_df=self.id_df.iloc[indices_with_slice])
        dataset_sliced = self.dataset.slice_by_criteria(inplace=False, weeks=weeks)

        self.assertTrue(dataset_sliced.equals(dataset_expected))
        
    def test_slice_by_year(self):
        years = [2023]
        indices_with_slice = [8] # Set manually
        dataset_expected = StatsDataset(name=self.name,
                                        pbp_df=self.pbp_df.iloc[indices_with_slice],
                                        boxscore_df=self.bs_df.iloc[indices_with_slice],
                                        id_df=self.id_df.iloc[indices_with_slice])
        dataset_sliced = self.dataset.slice_by_criteria(inplace=False, years=years)

        self.assertTrue(dataset_sliced.equals(dataset_expected))
        
    def test_slice_by_team(self):
        teams = ['ARI']
        indices_with_slice = [8] # Set manually
        dataset_expected = StatsDataset(name=self.name,
                                        pbp_df=self.pbp_df.iloc[indices_with_slice],
                                        boxscore_df=self.bs_df.iloc[indices_with_slice],
                                        id_df=self.id_df.iloc[indices_with_slice])
        dataset_sliced = self.dataset.slice_by_criteria(inplace=False, teams=teams)

        self.assertTrue(dataset_sliced.equals(dataset_expected))
        
    def test_slice_by_player(self):
        players = ['Olamide Zaccheaus']
        indices_with_slice = [4,5,6] # Set manually
        dataset_expected = StatsDataset(name=self.name,
                                        pbp_df=self.pbp_df.iloc[indices_with_slice],
                                        boxscore_df=self.bs_df.iloc[indices_with_slice],
                                        id_df=self.id_df.iloc[indices_with_slice])
        dataset_sliced = self.dataset.slice_by_criteria(inplace=False, players=players)

        self.assertTrue(dataset_sliced.equals(dataset_expected))
        
    def test_slice_by_elapsed_time(self):
        times = [53]
        indices_with_slice = [9] # Set manually
        dataset_expected = StatsDataset(name=self.name,
                                        pbp_df=self.pbp_df.iloc[indices_with_slice],
                                        boxscore_df=self.bs_df.iloc[indices_with_slice],
                                        id_df=self.id_df.iloc[indices_with_slice])
        dataset_sliced = self.dataset.slice_by_criteria(inplace=False, elapsed_time=times)

        self.assertTrue(dataset_sliced.equals(dataset_expected))

    def test_slice_by_unsupported_variable_gives_unmodified_dataset(self):
        opponents = ['TB']
        dataset_sliced = self.dataset.slice_by_criteria(inplace=False, opponents=opponents)

        self.assertTrue(self.dataset.equals(dataset_sliced))     
    
    def test_slice_by_list_of_values(self):
        weeks = [1,3]
        indices_with_slice = [0,1,2,3,7] # Set manually
        dataset_expected = StatsDataset(name=self.name,
                                        pbp_df=self.pbp_df.iloc[indices_with_slice],
                                        boxscore_df=self.bs_df.iloc[indices_with_slice],
                                        id_df=self.id_df.iloc[indices_with_slice])
        dataset_sliced = self.dataset.slice_by_criteria(inplace=False, weeks=weeks)

        self.assertTrue(dataset_sliced.equals(dataset_expected))
        
    def test_slice_by_non_list_gives_error(self):
        weeks = 1

        with self.assertRaises(TypeError):
            self.dataset.slice_by_criteria(weeks=weeks)

    def test_slice_by_multiple_criteria_simultaneously(self):
        weeks = [1]
        elapsed_time = [2,3]
        indices_with_slice = [2,3] # Set manually
        dataset_expected = StatsDataset(name=self.name,
                                        pbp_df=self.pbp_df.iloc[indices_with_slice],
                                        boxscore_df=self.bs_df.iloc[indices_with_slice],
                                        id_df=self.id_df.iloc[indices_with_slice])
        dataset_sliced = self.dataset.slice_by_criteria(inplace=False, weeks=weeks, elapsed_time=elapsed_time)

        self.assertTrue(dataset_sliced.equals(dataset_expected))

    def test_slice_by_multiple_criteria_sequentially(self):
        weeks = [1,3]
        elapsed_time = [2,3,18]
        indices_with_slice = [2,3,7] # Set manually
        dataset_expected = StatsDataset(name=self.name,
                                        pbp_df=self.pbp_df.iloc[indices_with_slice],
                                        boxscore_df=self.bs_df.iloc[indices_with_slice],
                                        id_df=self.id_df.iloc[indices_with_slice])
        dataset_sliced = self.dataset.slice_by_criteria(inplace=False, weeks=weeks)
        dataset_sliced.slice_by_criteria(inplace=True, elapsed_time=elapsed_time)

        self.assertTrue(dataset_sliced.equals(dataset_expected))

    # Tear Down
    def tearDown(self):
        pass

class TestRemoveGameDuplicates_StatsDataset(unittest.TestCase):
    # Set Up
    def setUp(self):
        # Dataset with duplicate games (e.g. multiple entries for same player/week)
        id_df = mock_data.id_df.copy()
        pbp_df = mock_data.pbp_df.copy()
        bs_df = mock_data.bs_df.copy()
        self.dummy_dataset = StatsDataset(name='dataset',
                                          id_df=id_df,
                                          pbp_df=pbp_df,
                                          boxscore_df=bs_df)

        # Desired result: dataset with a single row per player/game
        unique_game_indices = np.logical_not(id_df.duplicated(subset=('Player','Year','Week'),keep='first'))
        id_df_no_dupe_games = id_df[unique_game_indices].reset_index(drop=True)
        pbp_df_no_dupe_games = pbp_df[unique_game_indices].reset_index(drop=True)
        bs_df_no_dupe_games = bs_df[unique_game_indices].reset_index(drop=True)
        self.dataset_no_dupe_games = StatsDataset(name='dataset',
                                                  id_df=id_df_no_dupe_games,
                                                  pbp_df=pbp_df_no_dupe_games,
                                                  boxscore_df=bs_df_no_dupe_games)

    def test_correct_duplicates_removed_inplace_false(self):
        result = self.dummy_dataset.remove_game_duplicates(inplace=False)
        self.assertTrue(result.equals(self.dataset_no_dupe_games))

    def test_correct_duplicates_removed_inplace_true(self):
        self.dummy_dataset.remove_game_duplicates(inplace=True)
        self.assertTrue(self.dummy_dataset.equals(self.dataset_no_dupe_games))


    # Tear Down
    def tearDown(self):
        pass

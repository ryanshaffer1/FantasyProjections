import numpy as np
import torch
from nn_helper_functions import stats_to_fantasy_points
from nn_plot_functions import gen_scatterplots


def train(dataloader, model, device, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = int(np.ceil(size / dataloader.batch_size))
    model.train()
    for batch, (x_matrix, y_matrix) in enumerate(dataloader):
        x_matrix, y_matrix = x_matrix.to(device), y_matrix.to(device)

        # Compute prediction error
        pred = model(x_matrix)
        loss = loss_fn(pred, y_matrix)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % int(num_batches / 10) == 0:
            loss, current = loss.item(), (batch + 1) * len(x_matrix)
            print(f"\tloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    # Names of all statistics
    stat_indices = [
        'Pass Att',
        'Pass Cmp',
        'Pass Yds',
        'Pass TD',
        'Int',
        'Rush Att',
        'Rush Yds',
        'Rush TD',
        'Rec',
        'Rec Yds',
        'Rec TD',
        'Fmb']
    # Gather all predicted/true outputs for the input dataset
    model.eval()
    pred = torch.empty([0, 12])
    y_matrix = torch.empty([0, 12])
    with torch.no_grad():
        for (x_matrix, y_vec) in dataloader:
            pred = torch.cat((pred, model(x_matrix)))
            y_matrix = torch.cat((y_matrix, y_vec))

    # Convert outputs into un-normalized statistics/fantasy points
    stat_predicts = stats_to_fantasy_points(pred, stat_indices, normalized=True)
    stat_truths = stats_to_fantasy_points(y_matrix, stat_indices, normalized=True)
    # Calculate average difference in fantasy points between prediction and
    # truth
    fp_predicts = stat_predicts['Fantasy Points']
    fp_truths = stat_truths['Fantasy Points']
    fp_diff = [abs(predict - truth)
               for predict, truth in zip(fp_predicts, fp_truths)]

    print(
        f"Test Error: Avg Fantasy Points Different = {(np.mean(fp_diff)):>0.2f}")

    return stat_predicts, stat_truths, np.mean(fp_diff)


def results_eval(stat_predicts, stat_truths, dataset):
    # ID (player, week, year, team, position, etc) for each data point
    data_ids = dataset.__getids__().reset_index(drop=True)

    # Configure scatter plots
    plots_kwargs = []
    plots_kwargs.append({'columns': ['Pass Att',
                                     'Pass Cmp',
                                     'Pass Yds',
                                     'Pass TD',
                                     'Int'],
                         'slice': {'Position': ['QB']},
                         'subtitle': 'Passing Stats',
                         'histograms': True})
    plots_kwargs.append({'columns': ['Pass Att', 'Pass Cmp', 'Pass Yds', 'Pass TD'],
                         'legend_slice': {'Position': [['QB'], ['RB', 'WR', 'TE']]},
                         'subtitle': 'Passing Stats',
                         })
    plots_kwargs.append({'columns': ['Rush Att', 'Rush Yds', 'Rush TD', 'Fmb'],
                         'subtitle': 'Rushing Stats',
                         'histogram': True
                         })
    plots_kwargs.append({'columns': ['Rec', 'Rec Yds', 'Rec TD'],
                         'legend_slice': {'Position': [['RB', 'WR', 'TE'], ['QB']]},
                         'subtitle': 'Receiving Stats',
                         })
    plots_kwargs.append({'columns': ['Fantasy Points'],
                         'subtitle': 'Fantasy Points',
                         'histograms': True
                         })

    # Generate scatter plots
    for plot_kwargs in plots_kwargs:
        gen_scatterplots(stat_truths, stat_predicts, data_ids, **plot_kwargs)

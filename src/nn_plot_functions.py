import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from nn_helper_functions import stats_to_fantasy_points


def scatter_hist(x, y, ax, ax_histx, ax_histy, alpha):
    # Configure axis ticks and tick labels for histograms
    ax_histx.tick_params(axis='x', labelbottom=True, bottom=True)
    ax_histx.tick_params(axis='y', labelleft=False, left=False)
    ax_histy.tick_params(axis='y', labelleft=True, left=True)
    ax_histy.tick_params(axis='x', labelbottom=False, bottom=False)

    # Scatter plot (with no tick labels)
    ax.scatter(x, y, alpha=alpha)
    ax.tick_params(axis='x', labelbottom=False, bottom=False)
    ax.tick_params(axis='y', labelleft=False, left=False)

    # Get axis limits
    min_lim = min(np.concat((ax.get_xlim(),ax.get_ylim())))
    max_lim = max(np.concat((ax.get_xlim(),ax.get_ylim())))
    lim_vals = np.linspace(min_lim, max_lim, num=2)
    # Set axis limits
    ax.set_xlim(lim_vals[0], lim_vals[-1])
    ax.set_ylim(lim_vals[0], lim_vals[-1])

    # Set bins based on x-axis tickmarks
    num_bins_per_tick = 4
    ticks = ax.get_xticks()
    tickspacing = ticks[1] - ticks[0]
    bins = np.arange(ticks[0], ticks[-1], step=tickspacing / num_bins_per_tick)
    bins -= tickspacing / num_bins_per_tick / 2

    # Histograms
    ax_histx.hist(x, bins=bins, edgecolor='k')
    ax_histy.hist(y, bins=bins, edgecolor='k', orientation='horizontal')


def check_df_for_slice(x,key,slice_var,slice_type='dict'):
    match slice_type:
        case 'dict':
            if isinstance(slice_var[key],list):
                return x[key] in slice_var[key]
            else:
                return x[key] == slice_var[key]
        case 'list':
            return x[key] in slice_var


def plot_test_results(truth_df, predict_df, id_df, **kwargs):
    # Handle keyword arguments
    columns = kwargs.get('columns', None)
    plot_slice = kwargs.get('slice', {})
    legend_slice = kwargs.get('legend_slice', {})
    subtitle = kwargs.get('subtitle', None)
    histograms = kwargs.get('histograms', False)

    # Bold text
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams["axes.labelweight"] = "bold"

    # Slice data based on optional input
    if plot_slice:
        for key in plot_slice:
            indices_to_keep = id_df.apply(
                check_df_for_slice, args=(key,plot_slice), axis=1)
            truth_df = truth_df[indices_to_keep]
            predict_df = predict_df[indices_to_keep]
            id_df = id_df[indices_to_keep]

    # Break up into multiple x/y data sets if legend slices are input
    if legend_slice:
        truth_dfs = []
        predict_dfs = []
        for key in legend_slice:
            for val in legend_slice[key]:
                indices_to_keep = id_df.apply(check_df_for_slice, args=(key,val,'list'), axis=1)
                truth_dfs.append(truth_df[indices_to_keep])
                predict_dfs.append(predict_df[indices_to_keep])
    else:
        truth_dfs = [truth_df]
        predict_dfs = [predict_df]

    # Determine arrangement of subplots needed
    num_subplots = len(columns)
    numrows = int(np.floor(np.sqrt(num_subplots)))
    numcols = int(np.ceil(num_subplots / numrows))

    # Set up figure with subplots
    fig, axs = plt.subplots(nrows=numrows, ncols=numcols, layout='constrained')

    plot_ind = -1 # Initialize loop variable, since it will be used outside of the loop
    for plot_ind, column in enumerate(columns):

        # Configure subplot
        ax = axs.transpose().flat[plot_ind] if num_subplots > 1 else axs
        # ax.set(aspect=1) # Doesn't seem to do anything
        if histograms:
            ax_histx = ax.inset_axes([0, -0.15, 1, 0.15], sharex=ax)
            ax_histy = ax.inset_axes([-0.15, 0, 0.15, 1], sharey=ax)
            # Axis labels
            ax_histx.set_xlabel(f'True {column}')
            ax_histy.set_ylabel(f'Predicted {column}')
        else:
            # Axis labels
            ax.set_xlabel(f'True {column}')
            ax.set_ylabel(f'Predicted {column}')

        # Plot each set of data (broken up by legend entries)
        for (sliced_truth_df, sliced_predict_df) in zip(
                truth_dfs, predict_dfs):
            sliced_truth_data = sliced_truth_df[column]
            sliced_predict_data = sliced_predict_df[column]
            # Set alpha based on data size (threshold set empirically by what
            # has looked good)
            alpha = 0.1 if sliced_truth_data.shape[0] > 2000 else 0.5

            if histograms:
                # Plot data on scatter plot with histograms axes
                scatter_hist(
                    sliced_truth_data,
                    sliced_predict_data,
                    ax,
                    ax_histx,
                    ax_histy,
                    alpha)
            else:
                # Plot data on scatter plot with NO histograms axes
                ax.scatter(sliced_truth_data, sliced_predict_data, alpha=alpha)

        # Legend (if legend slice was set)
        if legend_slice:
            leg = ax.legend([', '.join(val)
                            for val in legend_slice[key] for key in legend_slice])
            for lh in leg.legend_handles:
                lh.set_alpha(1)

        # Get axis limits
        min_lim = min(np.concat((ax.get_xlim(),ax.get_ylim())))
        max_lim = max(np.concat((ax.get_xlim(),ax.get_ylim())))
        lim_vals = np.linspace(min_lim, max_lim, num=2)

        # Trendlines: ideal trendline, and line of best fit
        # Re-consolidate x,y datasets
        truth_data = truth_df[column]
        predict_data = predict_df[column]
        if len(truth_data.unique(
        )) > 1:  # Only plot trendlines if there is more than one unique value on the x axis
            # Draw line along y=x (values should clump along this line for a
            # well-performing model)
            ideal_line, = ax.plot(lim_vals, lim_vals, '--', color='0.4')
            # Draw line of best fit, display r-squared as text on figure
            slope, intercept, r_value = scipy.stats.linregress(
                truth_data, predict_data)[0:3]
            bestfit_line, = ax.plot(
                lim_vals, slope * lim_vals + intercept, 'k--')
            ax.text(lim_vals[-1],
                    slope * lim_vals[-1] + intercept,
                    f'$R^2 = {(r_value**2):>0.3f}$ ',
                    horizontalalignment='right',
                    verticalalignment=('bottom' if r_value > 0 else 'top'))

        # Set axis limits
        ax.set_xlim(lim_vals[0], lim_vals[-1])
        ax.set_ylim(lim_vals[0], lim_vals[-1])

    # Make any remaining unused subplots invisible
    for i in range(plot_ind + 1, numrows * numcols):
        ax = axs.flat[i]
        ax.set_axis_off()

    # Add legend (if trendlines were plotted on at least one of the subplots)
    if 'ideal_line' in locals():
        if num_subplots > 1:
            fig.legend([ideal_line, bestfit_line], [
                       'Ideal Trendline', 'Actual Trendline'], loc='lower right')
        else:
            ax.legend([ideal_line, bestfit_line], [
                      'Ideal Trendline', 'Actual Trendline'], loc='lower right')

    # Figure title with optional subtitle
    title = 'Neural Net Performance: Predictions vs Truth'
    # Optional subtitle, with info on slices included
    if subtitle:
        title += '\n' + subtitle + \
            ''.join(['\n ' + key + ': ' + ', '.join(plot_slice[key])
                    for key in list(plot_slice.keys())])
    fig.suptitle(title, weight='bold')

    # "Display" plot (won't really be displayed until plt.show is called again without block=False)
    plt.show(block=False)


def plot_game_timeline(game_id, id_df, pbp_df, predict_df, truth_df):
    # Filter to only the desired game/player
    for key in list(game_id.keys()):
        indices_to_keep = id_df.apply(check_df_for_slice, args=(key,game_id), axis=1)
        pbp_df = pbp_df[indices_to_keep]
        truth_df = truth_df[indices_to_keep]
        predict_df = predict_df[indices_to_keep]
        id_df = id_df[indices_to_keep]

    ax = plt.subplots()[1]
    ax.plot(pbp_df['Elapsed Time'], pbp_df['Fantasy Points'])
    ax.plot(pbp_df['Elapsed Time'], predict_df['Fantasy Points'])
    ax.plot(pbp_df['Elapsed Time'], truth_df['Fantasy Points'], 'k--')
    ax.legend(['Live Fantasy Score',
               'Neural Net Predicted Final Fantasy Score',
               'True Final Fantasy Score'])
    ax.set_title(
        f'Fantasy Score, {
            game_id['Player']}, {
            game_id['Year']} Week {
                game_id['Week']}')
    ax.set_xlabel('Elapsed Game Time (min)')
    ax.set_ylabel('Fantasy Points')


def eval_single_games(predict_df, truth_df, dataset, game_ids=None, n_random=0):
    # Handle optional input of specific games/players
    if game_ids is None:
        game_ids = []

    # ID (player, week, year, team, position, etc) for each data point
    id_df = dataset.__getids__().reset_index(drop=True)

    # Play-by-play data (used as x [input] data in NN)
    pbp_df = pd.DataFrame(dataset.x_data, columns=dataset.x_data_labels)
    # Add fantasy points to play-by-play
    cols_to_keep = [
        'Elapsed Time',
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
    pbp_df = stats_to_fantasy_points(pbp_df.loc[:, cols_to_keep], normalized=True)

    # (Optionally) Add random players/games to plot
    for _ in range(n_random):
        valid_new_entry = False
        while not valid_new_entry:
            ind = np.random.randint(id_df.shape[0])
            game_dict = {col: id_df.iloc[ind][col]
                         for col in ['Player', 'Week', 'Year']}
            if game_dict not in game_ids:
                game_ids.append(game_dict)
                valid_new_entry = True

    for game_id in game_ids:
        plot_game_timeline(game_id, id_df, pbp_df, predict_df, truth_df)

    # "Display" plot (won't really be displayed until plt.show is called again without block=False)
    plt.show(block=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Bold text on everything
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["axes.labelweight"] = "bold"

def gen_scatterplots(result, **kwargs):
    # Handle keyword arguments
    columns = kwargs.get('columns', None)
    plot_slice = kwargs.get('slice', {})
    legend_slice = kwargs.get('legend_slice', {})
    subtitle = kwargs.get('subtitle', None)
    histograms = kwargs.get('histograms', False)

    # Copy dataframes (to preserve original object attributes)
    # And trim content to only the data spec'd in plot_slice/legend_slice
    [truth_dfs,predict_dfs] = data_slice_to_plot(result,plot_slice,legend_slice)[0:2]

    # Determine arrangement of subplots needed
    num_subplots = len(columns)
    numrows = int(np.floor(np.sqrt(num_subplots)))
    numcols = int(np.ceil(num_subplots / numrows))

    # Set up figure with subplots
    fig, axs = plt.subplots(nrows=numrows, ncols=numcols, layout='constrained')
    axs = np.array(axs) if num_subplots == 1 else axs

    for ax, column in zip(axs.transpose().flat,columns):
        # Create scatter plot with optional histograms on x/y axes
        ideal_line,bestfit_line = scatter_hist(
            truth_dfs,
            predict_dfs,
            ax,
            column=column,
            legend_slice=legend_slice,
            histograms=histograms)

    # Make any remaining unused subplots invisible
    for i in range(num_subplots, numrows * numcols):
        axs.flat[i].set_axis_off()

    # Add legend (if trendlines were plotted on at least one of the subplots)
    if ideal_line:
        if num_subplots > 1:
            fig.legend([ideal_line, bestfit_line], [
                       'Ideal Trendline', 'Actual Trendline'], loc='lower right')
        else:
            axs.flat[0].legend([ideal_line, bestfit_line], [
                      'Ideal Trendline', 'Actual Trendline'], loc='lower right')

    # Figure title with optional subtitle
    title = f'{result.predictor_name} Performance: Predictions vs Truth'
    # Optional subtitle, with info on slices included
    if subtitle:
        title += '\n' + subtitle + \
            ''.join(['\n ' + key + ': ' + ', '.join(plot_slice[key])
                    for key in list(plot_slice.keys())])
    fig.suptitle(title, weight='bold')

    # "Display" plot (won't really be displayed until plt.show is called again without block=False)
    plt.show(block=False)


def scatter_hist(truth_dfs, predict_dfs, ax, column, legend_slice, histograms=True):
    if histograms:
        # Axes for histograms
        ax_histx = ax.inset_axes([0, -0.15, 1, 0.15], sharex=ax)
        ax_histy = ax.inset_axes([-0.15, 0, 0.15, 1], sharey=ax)
        # Axis labels
        ax_histx.set_xlabel(f'True {column}')
        ax_histy.set_ylabel(f'Predicted {column}')
        # Configure axis ticks and tick labels for histograms
        ax_histx.tick_params(axis='x', labelbottom=True, bottom=True)
        ax_histx.tick_params(axis='y', labelleft=False, left=False)
        ax_histy.tick_params(axis='y', labelleft=True, left=True)
        ax_histy.tick_params(axis='x', labelbottom=False, bottom=False)
        # Remove tick labels on scatterplot
        ax.tick_params(axis='x', labelbottom=False, bottom=False)
        ax.tick_params(axis='y', labelleft=False, left=False)
    else:
        # Axis labels
        ax.set_xlabel(f'True {column}')
        ax.set_ylabel(f'Predicted {column}')

    # Plot each set of data (broken up by legend entries)
    for (x, y) in zip(truth_dfs, predict_dfs):
        x,y = x[column],y[column]
        # Set alpha based on data size (threshold set empirically by what has looked good)
        alpha = 0.1 if x.shape[0] > 2000 else 0.5
        alpha = 1

        # Create scatterplot
        ax.scatter(x, y, alpha=alpha)

        # Get and set axis limits for equal axes
        lim_vals = np.linspace(min(np.concat((ax.get_xlim(),ax.get_ylim()))),
                                max(np.concat((ax.get_xlim(),ax.get_ylim()))),
                                num=2)
        ax.set_xlim(lim_vals[0], lim_vals[-1])
        ax.set_ylim(lim_vals[0], lim_vals[-1])

        # Add optional histograms
        if histograms:
            # Set bins to use in histogram
            bins = set_hist_bins(ax,num_bins_per_tick=4)

            # Histograms
            ax_histx.hist(x, bins=bins, edgecolor='k')
            ax_histy.hist(y, bins=bins, edgecolor='k', orientation='horizontal')

    # Legend (if legend slice was set)
    if legend_slice:
        leg = ax.legend([', '.join(val)
                        for key in legend_slice for val in legend_slice[key]],
                               bbox_to_anchor=(0.1,0.9),loc='upper left')
        for lh in leg.legend_handles:
            lh.set_alpha(1)

    # Draw ideal line of best fit and actual line of best fit
    ideal_line,bestfit_line = draw_regression_lines(truth_dfs,predict_dfs,column,lim_vals,ax)

    return ideal_line, bestfit_line


def set_hist_bins(ax,num_bins_per_tick=4):
    # Set bins based on x-axis tickmarks
    ticks = ax.get_xticks()
    tickspacing = ticks[1] - ticks[0]
    bins = np.arange(ticks[0], ticks[-1], step=tickspacing / num_bins_per_tick)
    bins -= tickspacing / num_bins_per_tick / 2
    return bins


def check_df_for_slice(x,key,slice_var,slice_type='dict'):
    match slice_type:
        case 'dict':
            if isinstance(slice_var[key],list):
                return x[key] in slice_var[key]
            else:
                return x[key] == slice_var[key]
        case 'list':
            return x[key] in slice_var


def data_slice_to_plot(result, plot_slice, legend_slice, return_lists=True):
    # Copy results dataframes so that the result object's attributes are unmodified
    id_df = result.id_df.copy()
    dfs_to_slice = [getattr(result,df_name) for df_name in ['truths','predicts','pbp_df'] if hasattr(result,df_name)]

    # Slice data based on optional input
    if plot_slice:
        for key in plot_slice:
            indices_to_keep = id_df.apply(
                check_df_for_slice, args=(key,plot_slice), axis=1)
            id_df = id_df[indices_to_keep]
            for i,df in enumerate(dfs_to_slice):
                dfs_to_slice[i] = df[indices_to_keep]

    # Break up into multiple x/y data sets if legend slices are input
    if legend_slice:
        df_lists = [[] for df in dfs_to_slice]
        for key in legend_slice:
            for val in legend_slice[key]:
                indices_to_keep = id_df.apply(check_df_for_slice, args=(key,val,'list'), axis=1)
                for df_list,df in zip(df_lists,dfs_to_slice):
                    df_list.append(df[indices_to_keep])
    else:
        df_lists = []
        if return_lists:
            for df in dfs_to_slice:
                df_lists.append([df])
        else:
            for df in dfs_to_slice:
                df_lists.append(df)

    return df_lists


def draw_regression_lines(truth_dfs, predict_dfs, column, lim_vals, ax):
    # Trendlines: ideal trendline, and line of best fit
    # Re-consolidate x,y datasets
    truth_data = pd.concat(truth_dfs)[column]
    predict_data = pd.concat(predict_dfs)[column]
    if len(truth_data.unique()) > 1:
        # Only plot trendlines if there is more than one unique value on the x axis
        # Draw line along y=x (values should clump along this line for a
        # well-performing model)
        ideal_line, = ax.plot(lim_vals, lim_vals, '--', color='0.4')
        # Draw line of best fit, display r-squared as text on figure
        slope, intercept, r_value = linregress(truth_data, predict_data)[0:3]
        bestfit_line, = ax.plot(lim_vals, slope * lim_vals + intercept, 'k--')
        ax.text(lim_vals[-1],
                slope * lim_vals[-1] + intercept,
                f'$R^2 = {(r_value**2):>0.3f}$ ',
                horizontalalignment='right',
                verticalalignment=('bottom' if r_value > 0 else 'top'))
    else:
        ideal_line = None
        bestfit_line = None
    return ideal_line, bestfit_line


def plot_game_timeline(result, game_id, fig=None):
    # Copy dataframes (to preserve original object attributes)
    # And trim content to only the data spec'd in game_id
    [truth_df,predict_df,pbp_df] = data_slice_to_plot(result, game_id, None, return_lists=False)

    # Generate new figure if not plotting on a pre-existing fig
    if not fig:
        fig = plt.subplots()[0]

    # Plot fantasy points over time
    ax = fig.axes[0]
    ax.plot(pbp_df['Elapsed Time'], pbp_df['Fantasy Points'])
    ax.plot(pbp_df['Elapsed Time'], predict_df['Fantasy Points'])
    ax.plot(pbp_df['Elapsed Time'], truth_df['Fantasy Points'], 'k--')
    ax.legend(['Live Fantasy Score',
               f'{result.predictor_name} Final Fantasy Score',
               'True Final Fantasy Score'])
    ax.set_title(
        f'Fantasy Score, {
            game_id['Player']}, {
            game_id['Year']} Week {
            game_id['Week']}',
            weight='bold'
    )
    ax.set_xlabel('Elapsed Game Time (min)')
    ax.set_ylabel('Fantasy Points')

    # "Display" plot (won't really be displayed until plt.show is called again without block=False)
    plt.show(block=False)

def plot_error_histogram(result, absolute=False, fig=None):
    # Generate new figure if not plotting on a pre-existing fig
    if not fig:
        fig = plt.subplots()[0]

    # Generate histogram
    ax = fig.axes[0]
    ax.hist(result.avg_diff(absolute=absolute), bins=40, density=True, alpha=0.6)
    ax.set_xlabel(f'Fantasy Score{' Absolute' if absolute else ''} Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title('Fantasy Score Prediction Error Distribution Plot', weight='bold')

    plt.show(block=False)
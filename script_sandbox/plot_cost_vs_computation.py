import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from tuners import RandomSearchTuner



def plot_cost_vs_computation(tuners, tuner_names, num_runs, func_name, num_hps):

    # Establish figure
    _, ax = plt.subplots()

    # Format plot axes/labels
    variable_scales = _configure_axes(ax, func_name, num_runs, num_hps)

    # Configure legend - different symbols for each category
    legend_group_markers = ['o','s','h','p']

    for ind, tuner in enumerate(tuners):
        costs = tuner.min_vals_by_group
        costs = np.array([costs]) if costs.size ==1 else np.array(costs)
        num_evals = np.array(range(1,len(costs)+1))*(len(tuner.hyper_tuning_table)/len(costs))

        ax.scatter(
            num_evals, costs,
            marker=legend_group_markers[ind % len(legend_group_markers)],
            linewidth=2, plotnonfinite=True)

        # Data Labels
        for i, _ in enumerate(num_evals):
            if isinstance(tuner,RandomSearchTuner):
                label = (1, int(num_evals[0]*(i+1)))
            else:
                label = (i+1, int(num_evals[0]))
            x_val=num_evals[i]
            if variable_scales[1] == 'linear':
                y_val=costs[i] - 0.2
            elif variable_scales[1] == 'log':
                y_val=costs[i]*0.96

            ax.text(x_val,y_val,s=label,
                    horizontalalignment='center',
                    verticalalignment='top',
                    fontsize='x-small')

    # Legend
    _configure_legend(ax, tuner_names, legend_group_markers)

    plt.show(block=False)


def _configure_axes(ax, func_name, num_runs, num_dimensions):
    # Format plot axes/labels
    variable_scales = ['linear','log']
    # Set axis scale
    ax.set_xscale(variable_scales[0])
    ax.set_yscale(variable_scales[1])

    # Axis labels (with optional formatting)
    ax.set_xlabel('Function Evaluations')
    ax.set_ylabel(f'Min Cost Achieved (avg. of {num_runs} runs)')

    # Figure title
    plt.suptitle('Hyper-Parameter Tuning Algorithm Comparison',weight='bold')
    plt.title(f'Function: {func_name}, Dimensions: {num_dimensions}')

    # Grid lines
    ax.grid(which='major')
    
    return variable_scales


def _configure_legend(ax, tuner_names, legend_group_markers):
    # Generate and place legend
    if len(tuner_names) > 1:
        # Create dummy plots using each legend marker (with no color) in order to get a handle on a plot of each symbol with no formatting
        # new_handles = [ax.plot(np.nan,marker=legend_group_markers[i],ls='',mfc='w',mec='k',ms=10)[0] for i in range(len(tuner_names))]
        ax.legend(labels=[f'{tuner}' for tuner in tuner_names],
                bbox_to_anchor=(0.99,0.99),loc='upper right'
        )

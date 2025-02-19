"""Visualization functions used for displaying the results of Neural Net HyperParameter tuning.

    Functions:
        plot_tuning_results : Generates plot showing the Neural Net performance as two HyperParameters are varied simultaneously.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_tuning_results(filename, param_set, legend_column=None, **kwargs):
    """Generates plot showing the Neural Net performance as multiple HyperParameters are varied simultaneously.

        Caution: will only work for linear or log-scale variables

        Args:
            filename (str): File containing inputs and results for GridSearchTuner hyper-parameter tuning.
            param_set (HyperParameterSet): Group of HyperParameters varied over the course of the tuning process.
            legend_column (str, optional): Column header in csv file to use as a categorizer for different data groups in the plot legend.
                Different data groups will receive different shapes as markers. Defaults to None.

        Keyword-Arguments:
            maximize (bool, optional): whether to maximize (True) or minimize (False) the values returned from eval_function. Defaults to False (minimize).
            plot_variables (tuple | list, optional): 2-element array. Names of the HyperParameters to use as x- and y-axes of plot.
                Defaults to the first two HyperParameters in the HyperParameterSet.

    """

    # Optional Keyword Arguments
    maximize = kwargs.get("maximize", False)
    plot_variables = kwargs.get("plot_variables")

    # Read values and performance from file
    perf_df = pd.read_csv(filename,index_col=0)
    hyper_parameters = perf_df.columns[:perf_df.columns.get_loc("Model Performance")].tolist()

    # Handle optional input of variables to plot - default to first two in list
    if plot_variables is None or not all(var in hyper_parameters for var in plot_variables):
        plot_variables = tuple(perf_df.columns[0:2])

    # Add edge coloring to best performing data point
    perf_df["Edge Color"] = "w"
    optimal_ind = np.nanargmax(perf_df["Model Performance"]) if maximize else np.nanargmin(perf_df["Model Performance"])
    perf_df.loc[optimal_ind,"Edge Color"] = "g"

    # Drop duplicates
    perf_df = perf_df.drop_duplicates(subset=hyper_parameters).reset_index(drop=True)

    # Establish figure and colorbar/colormap
    ax = plt.subplots()[1]
    perf_df, cmap, norm = _set_colors(perf_df, ax)

    # Configure legend - different symbols for each category
    legend_group_markers = ["o","s","h","p"]
    # Column in file can optionally be used as a legend categorizer
    if legend_column is not None:
        legend_groups = perf_df[legend_column].unique()
        data_groups = [perf_df[perf_df[legend_column]==group] for group in legend_groups]
    else:
        legend_groups = []
        data_groups = [perf_df]

    # Iterate through all legend categories
    for ind, data in enumerate(data_groups):
        # Create scatterplot for all data within legend category
        ax.scatter(
            data[plot_variables[0]],data[plot_variables[1]],
            s=400*1/np.sqrt(data["Model Performance"].fillna(100).clip(lower=1,upper=100)), # size varies w/ sqrt
            c=data["ColorVal"],
            marker=legend_group_markers[ind % len(legend_group_markers)],
            edgecolors=data["Edge Color"], linewidth=2,
            cmap=cmap,norm=norm,plotnonfinite=True)

    # Generate and place legend
    if legend_column is not None:
        _configure_legend(ax, legend_groups, legend_group_markers, legend_column)

    # Format plot axes/labels
    _configure_axes(ax, param_set, plot_variables)

    plt.show(block=False)

def _set_colors(perf_df, ax):
    # Scatter plot with customized colorbar
    colorbar_scale = "linear"
    min_color_val = np.nanmin(perf_df["Model Performance"])
    max_color_val = np.nanmax(perf_df["Model Performance"])
    match colorbar_scale:
        case "log":
            norm = mpl.colors.Normalize(vmin=np.log10(min_color_val),vmax=np.log10(max_color_val))
            cbar_ticks = np.log10(np.logspace(np.log10(min_color_val),np.log10(max_color_val),3))
            cbar_tick_labels = np.round(10**np.array(cbar_ticks))
            perf_df["ColorVal"] = np.log10(perf_df["Model Performance"])
        case "linear":
            norm = mpl.colors.Normalize(vmin=min_color_val,vmax=max_color_val)
            cbar_ticks = np.linspace(min_color_val,max_color_val,3)
            cbar_tick_labels = np.round(cbar_ticks)
            perf_df["ColorVal"] = perf_df["Model Performance"]
    cmap = mpl.colormaps["plasma"]
    cmap.set_bad()
    cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),
                       ax=ax,label="Avg. Fantasy Score Error")
    cbar.set_ticks(cbar_ticks,labels=cbar_tick_labels)

    return perf_df, cmap, norm

def _configure_axes(ax, param_set, variables):
    # Format plot axes/labels
    # Obtain info on scale used for each variable
    param_set_names = [hp.name for hp in param_set.hyper_parameters]
    param_set_scales = [hp.val_scale for hp in param_set.hyper_parameters]
    param_set_scales = ["linear" if scale not in ["linear","log"] else scale for scale in param_set_scales]
    variable_scales = [param_set_scales[param_set_names.index(var_name)] for var_name in variables]
    # Set axis scale
    ax.set_xscale(variable_scales[0])
    ax.set_yscale(variable_scales[1])

    # Axis labels (with optional formatting)
    hp_label_formatting = {
        "learning_rate": r"Learning Rate ($\eta$)",
        "lmbda": r"Regularization ($\lambda$)",
    }
    ax.set_xlabel(hp_label_formatting.get(variables[0],variables[0]))
    ax.set_ylabel(hp_label_formatting.get(variables[1],variables[1]))

    # Figure title
    ax.set_title("Hyper-Parameter Tuning Results",weight="bold")

    # Grid lines
    ax.grid(which="major")


def _configure_legend(ax, legend_groups, legend_group_markers, legend_column):
    # Generate and place legend
    if len(legend_groups) > 1:
        # Create dummy plots using each legend marker (with no color) in order to get a handle on a plot of each symbol with no formatting
        new_handles = [ax.plot(np.nan,marker=legend_group_markers[i%len(legend_group_markers)],ls="",mfc="w",mec="k",ms=10)[0] for i in range(len(legend_groups))]
        ax.legend(handles=new_handles,labels=[f"{legend_column} {layer+1}" for layer in legend_groups],
                bbox_to_anchor=(0.06,0.94),loc="upper left",
        )

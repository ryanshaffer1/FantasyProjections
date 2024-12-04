"""Visualization functions used for displaying the results of Neural Net HyperParameter tuning.

    Functions:
        plot_grid_search_results : Generates plot showing the Neural Net performance as two HyperParameters are varied simultaneously.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


def plot_grid_search_results(filename,param_set,variables=None):
    """Generates plot showing the Neural Net performance as two HyperParameters are varied simultaneously.

        Args:
            filename (str): File containing inputs and results for GridSearchTuner hyper-parameter tuning.
            param_set (HyperParameterSet): Group of HyperParameters varied over the course of the tuning process.
            variables (tuple | list, optional): 2-element array. Names of the HyperParameters to use as x- and y-axes of plot. Defaults to None. (Won't work without input...)
    """

    # Caution: this will only work for linear or log-scale variables

    hp_label_dict = {
        'learning_rate': r'Learning Rate ($\eta$)',
        'lmbda': r'Regularization ($\lambda$)'
    }

    # Read values and performance from file
    grid_df = pd.read_csv(filename,index_col=0)


    # Drop duplicates
    grid_df = grid_df.drop_duplicates(subset=['mini_batch_size','learning_rate','lmbda','loss_fn']).reset_index(drop=True)

    # Add edge coloring to best performing data point
    grid_df['Edge Color'] = 'w'
    grid_df.loc[int(np.nanargmin(grid_df['Model Performance'])),'Edge Color'] = 'g'

    # Obtain info on scale used for each variable
    param_set_names = [hp.name for hp in param_set.hyper_parameters]
    param_set_scales = [hp.val_scale for hp in param_set.hyper_parameters]
    variable_scales = [param_set_scales[param_set_names.index(var_name)] for var_name in variables]

    # Scatter plot with customized colorbar
    colorbar_scale = 'linear'
    min_color_val = np.nanmin(grid_df['Model Performance'])
    max_color_val = np.nanmax(grid_df['Model Performance'])
    match colorbar_scale:
        case 'log':
            norm = mpl.colors.Normalize(vmin=np.log10(min_color_val),vmax=np.log10(max_color_val))
            cbar_ticks = np.log10(np.logspace(np.log10(min_color_val),np.log10(max_color_val),3))
            cbar_tick_labels = np.round(10**np.array(cbar_ticks))
            grid_df['ColorVal'] = np.log10(grid_df['Model Performance'])
        case 'linear':
            norm = mpl.colors.Normalize(vmin=min_color_val,vmax=max_color_val)
            cbar_ticks = np.linspace(min_color_val,max_color_val,3)
            cbar_tick_labels = np.round(cbar_ticks)
            grid_df['ColorVal'] = grid_df['Model Performance']
    ax = plt.subplots()[1]
    cmap = 'plasma'
    marker_by_layer = ['o','s','h','p']
    scats = []
    search_layers = grid_df['Grid Search Layer'].unique()
    for layer in grid_df['Grid Search Layer'].unique():
        data = grid_df[grid_df['Grid Search Layer']==layer]
        scat = ax.scatter(data[variables[0]],data[variables[1]],
                        s=400*1/np.sqrt(data['Model Performance'].fillna(100).clip(lower=1,upper=100)), # size varies w/ sqrt
                        #   s=2000*1/data['Model Performance'].fillna(100).clip(lower=1,upper=100), # size varies linearly
                        # s=200*1/np.log10(data['Model Performance'].fillna(100)), # size varies logarithmically
                        c=data['ColorVal'], marker=marker_by_layer[layer],
                        edgecolors=data['Edge Color'], linewidth=2,
                        cmap=cmap,norm=norm,plotnonfinite=True)
        scat.cmap.set_bad()
        scats.append(scat)
    cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),
                       ax=ax,label='Avg. Fantasy Score Error')
    cbar.set_ticks(cbar_ticks,labels=cbar_tick_labels)

    # Legend for multiple search layers
    if len(search_layers) > 1:
        new_handles = [ax.plot(np.nan,marker=marker_by_layer[i],ls='',mfc='w',mec='k',ms=10)[0] for i in range(len(search_layers))]
        ax.legend(handles=new_handles,labels=[f'Grid Search Layer {layer+1}' for layer in search_layers],
                #   bbox_to_anchor=(0.5,1.01),loc='upper center',ncols=len(search_layers))
                bbox_to_anchor=(0.06,0.94),loc='upper left'
        )

    # Format plot axes/labels
    ax.set_xscale(variable_scales[0])
    ax.set_yscale(variable_scales[1])
    ax.set_xlabel(hp_label_dict[variables[0]])
    ax.set_ylabel(hp_label_dict[variables[1]])
    ax.set_title('Hyper-Parameter Grid Search Results',weight='bold')
    ax.grid(which='major')
    plt.show(block=False)

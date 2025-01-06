"""Contains variables defining the configuration of data visualization in Fantasy Projections.

    Variables: 
        scatter_plot_settings (list of dicts): Each entry of the list defines a unique scatterplot figure. 
        Each entry is a dictionary with the following optional fields:
            columns (list, optional): list of the stats (e.g. 'Pass Yds') to plot in the figure. Each element is plotted on a separate subplot. Defaults to None.
            slice (dict, optional): subset of the evaluated dataset to include in the figure. Keys may be 'Position', 'Team', or 'Player Name'. Defaults to empty.
            legend_slice (dict, optional): subsets of the evaluated dataset to split into separate entities in the plot legend. Same keys as slice. Defaults to empty.
            subtitle (str, optional): text to include as a subtitle on the figure. Defaults to None.
            histograms (bool, optional): whether to include histograms on the axes of each subplot. Defaults to False.
"""


# Configure scatter plots
scatter_plot_settings = []
scatter_plot_settings.append({
                    'columns': ['Pass Att',
                                'Pass Cmp',
                                'Pass Yds',
                                'Pass TD',
                                'Int'],
                    'slice': {'Position': ['QB']},
                    'subtitle': 'Passing Stats',
                    'histograms': True})
scatter_plot_settings.append({
                    'columns': ['Pass Att', 'Pass Cmp', 'Pass Yds', 'Pass TD'],
                    'legend_slice': {'Position': [['QB'], ['RB', 'WR', 'TE']]},
                    'subtitle': 'Passing Stats',
                    })
scatter_plot_settings.append({
                    'columns': ['Rush Att', 'Rush Yds', 'Rush TD', 'Fmb'],
                    'subtitle': 'Rushing Stats',
                    'histogram': True
                    })
scatter_plot_settings.append({
                    'columns': ['Rec', 'Rec Yds', 'Rec TD'],
                    'legend_slice': {'Position': [['RB', 'WR', 'TE'], ['QB']]},
                    'subtitle': 'Receiving Stats',
                    })
scatter_plot_settings.append({
                    'columns': ['Fantasy Points'],
                    'subtitle': 'Fantasy Points',
                    'histograms': True
                    })

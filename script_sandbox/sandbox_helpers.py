import numpy as np
import pandas as pd


def get_min_by_layer(tuner, num_hps):
    df = pd.DataFrame(tuner.hyper_tuning_table)
    mins = []
    if len(df.columns) == num_hps+1:
        minval = np.nanmin(df[num_hps])
        return minval
    for group in df[num_hps+1].unique():
        group_min = np.nanmin(df[df[num_hps+1]<=group][num_hps])
        mins.append(group_min)

    return mins


def add_artificial_layers(tuner, layer_size=None):

    if layer_size is None:
        layer_size = len(tuner.hyper_tuning_table)

    new_table = []
    for i, row in enumerate(tuner.hyper_tuning_table):
        row.append(i//layer_size)
        new_table.append(row)

    tuner.hyper_tuning_table = new_table

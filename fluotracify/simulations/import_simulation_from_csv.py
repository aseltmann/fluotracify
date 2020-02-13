import os
from pathlib import Path

import numpy as np
import pandas as pd


def import_from_csv(path, header, frac_train, col_per_example, dropindex,
                    dropcolumns):
    """
    Import a directory of CSV files created by one of the
    fluotracify.simulations methods.

    Parameters
    ----------
    path : str
        Folder which contains .csv files with data
    header : int
        param for pd.read_csv = rows to skip at beginning
    frac_train : float in interval [0.0, 1.0]
        Fraction of sample files used for training. The rest will be used for
        testing. For '1.0', all the data will be loaded in train
    col_per_example : int
        Number of columns per example, first column being a trace, and then
        one or multiple labels
    dropindex : int
        Which indeces in the csv file should be dropped
    dropcolumns : str or int
        Which columns in the csv file should be dropped

    Returns
    -------
    train, test : pandas DataFrames
        Contain training and testing data columnwise in the manner data_1,
        label_1, data_2, label_2, ...
    nsamples : int
        list containing no of examples per file
    experiment_param :  pandas DataFrame
        Contain metadata of the files
    """
    path = Path(path)
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    # sort the file names (os.listdir() returns them in arbitrary order)
    files.sort()
    np.random.seed(0)  # for reproducible random state
    np.random.shuffle(files)

    ntrainfiles = int(round(frac_train * len(files), 0))

    train = pd.DataFrame()
    test = pd.DataFrame()
    nsamples = []
    experiment_params = pd.DataFrame()

    for idx, file in enumerate(files):
        file = Path(file)
        path_and_file = path / file
        raw_dataset = pd.read_csv(path_and_file, sep=',', header=header)
        df = raw_dataset.copy()
        try:
            df = df.drop(index=dropindex, columns=dropcolumns)
        except ValueError:
            pass
        # convert from float64 to float32 and from object to float32
        # -> shrinks memory usage of train dataset from 2.4 GB to 1.2GB
        try:
            converted_float = df.apply(pd.to_numeric, downcast='float')
        except ValueError:
            raise ValueError(
                'Probably the header parameter is too low and points to the metadata. Try a higher value.'
            )
        # save number of examples per file
        nsamples.append(round(len(converted_float.columns) / col_per_example))
        # save some parameters of the experiment from csv file
        experiment_param = pd.read_csv(path_and_file,
                                       sep=',',
                                       header=None,
                                       index_col=0,
                                       usecols=[0, 1],
                                       skipfooter=len(raw_dataset),
                                       squeeze=True)
        experiment_params = pd.concat([experiment_params, experiment_param],
                                      axis=1,
                                      ignore_index=True,
                                      sort=False)

        if idx < ntrainfiles:
            train = pd.concat([train, converted_float], axis=1)
            print('train', idx, path_and_file)
        else:
            test = pd.concat([test, converted_float], axis=1)
            print('test', idx, path_and_file)

    return train, test, nsamples, experiment_params

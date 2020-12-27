from pathlib import Path

import numpy as np
import pandas as pd


def import_from_csv(folder,
                    header,
                    frac_train,
                    col_per_example,
                    dropindex=None,
                    dropcolumns=None):
    """Import CSV files containing data from flotracify.simulations

    Import a directory of CSV files created by one of the
    fluotracify.simulations methods and output two pandas DataFrames containing
    test and train data for machine learning pipelines.

    Parameters
    ----------
    folder : str
        Folder which contains .csv files with data
    header : int
        param for `pd.read_csv` = rows to skip at beginning
    frac_train : float in interval [0.0, 1.0]
        Fraction of sample files used for training. The rest will be used for
        testing. For '1.0', all the data will be loaded in train
    col_per_example : int
        Number of columns per example, first column being a trace, and then
        one or multiple labels
    dropindex : int, optional
        Which indeces in the csv file should be dropped
    dropcolumns : str or int, optional
        Which columns in the csv file should be dropped

    Returns
    -------
    train, test : pandas DataFrames
        Contain training and testing data columnwise in the manner data_1,
        label_1, data_2, label_2, ...
    nsamples : int
        list containing no of examples per file
    experiment_param :  pandas DataFrame
        Contains metadata of the files

    Raises
    ------
    ValueError
        If pandas read_csv fails
    """
    files = list(Path(folder).rglob('*.csv'))
    files.sort()
    np.random.seed(0)  # for reproducible random state
    np.random.shuffle(files)

    ntrainfiles = int(round(frac_train * len(files), 0))

    train = pd.DataFrame()
    test = pd.DataFrame()
    nsamples = []
    experiment_params = pd.DataFrame()

    for idx, file in enumerate(files):
        try:
            raw_dataset = pd.read_csv(file, sep=',', header=header)
        except pd.errors.ParserError:
            raise ValueError('Probably the header parameter is too low '
                             'and points to the metadata. Try a higher value.')
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
            raise ValueError('Probably the header parameter is too low '
                             'and points to the metadata. Try a higher value.')
        # save number of examples per file
        nsamples.append(round(len(converted_float.columns) / col_per_example))
        # save some parameters of the experiment from csv file
        experiment_param = pd.read_csv(file,
                                       sep=',',
                                       header=None,
                                       index_col=0,
                                       usecols=[0, 1],
                                       skipfooter=len(raw_dataset),
                                       squeeze=True,
                                       engine='python')
        experiment_params = pd.concat([experiment_params, experiment_param],
                                      axis=1,
                                      ignore_index=True,
                                      sort=False)

        if idx < ntrainfiles:
            train = pd.concat([train, converted_float], axis=1)
            print('train', idx, file)
        else:
            test = pd.concat([test, converted_float], axis=1)
            print('test', idx, file)

    return train, test, nsamples, experiment_params


def separate_data_and_labels(array, nsamples, col_per_example):
    """Take pandas DataFrame containing feature and label data output a
    dictionary containing them separately

    Parameters
    ----------
    array : pandas DataFrame
        features and labels ordered columnwise in the manner: feature_1,
        label_1, feature_2, label_2, ...
    nsamples : list of int
        list containing no of examples per file
    col_per_example : int
        Number of columns per example, first column being a trace, and then
        one or multiple labels

    Returns
    -------
    array_dict : dict of pandas DataFrames
        Contains one key per column in each simulated example. E.g. if the
        simulated features comes with two labels, the key '0' will be the
        array with the features, '1' will be the array with label A and
        '2' will be the array with label B

    Raises
    ------
    Exception Error, if the number of examples in each file (nsamples) is not
        the same in all of them
    """
    if not len(set(nsamples)) == 1:
        raise Exception(
            'Error: The number of examples in each file have to be the same')

    array_dict = {}

    for i in range(col_per_example):
        array_dict['{}'.format(i)] = array.iloc[:, i::col_per_example]

    array_dict_shapes = [a.shape for a in array_dict.values()]
    print('The given DataFrame was split into {} parts with shapes:'
          ' {}'.format(col_per_example, array_dict_shapes))

    return array_dict

from pathlib import Path

import pandas as pd


def import_dat_to_memory(path, droplastindex=None, dropindex=None, dropcolumns=None):
    """Import .dat files containing Fluorescence Correlation Spectroscopy data

    Parameters
    ----------
    path : str
        Folder which contains .dat files with data
    droplastindex : int, optional
        Number of rows which should be dropped at the end of the data frame.
    dropindex : int, optional
        Arbitrary indeces which should be dropped
    dropcolumns : int or str, optional
        Arbitrary columns which sould be dropped

    Returns
    -------
    dat : pandas DataFrame
        Contains timetraces and histograms from FCS files
    """
    path = Path(path)
    files = list(path.glob('**/*.dat'))
    dat = pd.DataFrame()
    for file in files:
        dat_tmp = pd.read_table(file, sep='\t', header=1)
        if droplastindex is not None:
            dat_tmp = dat_tmp.drop(index=dat_tmp.tail(droplastindex).index)
        try:
            dat_tmp = dat_tmp.drop(index=dropindex, columns=dropcolumns)
        except ValueError:
            pass
        dat = pd.concat([dat, dat_tmp], axis=1)
        print(file, ', shape now:', dat.shape)
    return dat

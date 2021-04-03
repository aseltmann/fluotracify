"""helper functions for Christoph Gohlke's fcsfiles module for reading
Fluorescence Correlation Spectroscopy files
"""
from pathlib import Path

import fcsfiles
import matplotlib.pyplot as plt
import pandas as pd
from chardet.universaldetector import UniversalDetector


def check_encoding(filepath, openmode):
    """Check encoding of file. It should be utf-8 if you want to use fcsfiles

    More info: https://chardet.readthedocs.io/en/latest/usage.html

    Parameters
    ----------
    filepath : str
        path to file which should be checked
    openmode : str
        mode for Python's inbuilt open()
    """
    detector = UniversalDetector()
    with open(filepath, openmode) as file:
        for line in file.readlines():
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    print(detector.result)


def print_metadata_from_fcsfiles(fcs, fcs_entry_idx):
    """Print selected metadata from Fluorescence Correlation Spectroscopy read
    in through Christoph Gohlke's fcsfiles module

    Parameters
    ----------
    fcs : dict
        FCS file as read in through the fcsfiles module
    fcs_entry_idx : int
        Which trace index to choose
    """
    print('traces: ', len(fcs['FcsData']['FcsEntry']))

    metadata_set = set([
        'DateTime', 'Channel', 'MeasurementTime', 'CorrelatorBinning',
        'CorrelatorTauChannels', 'CountRateBinnig',
        'PhotonCountHistogramBinnig'
    ])
    for key, value in fcs['FcsData']['FcsEntry'][fcs_entry_idx]['FcsDataSet'][
            'Acquisition']['AcquisitionSettings'].items():
        if key in metadata_set:
            print(key, value)


def plots_from_fcsfiles(fcs, fcs_entry_idx):
    """Plots data from Fluorescence Correlation Spectroscopy read in through
    Christoph Gohlke's fcsfiles module

    Parameters
    ----------
    fcs : dict
        FCS file as read in through the fcsfiles module
    fcs_entry_idx : int
        Which trace index to choose
    """
    _, axes = plt.subplots(2, 2, figsize=(16, 9))
    for key, value in fcs['FcsData']['FcsEntry'][fcs_entry_idx][
            'FcsDataSet'].items():
        if key == 'CountRateArray':
            axes[0, 0].set_title('{} with shape: {}'.format(key, value.shape))
            axes[0, 0].plot(value[:, 0], value[:, 1])
        elif key == 'CorrelationArray':
            axes[0, 1].set_title('{} with shape: {}'.format(key, value.shape))
            axes[0, 1].semilogx(value[:, 0], value[:, 1])
        elif key == 'PhotonCountHistogramArray':
            axes[1, 0].set_title('{} with shape: {}'.format(key, value.shape))
            bins = value[:, 0]
            counts = value[:, 1]
            axes[1, 0].hist(x=bins, bins=bins, weights=counts, log=True)
        elif key == 'PulseDistanceHistogramArray':
            axes[1, 1].set_title('{} with shape: {}'.format(key, value.shape))
            bins = value[:, 0]
            counts = value[:, 1]
            axes[1, 1].hist(x=bins, bins=bins, weights=counts, log=True)
    plt.show()


def plot_countratearray_from_fcsfiles(fcs):
    """Plot CountRateArray from Fluorescence Correlation Spectroscopy read in
    through Christoph Gohlke's fcsfiles module

    Parameters
    ----------
    fcs : dict
        FCS file as read in through the fcsfiles module
    """
    for idx in range(len(fcs['FcsData']['FcsEntry'])):
        trace = fcs['FcsData']['FcsEntry'][idx]['FcsDataSet']['CountRateArray']
        trace_length = fcs['FcsData']['FcsEntry'][idx]['FcsDataSet'][
            'Acquisition']['AcquisitionSettings']['MeasurementTime']
        timestep = fcs['FcsData']['FcsEntry'][idx]['FcsDataSet'][
            'Acquisition']['AcquisitionSettings']['CountRateBinnig']
        datetime = fcs['FcsData']['FcsEntry'][idx]['FcsDataSet'][
            'AcquisitionTime']
        # of our 12 measurements, index 5 and index 11 are empty
        if len(trace) > 0:
            print(idx, datetime, trace_length, timestep, trace.shape)
            plt.plot(trace[:, 0], trace[:, 1])
            plt.show()


def import_fcsfiles_countrates_to_memory(path):
    """Import .fcs files containing Fluorescence Correlation Spectroscopy data

    Uses Christoph Gohlke's fcsfiles module. Imports only CountRateArray (the
    timetraces) for further processing. Binning is written in the columns.
    Parameters
    ----------
    path : str
        Folder which contains .fcs files with data

    Returns
    -------
    fcs : pandas DataFrame
        Contains timetraces ('CountRateArray') from FCS files

    Raises
    ------
    FileNotFoundError
        if the path provided does not include any .fcs files
    """
    path = Path(path)
    files = list(path.glob('**/*.fcs'))
    if len(files) == 0:
        raise FileNotFoundError('The path provided does not include any'
                                ' .fcs files.')
    fcs = pd.DataFrame()
    for file in files:
        fcs_tmp = fcsfiles.ConfoCor3Fcs(file)
        for idx, _ in enumerate(fcs_tmp['FcsData']['FcsEntry']):
            trace_tmp = fcs_tmp['FcsData']['FcsEntry'][idx]['FcsDataSet'][
                'CountRateArray']
            if len(trace_tmp) > 0:
                trace_tmp = pd.DataFrame(trace_tmp)
                trace_binning = fcs_tmp['FcsData']['FcsEntry'][idx][
                    'FcsDataSet']['Acquisition']['AcquisitionSettings'][
                        'CountRateBinnig']
                trace_tmp.columns = [
                    'time in s, binning: {}'.format(trace_binning),
                    'counts in AU'
                ]
                fcs = pd.concat([fcs, trace_tmp], axis=1)
        print(file, ', shape now:', fcs.shape)
    return fcs

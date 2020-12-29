"""This module contains functions to examine and plot simulated fluorescence
timetraces with artifacts"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluotracify.applications import correlate


def correct_correlation_by_label(ntraces, traces_of_interest,
                                 labels_of_interest, fwhm):
    """Given corrupted traces with boolean label information, this function
    will return the diffusion rates, transit times and length of traces after
    taking only non-corrupted parts of the traces for correlation (where
    labels_of_interest=False).

    Parameters
    ----------
    ntraces : int
        Number of ntraces from given DataFrames to choose for correlation
    traces_of_interest : Pandas DataFrame
        Contains the traces columnwise
    labels_of_interest : Pandas DataFrame
        Contains the labels columnwise

    Returns
    -------
    tuple of lists
    - diffrates_corrected_bylabel
        diffusion rates in micrometer^2 / s
    - transit_times_corrected_bylabel
        transit times in ms
    - tracelen_corrected_bylabel
        lengths of traces after correction
    """

    diffrates_corrected_bylabel = []
    transit_times_corrected_bylabel = []
    tracelen_corrected_bylabel = []
    for ntraces_index in range(ntraces):
        idx_corrected_bylabel = []
        for idx, label in enumerate(labels_of_interest.iloc[:, ntraces_index]):
            if not label:
                idx_corrected_bylabel.append(idx)

        trace_corrected_bylabel = traces_of_interest.iloc[:, ntraces_index]
        trace_corrected_bylabel = trace_corrected_bylabel.take(
            idx_corrected_bylabel, axis=0).values

        # multipletau does not accept traces which are too short
        if len(trace_corrected_bylabel) < 32:
            continue

        diff_corrected_bylabel, trans_corrected_bylabel, _ = correlate.correlate(
            trace=trace_corrected_bylabel.astype(np.float64),
            fwhm=fwhm,
            diffrate=None,
            time_step=1.,
            verbose=False)
        diffrates_corrected_bylabel.append(diff_corrected_bylabel)
        transit_times_corrected_bylabel.append(trans_corrected_bylabel)
        tracelen_corrected_bylabel.append(len(trace_corrected_bylabel))

    return (diffrates_corrected_bylabel, transit_times_corrected_bylabel,
            tracelen_corrected_bylabel)


def plot_distribution_of_correlations_by_label_thresholds(
        diffrate_of_interest, thresh_arr, xunit, artifact, xunit_bins,
        diffrates, features, labels):
    """Examine ensemble correction of simulated fluorescence traces with artifacts

    The features (=fluorescence traces), labels (=ground truth of corruptions
    for each time step of a fluorescence trace), and diffrates have to be
    pandas DataFrames and their indeces have to match. Also, this function
    right now only works properly, if you read in the data via
    fluotracify.simulations.import_simulation_from_csv and if there are 100
    traces in each file.

    Parameters
    ----------
    diffrate_of_interest : float
        diffusion rate used to simulate the traces of interest
    thresh_arr : list of float
        The different thresholds you want to apply on the label data for
        correction
    xunit : {0, 1}
        0: plot histogram of diffusion rates in um^2 / s
        1: plot histogram of transit times in ms
    artifact : {0, 1, 2}
        0: bright clusters / bursts
        1: detector dropout
        2: photobleaching
    xunit_bins : e.g. np.arange(min, max, step)
        Binning of the histogram of xunit, check matplotlib.pyplot.hist
        documentation
    diffrates : pandas DataFrame
        Contains the simulated diffusion rates for each read-in CSV file
        (Diffusion rate stays the same for each file, each file contains 100
        traces)
    features : pandas DataFrame
        Contains simulated fluorescence traces
    labels : pandas DataFrame
        Contains ground truth information of the simulated artifacts for
        each time step
    Returns
    -------
    Plot with subplot of xunit on the left and subplot of trace lengths on the
    right

    Raises
    ------
    ValueError
    - if xunit not {0, 1} or artifact not {0, 1, 2}

    Notes
    -----
    Parameter fwhm is fixed to 250. Needs to be changed, if another FWHM is
    simulated
    """

    fwhm = 250
    # Calculate expected transit time for title of plot
    transit_time_expected = ((float(fwhm) / 1000)**2 *
                             1000) / (diffrate_of_interest * 8 * np.log(2.0))

    # get pandas Series of diffrates of interest
    diff = diffrates.where(diffrates == diffrate_of_interest).dropna()

    traces_of_interest = pd.DataFrame()
    for diff_idx in diff.index:
        # number of files read in is 100, every 100th trace belongs to the
        # same file (and the same diffusion coefficient)
        traces_tmp = features.iloc[:, diff_idx::100]
        traces_of_interest = pd.concat([traces_of_interest, traces_tmp],
                                       axis=1)

    ntraces = len(traces_of_interest.columns)

    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    for idx, thresh in enumerate(thresh_arr):
        print(idx, thresh)
        labels_of_interest = pd.DataFrame()
        for diff_idx in diff.index:
            labels_tmp = labels.iloc[:, diff_idx::100]
            labels_of_interest = pd.concat([labels_of_interest, labels_tmp],
                                           axis=1)

        if artifact == 0:
            labels_of_interest = labels_of_interest > thresh
        elif artifact == 1:
            labels_of_interest = labels_of_interest < thresh
        elif artifact == 2:
            labels_of_interest = labels_of_interest > thresh
        else:
            raise ValueError('value for artifact has to be 0, 1 or 2')

        out = correct_correlation_by_label(
            ntraces=ntraces,
            traces_of_interest=traces_of_interest,
            labels_of_interest=labels_of_interest,
            fwhm=fwhm)

        ax1 = fig.add_subplot(gs[:, :-1])
        ax1.hist(out[xunit],
                 bins=xunit_bins,
                 alpha=0.5,
                 label='threshold: {}'.format(thresh))
        ax2 = fig.add_subplot(gs[:, -1])
        ax2.hist(out[2],
                 bins=np.arange(0, 20001, 1000),
                 alpha=0.5,
                 label='threshold: {}'.format(thresh))
    if xunit == 0:
        ax1.set_title(
            r'Diffusion rates by correlation with expected value '
            r'{:.2f}$\mu m^2/s$'.format(diffrate_of_interest),
            fontsize=20)
        ax1.set_xlabel(r'Diffusion coefficients in $\mu m^2/s$', size=16)
        ax1.axvline(x=diffrate_of_interest,
                    color='r',
                    label='simulated diffusion rate')
    elif xunit == 1:
        ax1.set_title(
            r'Transit times by correlation with expected value {:.2f}$ms$'.
            format(transit_time_expected),
            fontsize=20)
        ax1.set_xlabel(r'Transit times in $ms$', size=16)
        ax1.axvline(x=transit_time_expected,
                    color='r',
                    label='simulated transit time')
    else:
        raise ValueError('value for xunit has to be 0 or 1')
    ax1.set_ylabel('Number of fluorescence traces', fontsize=16)

    ax1.legend(fontsize=16)
    ax2.set_title('Length of traces for correlation', fontsize=20)
    ax2.set_xlabel('Length of traces in $ms$', fontsize=16)
    ax2.set_ylabel('Number of fluorescence traces', fontsize=16)

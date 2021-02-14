"""This module contains functions to examine and plot simulated fluorescence
timetraces with artifacts"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluotracify.applications import correlate
from fluotracify.simulations import import_simulation_from_csv as isfc


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


def plot_traces_by_diffrates(ntraces, col_per_example, diffrate_of_interest,
                             data_label_array, experiment_params, nsamples,
                             artifact):
    """A plot to examine simulated traces via
       fluotracify.simulations.simulate_trace_with_artifacts

    Parameters
    ----------
    ntraces : int
        The number of traces you want to plot. It determines the size of the
        plot as well, where columns are fixed at 6 and depending on ntraces
        and col_per_example the number of rows is determined.
    col_per_example : int
        Number of columns per example, first column being a trace, and then
        one or multiple labels
    diffrate_of_interest : float
        diffusion rate used to simulate the traces of interest
    data_label_array : dict of pandas DataFrames
        Contains one key per column in each simulated example. E.g. if the
        simulated features comes with two labels, the key '0' will be the
        array with the features, '1' will be the array with label A and
        '2' will be the array with label B.
    experiment_params : pandas DataFrame
        Contains metadata of the files (is obtained while loading the
        simulated data with the `import_from_csv` function from the
        fluotracify.simulations.import_simulation_from_csv module)
    nsamples : int
        Number of traces per .csv file (is obtained while loading the
        simulated data with the `import_from_csv` function from the
        fluotracify.simulations.import_simulation_from_csv module)
    artifact : {0, 1, 2, 3}
        0 = no artifact, 1 = bright clusters, 2 = detector dropout,
        3 = photobleaching

    Returns
    -------
    Plot of fluorescence traces and labels from their simulations
    """
    drates = experiment_params.loc[
        'diffusion rate of molecules in micrometer^2 / s']
    # get indices of diffusion rates of interest
    dindices = drates.index.where(drates == str(diffrate_of_interest))
    dindices = dindices.dropna().astype(int)
    # get indices of first of each of the <nsamples> traces per file as an example
    tindices = dindices * nsamples
    if artifact == 1:
        nclusts = experiment_params.loc['number of slow clusters'][dindices]
        dclusts = experiment_params.loc[
            'diffusion rate of clusters in micrometer^2 / s'][dindices]

    cols = 6
    rows = int(ntraces // (cols / col_per_example) +
               (ntraces % (cols / col_per_example) > 0))
    # share y axis only if col_per_example is 1
    sharey = col_per_example == 1
    fig, ax = plt.subplots(rows,
                           cols,
                           figsize=(cols * 4, rows * 4),
                           sharex=True,
                           sharey=sharey)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none',
                    top=False,
                    bottom=False,
                    left=False,
                    right=False)
    plt.grid(False)
    plt.xlabel("time steps in $ms$", fontsize=20)
    plt.ylabel('fluorescence intensity in $a.u.$', labelpad=20, fontsize=20)
    suptitle_height = 1 - (fig.get_figheight() * 0.004)
    plt.suptitle(t='Simulated Fluorescence Traces With D = {} '
                 '$\\frac{{\mu m^2}}{{s}}$'.format(diffrate_of_interest),
                 y=suptitle_height,
                 fontsize=20)
    traceid = 0
    for idx in range(rows):
        for jdx in range(0, cols, col_per_example):
            # first plot the trace
            try:
                ax[idx,
                   jdx].plot(data_label_array['0'].iloc[:, tindices[traceid]])
            except IndexError:
                break
            if artifact == 1:
                ax[idx, jdx].set_title('trace {} ({} clusters, $D_c$ = '
                                       '{} $\\frac{{\mu m^2}}{{s}}$)'.format(
                                           traceid + 1, nclusts.iloc[traceid],
                                           dclusts.iloc[traceid]))
            else:
                ax[idx, jdx].set_title('trace {}'.format(traceid + 1))
            ax[idx, jdx].set_ylim(0, 12000)
            for kdx in range(1, col_per_example):
                # then plot the labels, if they are given
                ax[idx, jdx + kdx].plot(
                    data_label_array['{}'.format(kdx)].iloc[:,
                                                            tindices[traceid]])
                ax[idx, jdx + kdx].set_title('label {}, type {}'.format(
                    traceid + 1, kdx))
            traceid += 1
    plt.show()

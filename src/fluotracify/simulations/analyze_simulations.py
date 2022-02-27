"""This module contains functions which are used to analyze the simulated fluorescence traces"""

import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluotracify.applications import correction, correlate


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

        (diff_corrected_bylabel, trans_corrected_bylabel,
         _) = correlate.correlate_and_fit(
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


def correlate_simulations_corrected_by_prediction(model,
                                                  lab_thresh,
                                                  pred_thresh,
                                                  artifact,
                                                  model_type,
                                                  experiment_params,
                                                  nsamples,
                                                  features,
                                                  labels_artifact,
                                                  labels_puretrace,
                                                  save_as_csv=False):
    """plot the distribution of correlations after correcting fluorescence
    traces corrupted by the given artifact applying different thresholds
    caveat: The parameters fwhm, win_len and zoomvector are currently fixed
    to reduce the number of parameters of this function, but can easily be
    introduced as parameters.

    Parameters
    ----------
    model
        Model used to correct the corrupted traces
    lab_thresh : float
        Threshold you want to apply on the label data
    pred_thresh : float between 0 and 1
        Threshold you want to apply on the predictions
    artifact : {0, 1, 2}
        0: bright clusters / bursts
        1: detector dropout
        2: photobleaching
    model_type : {0, 1}
        0: vgg
        1: unet
    experiment_params : pandas DataFrame
        Contains metadata of the files (is obtained while loading the
        simulated data with the `import_from_csv` function from the
        fluotracify.simulations.import_simulation_from_csv module)
    nsamples : int
        Number of traces per .csv file (is obtained while loading the
        simulated data with the `import_from_csv` function from the
        fluotracify.simulations.import_simulation_from_csv module)
    features : pandas DataFrame or np.array with two axes
        Contains simulated fluorescence traces
    labels_artifact : pandas DataFrame or np.array with two axes
        Contains ground truth information of the simulated artifacts for
        each time step
    labels_puretrace : pandas DataFrame or np.array
        Contains ground truth information of the simulated trace without
        the addition of artifacts
    save_as_csv : bool, optional
        If True, saves a .csv of 'out' in the default directory

    Returns
    -------
    out : pandas DataFrame with the following columns:
        'Simulated $D$': simulated diffusion
            coefficients of small molecules we wanted to examine with FCS
        'Simulated $D_{{clust}}$': simulated diffusion
            coefficients of the bright clusters which disturb our fluorescence
            trace
        'nmol': number of small molecules we wanted to examine with FCS
        '$D$ in $\\frac{{\mu m^2}}{{s}}$': diffusion coefficients
        '$\\tau_{{D}}$ in $ms$': transit times
        'Trace lengths': lengths of the fluorescence trace at the time they
            were correlated.
        'Traces used': 4 possible values: 'corrupted without correction', 'pure
            traces (control)', 'corrected by labels (control)', 'corrected by
            predictions'

    Notes
    -----
    - during one of the correction algorithms (correction by label, correction
    by unet, correction by vgg), it could happen that less than 32 time steps
    of the trace remain. Then the diffusion rates and transit times will be
    given as np.nan values, since the trace can not be correlated using
    multipletau.
    """
    fwhm = 250
    win_len = 128  # only for model_type 0 (vgg)
    zoomvector = (5, 21)  # only for model_type 0 (vgg)
    length_delimiter = 16384

    if not len(set(nsamples)) == 1:
        raise Exception(
            'Error: The number of examples in each file have to be the same')

    nsamples = next(iter(set(nsamples)))

    try:
        diffrates = experiment_params.loc[
            'diffusion rate of molecules in micrometer^2 / s'].astype(
                np.float32)
        clusters = experiment_params.loc[
            'diffusion rate of clusters in micrometer^2 / s'].astype(
                np.float32)
        nmols = experiment_params.loc['number of fast molecules'].astype(
            np.float32)
    except AttributeError:
        print('Make sure that experiment_params is a pandas DataFrame')
    except KeyError:
        print('Make sure that experiment_params is a pandas DataFrame created'
              ' while using the import_simulation_from_csv module')
    ntraces = np.size(features, axis=1)

    if artifact in (0, 2):
        labels_artifact = labels_artifact > lab_thresh
    elif artifact == 1:
        labels_artifact = labels_artifact < lab_thresh
    else:
        raise ValueError('value for artifact has to be 0 (bright clusters), 1'
                         ' (detector dropout) or 2 (photobleaching)')
    features = pd.DataFrame(features).astype(np.float64)
    labels_puretrace = pd.DataFrame(labels_puretrace).astype(np.float64)

    lab_out = correct_correlation_by_label(ntraces=ntraces,
                                           traces_of_interest=features,
                                           labels_of_interest=labels_artifact,
                                           fwhm=fwhm)
    print('processed correlation of {} traces with correction by'
          ' label'.format(len(lab_out[0])))

    if model_type == 0:
        pred_out = correction.correct_correlation_by_vgg_prediction(
            ntraces=ntraces,
            traces_of_interest=features,
            model=model,
            pred_thresh=pred_thresh,
            win_len=win_len,
            zoomvector=zoomvector,
            fwhm=fwhm)
    elif model_type == 1:
        pred_out = correction.correct_correlation_by_unet_prediction(
            ntraces=ntraces,
            traces_of_interest=features,
            model=model,
            pred_thresh=pred_thresh,
            length_delimiter=length_delimiter,
            fwhm=fwhm)
    else:
        raise ValueError('value for model_type has to be 0 (vgg) or 1 (unet)')
    print('processed correlation of {} traces with correction by'
          ' prediction'.format(len(pred_out[0])))

    corrupt_out = correlate.correlation_of_arbitrary_trace(
        ntraces=ntraces, traces_of_interest=features, fwhm=fwhm, time_step=1.)
    print('processed correlation of {} traces without correction'.format(
        len(corrupt_out[0])))

    pure_out = correlate.correlation_of_arbitrary_trace(
        ntraces=ntraces,
        traces_of_interest=labels_puretrace,
        fwhm=fwhm,
        time_step=1.)
    print('processed correlation of pure {} traces'.format(len(pure_out[0])))

    data_diffrates = np.concatenate(
        (corrupt_out[0], pure_out[0], lab_out[0], pred_out[0]), axis=0)
    data_transittimes = np.concatenate(
        (corrupt_out[1], pure_out[1], lab_out[1], pred_out[1]), axis=0)
    data_tracelengths = np.concatenate(
        (corrupt_out[2], pure_out[2], lab_out[2], pred_out[2]), axis=0)
    data_tracesused = np.repeat(
        np.array(('corrupted without correction', 'pure traces (control)',
                  'corrected by labels (control)', 'corrected by prediction')),
        ntraces)
    data_simdiffrates = np.tile(np.repeat(diffrates.values, nsamples), 4)
    data_simclusters = np.tile(np.repeat(clusters.values, nsamples), 4)
    data_nmols = np.tile(np.repeat(nmols.values, nsamples), 4)
    data_out = pd.DataFrame(data=[
        data_simdiffrates, data_simclusters, data_nmols, data_diffrates,
        data_transittimes, data_tracelengths, data_tracesused
    ],
                            index=[
                                'Simulated $D$', 'Simulated $D_{{clust}}$',
                                'nmol', '$D$ in $\\frac{{\mu m^2}}{{s}}$',
                                '$\\tau_{{D}}$ in $ms$', 'Trace lengths',
                                'Traces used'
                            ]).T
    if save_as_csv:
        data_out.to_csv(path_or_buf='{}_correlations.csv'.format(
            datetime.date.today()),
                        na_rep='NaN',
                        index=False)
    return data_out

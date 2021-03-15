"""This module contains functions to apply trained neural networks on
fluorescence timetraces to correct artifacts in them"""

import datetime
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluotracify.applications import correlate
from fluotracify.imports import ptu_utils as ptu
from fluotracify.training import preprocess_data as ppd


def correct_correlation_by_vgg_prediction(ntraces,
                                          traces_of_interest,
                                          model,
                                          pred_thresh,
                                          win_len,
                                          zoomvector,
                                          fwhm,
                                          verbose=False):
    """Given corrupted traces, this function will apply a trained
    convolutional neural network on the data to correct the traces and return
    the diffusion rates, transit times and length of traces after correlation
    Parameters
    ----------
    ntraces : Int
        Number of ntraces from given DataFrames to choose for correlation
    traces_of_interest : Pandas DataFrame
        Contains the traces columnwise
    model
        model loaded by tf.keras.models.load_model, should be compiled already
    pred_thresh : float between 0 and 1
        If prediction is lower, it is assumed to show 'no corruption'
    """
    diffrates_corrected_bypred = []
    transit_times_corrected_bypred = []
    tracelen_corrected_bypred = []

    features_orig, features_prepro, _num_examples, _ = ppd.vgg_preprocessing(
        features_df=traces_of_interest,
        win_len=win_len,
        ntraces_index=0,
        ntraces_delimiter=ntraces,
        zoomvector=zoomvector,
        verbose=False)

    for ntraces_index in range(ntraces):
        chunks_index = np.r_[np.arange(ntraces_index, _num_examples,
                                       ntraces)].astype(np.int64)
        predtrace = features_prepro.take(chunks_index, axis=0)
        predictions = model.predict(predtrace, verbose=0)
        idx_corrected_bypred = []
        for jdx, pred in enumerate(predictions):
            if pred < pred_thresh:
                idx_corrected_bypred.append(jdx)

        trace_corrected_bypred = features_orig[:, :, 0].take(chunks_index,
                                                             axis=0)
        trace_corrected_bypred = trace_corrected_bypred.take(
            idx_corrected_bypred, axis=0).reshape(-1)

        # multipletau does not accept traces which are too short
        if len(trace_corrected_bypred) < 32:
            continue

        diff_corrected_bypred, trans_corrected_bypred, _ = correlate.correlate(
            trace=trace_corrected_bypred,
            fwhm=fwhm,
            diffrate=None,
            time_step=1.,
            verbose=False)
        diffrates_corrected_bypred.append(diff_corrected_bypred)
        transit_times_corrected_bypred.append(trans_corrected_bypred)
        tracelen_corrected_bypred.append(len(trace_corrected_bypred))

        if verbose:
            plt.figure(figsize=(16, 9))
            plt.suptitle('Correction and Correlation of trace '
                         '{}'.format(ntraces_index + 1),
                         fontsize=20,
                         y=1.05)
            trace_corrupted = features_orig[:, :, 0].take(chunks_index,
                                                          axis=0).reshape(-1)
            x_fill_pred = np.arange(0, len(trace_corrupted), 1)
            where_fill_pred = np.repeat(predictions >= pred_thresh, win_len)
            plt.subplot(221)
            plt.ylabel('intensity in AU', fontsize=12)
            plt.xlabel('time steps in ms', fontsize=12)
            plt.title('Fluorescence trace with bright cluster artifact',
                      fontsize=16)
            plt.plot(trace_corrupted)
            plt.fill_between(x=x_fill_pred,
                             y1=min(trace_corrupted),
                             y2=max(trace_corrupted),
                             where=where_fill_pred,
                             facecolor='g',
                             alpha=0.5,
                             label='Corrupted chunks (Prediction)')
            plt.subplot(222)
            print('correlation of corrupted trace:')
            diff_corrupted, trans_corrupted, _ = correlate.correlate(
                trace=trace_corrupted,
                fwhm=fwhm,
                diffrate=None,
                time_step=1.,
                verbose=True)
            plt.title(r'Correlation of trace with artefacts (D = {:0.4f} $\mu '
                      'm^2$ / $s$)'.format(diff_corrupted),
                      fontsize=16)
            plt.ylabel(r'G($\tau$)', fontsize=12)
            plt.xlabel(r'delay time $\tau$ in ms', fontsize=12)
            plt.subplot(223)
            plt.plot(trace_corrected_bypred)
            plt.ylabel('intensity in AU', fontsize=12)
            plt.xlabel('time steps in ms', fontsize=12)
            plt.title('Fluorescence trace corrected by prediction',
                      fontsize=16)
            plt.subplot(224)
            print('correlation of trace corrected by prediction')
            diff_corrected_bypred, trans_corrected_bypred, _ = correlate.correlate(
                trace=trace_corrected_bypred,
                fwhm=fwhm,
                diffrate=None,
                time_step=1.,
                verbose=True)
            plt.title(
                r'Correlation of trace corrected by prediction (D = '
                r'{:0.4f} $\mu m^2$ / $s$)'.format(diff_corrected_bypred),
                fontsize=16)
            plt.ylabel(r'G($\tau$)', fontsize=12)
            plt.xlabel(r'delay time $\tau$ in ms', fontsize=12)
            plt.tight_layout()
            plt.show()

    return (diffrates_corrected_bypred, transit_times_corrected_bypred,
            tracelen_corrected_bypred)


def correct_correlation_by_unet_prediction(ntraces,
                                           traces_of_interest,
                                           model,
                                           pred_thresh,
                                           fwhm,
                                           length_delimiter=None,
                                           traces_for_correlation=None,
                                           bin_for_correlation=None,
                                           verbose=False):
    """Given corrupted traces, this function will apply a trained
    convolutional neural network on the data to correct the traces and return
    the diffusion rates, transit times and length of traces after correlation

    Parameters
    ----------
    ntraces : Int
        Number of ntraces from given DataFrames to choose for correlation
    traces_of_interest : Pandas DataFrame
        Contains the traces columnwise
    model
        model loaded by tf.keras.models.load_model, should be compiled already
    pred_thresh : float between 0 and 1
        If prediction is lower, it is assumed to show 'no corruption'
    fwhm : float
        The full width half maximum of the excitation beam in nm. Used for
        Calculation of the diffusion coefficient.
    length_delimiter : int, optional
        Length of the output traces in the returned dataset. If None, then the
        whole length of the DataFrame is used
    traces_for_correlation : pandas DataFrame or None, optional
        If None, traces_of_interest is used for prediction and correlation. If
        an extra pd DataFrame is supplied, the artifacts will be predicted
        using traces_of_interest, and the correction and correlation will be
        done on traces_for_correlation. This makes sense, if the diffusion
        processes are too fast for a binning window of 1ms, so the same traces
        with a smaller binning window can be supplied in traces_for_correlation
    bin_for_correlation : integer, optional
        Size of bin in ns which was used to construct the time trace.
        E.g. 1e6 gives a time trace binned to ms, 1e3 gives a time trace binned
        to us. It is assumed that the bin for the prediction traces is 1e6 and
        that the bin_for_correlation is smaller. If the bin_for_correlation is
        1e6 or higher, the traces_for_correlation are not used.

    Returns
    -------
    out : tuple of lists
        diffrates_corrected_bypred : list of float
            Contains diffusion rates corrected by prediction
        transit_times_corrected_bypred : list of float
            Contains transit times corrected by prediction
        tracelen_corrected_bypred : list of int
            Contains lengths of traces after correction by prediction
    """
    if traces_for_correlation is not None:
        if len(traces_for_correlation.columns) != len(
                traces_of_interest.columns):
            raise ValueError('traces_for_correlation and traces_of_interest'
                             'have to have the same column number, since they'
                             'have to come from the same tcspc data')

    if ntraces is None:
        ntraces = len(traces_of_interest.columns)

    diffrates_corrected_bypred = []
    transit_times_corrected_bypred = []
    tracelen_corrected_bypred = []

    for ntraces_index in range(ntraces):
        # this returns only 1 trace
        features_orig, features_prepro = ppd.unet_preprocessing(
            features_df=traces_of_interest,
            length_delimiter=length_delimiter,
            ntraces_index=ntraces_index)

        predictions = model.predict(features_prepro, verbose=0)
        predictions = predictions.flatten()

        if traces_for_correlation is None or bin_for_correlation >= 1e6:
            pass
        else:
            repeat_pred_by = int(1e6 / bin_for_correlation)
            predictions = np.repeat(predictions, repeat_pred_by)

        idx_corrected_bypred = []
        for jdx, pred in enumerate(predictions):
            if pred < pred_thresh:
                idx_corrected_bypred.append(jdx)

        if traces_for_correlation is None or bin_for_correlation >= 1e6:
            time_step_for_correlation = 1.
        else:
            features_orig = traces_for_correlation.iloc[:, ntraces_index:(
                ntraces_index + 1)]
            features_orig = features_orig.dropna()
            features_orig = np.array(features_orig).flatten()
            features_orig = features_orig[:(length_delimiter * repeat_pred_by)]
            # time_step_for_correlation = float(bin_for_correlation / 1e6)
            time_step_for_correlation = 1.

        trace_corrected_bypred = features_orig.take(idx_corrected_bypred,
                                                    axis=0).astype(np.float64)

        # multipletau does not accept traces which are too short
        if len(trace_corrected_bypred) < 32:
            continue

        diff_corrected_bypred, trans_corrected_bypred, _ = correlate.correlate(
            trace=trace_corrected_bypred,
            fwhm=fwhm,
            diffrate=None,
            time_step=time_step_for_correlation,
            verbose=False)

        diffrates_corrected_bypred.append(diff_corrected_bypred)
        transit_times_corrected_bypred.append(trans_corrected_bypred)
        if traces_for_correlation is None or bin_for_correlation >= 1e6:
            tracelen_corrected_bypred.append(len(trace_corrected_bypred))
        else:
            tracelen_corrected_bypred.append(
                len(trace_corrected_bypred) / repeat_pred_by)

        if verbose:
            plt.figure(figsize=(16, 9))
            plt.suptitle('Correction and Correlation of trace '
                         '{}'.format(ntraces_index + 1),
                         fontsize=20,
                         y=1.05)
            x_fill_pred = np.arange(0, len(predictions), 1)
            where_fill_pred = predictions >= pred_thresh
            plt.subplot(221)
            plt.ylabel('intensity in AU', fontsize=12)
            plt.xlabel('time steps in ms', fontsize=12)
            plt.title('Fluorescence trace with bright cluster artifact',
                      fontsize=16)
            plt.plot(features_orig)
            plt.fill_between(x=x_fill_pred,
                             y1=min(features_orig),
                             y2=max(features_orig),
                             where=where_fill_pred,
                             facecolor='g',
                             alpha=0.5,
                             label='Corrupted chunks (Prediction)')
            plt.subplot(222)
            print('correlation of corrupted trace:')
            diff_corrupted, trans_corrupted, _ = correlate.correlate(
                trace=features_orig.astype(np.float64),
                fwhm=fwhm,
                diffrate=None,
                time_step=time_step_for_correlation,
                verbose=True)
            plt.title(r'Correlation of trace with artefacts (D = {:0.4f} $\mu '
                      'm^2$ / $s$)'.format(diff_corrupted),
                      fontsize=16)
            plt.ylabel(r'G($\tau$)', fontsize=12)
            plt.xlabel(r'delay time $\tau$ in ms', fontsize=12)
            plt.subplot(223)
            plt.plot(trace_corrected_bypred)
            plt.ylabel('intensity in AU', fontsize=12)
            plt.xlabel('time steps in ms', fontsize=12)
            plt.title('Fluorescence trace corrected by prediction',
                      fontsize=16)
            plt.subplot(224)
            print('correlation of trace corrected by prediction')
            diff_corrected_bypred, trans_corrected_bypred, _ = correlate.correlate(
                trace=trace_corrected_bypred,
                fwhm=fwhm,
                diffrate=None,
                time_step=time_step_for_correlation,
                verbose=True)
            plt.title(
                r'Correlation of trace corrected by prediction (D = '
                r'{:0.4f} $\mu m^2$ / $s$)'.format(diff_corrected_bypred),
                fontsize=16)
            plt.ylabel(r'G($\tau$)', fontsize=12)
            plt.xlabel(r'delay time $\tau$ in ms', fontsize=12)
            plt.tight_layout()
            plt.show()
    return (diffrates_corrected_bypred, transit_times_corrected_bypred,
            tracelen_corrected_bypred)


def correct_experimental_traces_from_ptu_by_unet_prediction(
        path_list,
        model,
        pred_thresh,
        photon_count_bin=1e6,
        ntraces=None,
        save_as_csv=False):
    """plot the distribution of correlations after correcting fluorescence
    traces corrupted by the given artifact applying different thresholds
    Parameters
    ----------
    path_list : str or list of str
        Folder or list of folders  which contain .ptu files with data.
    model
        Model used to correct the corrupted traces
    pred_thresh : float or list of floats between 0 and 1
        Threshold or list of thresholds you want to apply on the predictions
    photon_count_bin : integer, optional
        Standard is 1e6, which means binning the tcspc data in the .ptu files
        to ms, which is the binning the neural net was trained for. If you want
        the prediction to be applied to a photon trace with *smaller* binning,
        you can choose this here (e.g. 1e5 means binning of 100us, 1e4 means
        binning of 10us etc). *Larger* binning will be ignored.
    ntraces : int, None, optional
        Number of traces which shall be chosen. If 'None', all available traces
        will be used.
    save_as_csv : bool, optional
        If True, saves a .csv of 'out' in the default directory

    Returns
    -------
    out : pandas DataFrame
        Contains the following columns:
        - calculated diffusion rates
        - calculated transit times
        - the trace lengths of the traces at time of correlation
        - a label giving the folder_id and info if the trace was computed
          from the original trace or after correction by unet prediction
        - photon count bin in ns which was used for the trace

    Raises
    ------
    ValueError
        If there was an error in loading .ptu files
    ValueError
        If photon_count_bin is not a positive integer
    """
    # fix constants (could be introduced as variables)
    fwhm = 250
    # data from Pablo_structured_experiment has length of 10s = 10000ms and
    # unet can at the moment only take powers of 2 as input size
    length_delimiter = 8192

    if not isinstance(path_list, list):
        path_list = [path_list]
    if not isinstance(pred_thresh, list):
        pred_thresh = [pred_thresh]

    ptu_metadata = {}
    data = {}

    for i, p in enumerate(path_list):
        path = Path(p)

        print('Loading dataset {} from path {} with bin=1e6. This can take a'
              ' while...'.format(i + 1, path))
        try:
            ptu_1ms, ptu_metadata['{}'.format(i)] = ptu.import_from_ptu(
                path=path,
                file_delimiter=ntraces,
                photon_count_bin=1e6,
                verbose=True)
        except (ValueError, FileNotFoundError):
            raise ValueError('There was an error in loading .ptu files'
                             'from folder {}'.format(path))

        if photon_count_bin >= 1e6:
            print('Processing correlation of unprocessed dataset {}'.format(i +
                                                                            1))
            data['{}-orig'.format(
                i)] = correlate.correlation_of_arbitrary_trace(
                    ntraces=ntraces,
                    traces_of_interest=ptu_1ms.astype(np.float64),
                    fwhm=fwhm,
                    time_step=1.,
                    length_delimiter=length_delimiter)
            print('Processing correlation with correction by prediction '
                  'of dataset {}'.format(i + 1))
            for thr in pred_thresh:
                data['{}-pred-{}'.format(
                    i, thr)] = correct_correlation_by_unet_prediction(
                        ntraces=ntraces,
                        traces_of_interest=ptu_1ms.astype(np.float64),
                        model=model,
                        pred_thresh=thr,
                        length_delimiter=length_delimiter,
                        fwhm=fwhm)
                data['{}-pred-{}'.format(i, thr)] += (1e6, )
                data['{}-pred-{}'.format(i, thr)] += (len(
                    data['{}-pred-{}'.format(i, thr)][0]), )
        elif photon_count_bin < 1e6:
            # correlate function expects time_step for shifting x-axis
            # time_step_for_correlation = float(photon_count_bin / 1e6)
            # I changed it back to "1." because it gave strange results
            time_step_for_correlation = 1.

            print('Different binning was chosen for correlation. Loading '
                  'dataset {} with bin={}. This can take a while...'.format(
                      i + 1, photon_count_bin))
            ptu_cor, _ = ptu.import_from_ptu(path=path,
                                             file_delimiter=ntraces,
                                             photon_count_bin=photon_count_bin,
                                             verbose=False)
            print('Processing correlation of unprocessed dataset {}'.format(i +
                                                                            1))
            data['{}-orig'.format(
                i)] = correlate.correlation_of_arbitrary_trace(
                    ntraces=ntraces,
                    traces_of_interest=ptu_cor.astype(np.float64),
                    fwhm=fwhm,
                    time_step=time_step_for_correlation,
                    length_delimiter=length_delimiter)
            print('Processing correlation with correction by prediction '
                  'of dataset {}'.format(i + 1))
            for thr in pred_thresh:
                data['{}-pred-{}'.format(
                    i, thr)] = correct_correlation_by_unet_prediction(
                        ntraces=ntraces,
                        traces_of_interest=ptu_1ms,
                        model=model,
                        pred_thresh=thr,
                        length_delimiter=length_delimiter,
                        fwhm=fwhm,
                        traces_for_correlation=ptu_cor.astype(np.float64),
                        bin_for_correlation=photon_count_bin)
                data['{}-pred-{}'.format(i, thr)] += (1e6, )
                data['{}-pred-{}'.format(i, thr)] += (len(
                    data['{}-pred-{}'.format(i, thr)][0]), )
        else:
            raise ValueError('photon_count_bin has to be a positive integer')
        data['{}-orig'.format(i)] += (photon_count_bin, )
        data['{}-orig'.format(i)] += (len(data['{}-orig'.format(i)][0]), )

    data_ntraces = [data[key][4] for key in data]
    data_keys = list(data.keys())
    data_keys = list(map(lambda x, y: (x, ) * y, data_keys, data_ntraces))
    data_keys = np.concatenate((data_keys), axis=None)
    data_diffrates = [data[key][0] for key in data]
    data_diffrates = np.concatenate((data_diffrates), axis=None)
    data_transittimes = [data[key][1] for key in data]
    data_transittimes = np.concatenate((data_transittimes), axis=None)
    data_tracelengths = [data[key][2] for key in data]
    data_tracelengths = np.concatenate((data_tracelengths), axis=None)
    data_photonbins = [data[key][3] for key in data]
    data_photonbins = list(
        map(lambda x, y: (x, ) * y, data_photonbins, data_ntraces))
    data_photonbins = np.concatenate((data_photonbins), axis=None)
    data_out = pd.DataFrame(data=[
        data_diffrates, data_transittimes, data_tracelengths, data_keys,
        data_photonbins
    ],
                            index=[
                                '$D$ in $\\frac{{\mu m^2}}{{s}}$',
                                '$\\tau_{{D}}$ in $ms$', 'Trace lengths',
                                'folder_id-traces_used',
                                'Photon count bin in $ns$'
                            ]).T

    ptum_list = [ptu_metadata[key] for key in ptu_metadata]
    data_metadata = pd.DataFrame()
    for ptum_df in ptum_list:
        # Since we saved it only once, but we compute the correlation 2 times
        # (orig vs pred), we have to append the metadata here two times as well
        ptum = ptum_df.iloc[:, 1::2].T
        for _ in range(len(pred_thresh)):
            data_metadata = pd.concat((data_metadata, ptum), axis=0)
    data_metadata = data_metadata.reset_index(drop=True)
    data_metadata.columns = ptum_list[0].iloc[:, 0].values

    if len(data_out) == len(data_metadata):
        data_out = pd.concat((data_out, data_metadata), axis=1)
    else:
        warnings.warn('Metadata is not saved with data. Reason: the '
                      'correlation algorithm failed for one or more traces '
                      'which were shorter than 32 time steps after correction.'
                      'Since metadata is loaded in the beginning, it is not '
                      'sure, which correlation is missing to ensure proper '
                      'joining of data and metadata.')

    if save_as_csv:
        data_out.to_csv(path_or_buf='{}_correlations.csv'.format(
            datetime.date.today()),
                        na_rep='NaN',
                        index=False)

    return data_out

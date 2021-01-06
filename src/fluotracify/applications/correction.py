"""This module contains functions to apply trained neural networks on
fluorescence timetraces to correct artifacts in them"""

import matplotlib.pyplot as plt
import numpy as np

from fluotracify.applications import correlate
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
        done no traces_for_correlation. This makes sense, if the diffusion
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

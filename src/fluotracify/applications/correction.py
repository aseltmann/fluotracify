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
                                           length_delimiter,
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

    for ntraces_index in range(ntraces):
        # this returns only 1 trace
        features_orig, features_prepro = ppd.unet_preprocessing(
            features_df=traces_of_interest,
            length_delimiter=length_delimiter,
            ntraces_index=ntraces_index)

        predictions = model.predict(features_prepro, verbose=0)
        predictions = predictions.flatten()

        idx_corrected_bypred = []
        for jdx, pred in enumerate(predictions):
            if pred < pred_thresh:
                idx_corrected_bypred.append(jdx)

        trace_corrected_bypred = features_orig.take(idx_corrected_bypred,
                                                    axis=0)

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
            x_fill_pred = np.arange(0, len(features_orig), 1)
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
                trace=features_orig,
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

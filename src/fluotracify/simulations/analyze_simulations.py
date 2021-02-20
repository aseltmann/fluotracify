"""This module contains functions which are used to analyze the simulated fluorescence traces"""

import numpy as np
import pandas as pd

from fluotracify.applications import correction, correlate
from fluotracify.simulations import plot_simulations as ps


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
        '$\tau_{{D}}$ in $ms$': transit times
        'Trace lengths': lengths of the fluorescence trace at the time they
            were correlated.
        'Traces used': 4 possible values: 'corrupted without correction',
        'corrected by predictions', 'corrected by labels (control)', 'pure
            traces (control)'
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
    nfiles = ntraces / nsamples

    if artifact in (0, 2):
        labels_artifact = labels_artifact > lab_thresh
    elif artifact == 1:
        labels_artifact = labels_artifact < lab_thresh
    else:
        raise ValueError('value for artifact has to be 0 (bright clusters), 1'
                         ' (detector dropout) or 2 (photobleaching)')
    features = pd.DataFrame(features).astype(np.float64)
    labels_puretrace = pd.DataFrame(labels_puretrace).astype(np.float64)

    lab_out = ps.correct_correlation_by_label(
        ntraces=ntraces,
        traces_of_interest=features,
        labels_of_interest=labels_artifact,
        fwhm=fwhm)
    print('processed correlation with correction by label')

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
    print('processed correlation with correction by prediction')

    corrupt_out = correlate.correlation_of_arbitrary_trace(
        ntraces=ntraces, traces_of_interest=features, fwhm=fwhm, time_step=1.)
    print('processed correlation without correction')

    pure_out = correlate.correlation_of_arbitrary_trace(
        ntraces=ntraces,
        traces_of_interest=labels_puretrace,
        fwhm=fwhm,
        time_step=1.)
    print('processed correlation of pure traces')

    data_diffrates = np.concatenate(
        (lab_out[0], pred_out[0], corrupt_out[0], pure_out[0]), axis=0)
    data_transittimes = np.concatenate(
        (lab_out[1], pred_out[1], corrupt_out[1], pure_out[1]), axis=0)
    data_tracelengths = np.concatenate(
        (lab_out[2], pred_out[2], corrupt_out[2], pure_out[2]), axis=0)
    data_tracesused = np.repeat(
        np.array(('corrupted without correction', 'corrected by prediction',
                  'corrected by labels (control)', 'pure traces (control)')),
        ntraces)
    data_simdiffrates = np.tile(np.repeat(diffrates.values, nsamples), nfiles)
    data_simclusters = np.tile(np.repeat(clusters.values, nsamples), nfiles)
    data_nmols = np.tile(np.repeat(nmols.values, nsamples), nfiles)
    data_out = pd.DataFrame(data=[
        data_simdiffrates, data_simclusters, data_nmols, data_diffrates,
        data_transittimes, data_tracelengths, data_tracesused
    ],
                            index=[
                                'Simulated $D$', 'Simulated $D_{{clust}}$',
                                'nmol', '$D$ in $\\frac{{\mu m^2}}{{s}}$',
                                '$\tau_{{D}}$ in $ms$', 'Trace lengths',
                                'Traces used'
                            ]).T
    if save_as_csv:
        data_out.to_csv(path_or_buf='.', na_rep='NaN', index=False)

    return data_out
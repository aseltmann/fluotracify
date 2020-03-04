# at the moment, nanosimpy is not well maintained. Depending on the
# current state of the project when fluotracify is released, I might
# fork the functions I need from the package
import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("/home/lex/Programme/mynanosimpy/nanosimpy/")
sys.path.append("/home/lex/Programme/mynanosimpy/nanosimpy/nanosimpy")

from fluotracify.applications import correlate
from nanosimpy.simulation_methods import (
    brownian_only_numpy,
    calculate_psf,
    integrate_over_psf,
)


def simulate_trace_array(artifact,
                         nsamples,
                         foci_array,
                         foci_distance,
                         total_sim_time,
                         time_step,
                         nmol,
                         d_mol,
                         width,
                         height,
                         nclust=None,
                         d_clust=None):
    """Simulate a fluorescence trace using the nanosimpy package and
    introduce artifacts

    Parameters
    ----------
    artifact : {0, 1, 2, 3}
        0 = no artifact, 1 = bright clusters, 2 = detector dropout,
        3 = photobleaching
    nsamples : int
        Number of training examples to generate
    foci_array : np.array
        Array of FWHMs in nm of the excitation PSFs used for the foci detection
    foci_distance : int
        Extent of simulated PSF (distance to center of Gaussian)
    total_sim_time : int
        Total simulation time in ms
    time_step : int
        Duration of each time step in ms
    nmol : int
        Number of fastly diffusing molecules
    d_mol : float
        Diffusion rate of fastly diffusing molecules
    width : int
        Width of the simulation in ...
    height : int
        Height of the simulation in ...
    nclust : int, optional
        Number of bright slowly diffusing clusters (only for artifact = 1)
    d_clust float, optional
        Diffusion rate of slowly diffusing clusters (only for artifact = 1)

    Returns
    -------
    out_array : np.array
        np.array with fluorescence traces and labels as columns (trace A,
        label A, trace B, label B, ...)
    """
    def _simulate_bright_clusters(psf,
                                  pos_x,
                                  pos_y,
                                  total_sim_time=total_sim_time,
                                  time_step=time_step,
                                  nclust=nclust,
                                  d_clust=d_clust,
                                  width=width,
                                  height=height):
        cluster_brightness = (np.random.randint(10) + 5) * 1000
        # simulate brownian motion of slow clusters
        track_clust = brownian_only_numpy(
            total_sim_time=total_sim_time,
            time_step=time_step,
            num_of_mol=nclust,
            D=d_clust,
            width=width,
            height=height,
        )
        out_clust = integrate_over_psf(
            psf=copy.deepcopy(psf),
            track_arr=track_clust,
            num_of_mol=nclust,
            psy=pos_y,
            psx=pos_x,
        )
        clust_trace = out_clust["trace"][0]
        return clust_trace, cluster_brightness

    def _simulate_detector_dropout(clean_trace):
        num_of_dropouts = np.random.randint(50)
        detdrop_trace = clean_trace * 100
        # simulate detector dropout
        detdrop_mask = np.zeros(detdrop_trace.shape[0])
        for _ in range(num_of_dropouts):
            length_of_dropout = np.random.randint(25)
            start = int(np.random.random_sample() * detdrop_trace.shape[0])
            end = int(start + length_of_dropout)
            for mid in range(end - start):
                depth_of_dropout = np.random.random_sample()
                detdrop_mask[start + mid:start + mid +
                             1] = (-np.amin(detdrop_trace)) * depth_of_dropout

        return detdrop_trace, detdrop_mask

    def _simulate_photobleaching(track_arr,
                                 psf,
                                 pos_x,
                                 pos_y,
                                 total_sim_time=total_sim_time,
                                 time_step=time_step,
                                 nmol=nmol,
                                 width=width,
                                 height=height):
        d_immobile = 0.001
        exp_scale_rand = np.random.randint(20) * 0.01
        # scales between 0.01 and 0.02 seem to work nicely for a distribution
        # of total_sim_time=20000. if other simulation times are used, this
        # number has to be reevaluated lower scale means faster bleaching,
        # higher scale means slower bleaching
        rng = np.random.default_rng(seed=None)
        bleach_dist = rng.exponential(scale=exp_scale_rand, size=nmol)
        bleach_times = bleach_dist * total_sim_time
        bleach_times = np.clip(bleach_times, a_min=0, a_max=total_sim_time)
        # simulate brownian motion of mobilized and immobilized molecules
        track_arr_immob = brownian_only_numpy(total_sim_time=total_sim_time,
                                              time_step=time_step,
                                              num_of_mol=nmol,
                                              D=d_immobile,
                                              width=width,
                                              height=height)
        track_arr_mob = copy.deepcopy(track_arr)

        # do photobleaching
        for idx, dropout_idx in zip(range(nmol), bleach_times):
            # set fluorescence of each molecule to zero starting from bleach
            # time for each respective molecule
            track_tmp_mob = track_arr_mob[idx]
            track_tmp_immob = track_arr_immob[idx]
            track_tmp_mob[:, int(dropout_idx):] = 0
            track_tmp_immob[:, int(dropout_idx):] = 0
        ibleach_trace = integrate_over_psf(psf=copy.deepcopy(psf),
                                           track_arr=track_arr_immob,
                                           num_of_mol=nmol,
                                           psy=pos_y,
                                           psx=pos_x)
        mbleach_trace = integrate_over_psf(psf=copy.deepcopy(psf),
                                           track_arr=track_arr_mob,
                                           num_of_mol=nmol,
                                           psy=pos_y,
                                           psx=pos_x)
        ibleach_trace = ibleach_trace['trace'][0]
        mbleach_trace = mbleach_trace['trace'][0]
        return ibleach_trace, mbleach_trace, exp_scale_rand

    psf = calculate_psf(foci_array, foci_distance)

    pos_x = width // 2
    pos_y = height // 2

    num_of_steps = int(round(float(total_sim_time) / float(time_step), 0))

    out_array = np.zeros((num_of_steps, nsamples * 2))

    for i in range(nsamples):
        # define scaling summand for data augmentation
        scaling_summand = np.random.randint(10) * 100
        # simulate brownian motion of fast molecules
        track_arr = brownian_only_numpy(
            total_sim_time=total_sim_time,
            time_step=time_step,
            num_of_mol=nmol,
            D=d_mol,
            width=width,
            height=height,
        )
        out_clean = integrate_over_psf(
            psf=copy.deepcopy(psf),
            track_arr=track_arr,
            num_of_mol=nmol,
            psy=pos_y,
            psx=pos_x,
        )
        clean_trace = out_clean["trace"][0]
        if artifact == 0:
            # no artifact
            out_array[:, i * 2] = (
                clean_trace * 100 +
                np.random.random_sample(clean_trace.shape[0]) * 10 +
                scaling_summand)
            out_array[:, i * 2 + 1] = np.full_like(clean_trace.shape,
                                                   np.nan,
                                                   dtype=np.double)
            print('\nTrace {}: Nmol: {} d_mol: {}'.format(i, nmol, d_mol))
        elif artifact == 1:
            # bright clusters / spikes
            clust_trace, cluster_brightness = _simulate_bright_clusters(
                psf=psf, pos_x=pos_x, pos_y=pos_y)
            # combine fast and slow molecules
            out_array[:, i * 2] = (
                clean_trace * 100 +
                np.random.random_sample(clean_trace.shape[0]) * 10 +
                scaling_summand)
            out_array[:, i * 2] += (
                clust_trace * cluster_brightness +
                np.random.random_sample(clust_trace.shape[0]) * 10)
            # save labels
            out_array[:, i * 2 + 1] = clust_trace
            print(
                '\nTrace {}: Nmol: {} d_mol: {} Cluster multiplier: {}'.format(
                    i, nmol, d_mol, cluster_brightness))
        elif artifact == 2:
            # detector dropout
            detdrop_trace, detdrop_mask = _simulate_detector_dropout(
                clean_trace=clean_trace)

            # combine
            out_array[:, i * 2] = detdrop_trace + np.random.random_sample(
                detdrop_trace.shape[0]) * 10 + scaling_summand + detdrop_mask
            # save labels
            out_array[:, i * 2 + 1] = detdrop_mask
            print('\nTrace {}: Nmol: {} d_mol: {} max. drop: {:.2f}'.format(
                i, nmol, d_mol, -np.amin(detdrop_trace)))
        elif artifact == 3:
            # photobleaching
            ibleach_trace, mbleach_trace, exp_scale = _simulate_photobleaching(
                track_arr=track_arr, psf=psf, pos_x=pos_x, pos_y=pos_y)
            # combine all traces for features
            out_array[:, i * 2] = (
                (clean_trace + ibleach_trace + mbleach_trace) * 100 +
                np.random.random_sample(clean_trace.shape[0]) * 10 +
                scaling_summand)
            # combine artefact traces for labels
            out_array[:, i * 2 + 1] = ibleach_trace + mbleach_trace
            print('\nTrace {}: Nmol: {} d_mol: {} scale parameter: {:.2f}'.
                  format(i, nmol, d_mol, exp_scale))
        else:
            raise ValueError('artifact must be 0, 1, 2 or 3')
    return out_array


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
        diffrate_of_interest, thresh_arr, xunit, artifact, xunit_bins, diffrates,
        features, labels):
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

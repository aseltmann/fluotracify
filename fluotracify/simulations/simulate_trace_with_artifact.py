# at the moment, nanosimpy is not well maintained. Depending on the
# current state of the project when fluotracify is released, I might
# fork the functions I need from the package
import copy
import sys

import numpy as np

sys.path.append("/home/lex/Programme/mynanosimpy/nanosimpy/")
sys.path.append("/home/lex/Programme/mynanosimpy/nanosimpy/nanosimpy")

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
        elif artifact == 2:
            # detector dropout
            detdrop_trace, detdrop_mask = _simulate_detector_dropout(
                clean_trace=clean_trace)

            # combine
            out_array[:, i * 2] = detdrop_trace + np.random.random_sample(
                detdrop_trace.shape[0]) * 10 + scaling_summand + detdrop_mask
            # save labels
            out_array[:, i * 2 + 1] = detdrop_mask
            print('\n', nmol, d_mol, -np.amin(detdrop_trace))
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
            print('\n', nmol, d_mol, exp_scale)
        else:
            raise ValueError('artifact must be 0, 1, 2 or 3')
    return out_array

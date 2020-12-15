# at the moment, nanosimpy is not well maintained. Depending on the
# current state of the project when fluotracify is released, I might
# fork the functions I need from the package
import copy
import os
import random
import uuid
from pathlib import Path

import numpy as np

from fluotracify.simulations.simulation_methods import (
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
            print('\nTrace {}: Nmol: {} d_mol: {}'.format(i + 1, nmol, d_mol))
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
                    i + 1, nmol, d_mol, cluster_brightness))
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
                i + 1, nmol, d_mol, -np.amin(detdrop_trace)))
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
                  format(i + 1, nmol, d_mol, exp_scale))
        else:
            raise ValueError('artifact must be 0, 1, 2 or 3')
    return out_array


def savetrace_csv(artifact,
                  path_and_file_name,
                  traces_array,
                  col_per_example,
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
    """save out a series of simulated fluorescence traces and labels indluding
    metadata of the simulations

    Parameters
    ----------
    artifact : {0, 1, 2, 3}
        0 = no artifact, 1 = bright clusters, 2 = detector dropout,
        3 = photobleaching. For 1 and 3, additional metadata is saved out.
        If 1 is chosen, the parameters nclust and d_clust have to be given.
    path_and_file_name : str
        Destination path and file name
    traces_array : np.array
        fluorescence traces and labels as columns (trace A, label A1, label A2,
        trace B, label B1, label B2, ...)
    col_per_example : int
        Number of columns per example, first column being a trace, and then
        one or multiple labels
    ...

    Returns
    -------
    Saves a .csv file
    """
    unique = uuid.uuid4()

    header = ''

    for idx, _ in enumerate(traces_array[0, ::col_per_example], start=1):
        header += 'trace{:0>3},'.format(idx)
        for jdx in range(1, col_per_example):
            header += 'label{:0>3}_{},'.format(idx, jdx)
    # Remove trailing comma
    header = header.strip(',')

    # TODO: include path handling with pathlib to make code work independent
    # of OS
    with open(path_and_file_name, 'w') as my_file:
        my_file.write('unique identifier,{}\n'.format(unique))
        my_file.write('path and file name,{}\n'.format(path_and_file_name))
        my_file.write('FWHMs of excitation PSFs used in nm,'
                      '{}\n'.format(foci_array))
        my_file.write('Extent of simulated PSF (distance to center of '
                      'Gaussian) in nm,{}\n'.format(foci_distance))
        my_file.write('total simulation time in ms,'
                      '{}\n'.format(total_sim_time))
        my_file.write('time step in ms,{}\n'.format(time_step))
        my_file.write('number of fast molecules,{}\n'.format(nmol))
        my_file.write('diffusion rate of molecules in micrometer^2 / s,'
                      '{}\n'.format(d_mol))
        my_file.write('width of the simulation in nm,{}\n'.format(width))
        my_file.write('height of the simulation in nm,{}\n'.format(height))
        if artifact == 1:
            my_file.write('number of slow clusters,{}\n'.format(nclust))
            my_file.write('diffusion rate of clusters in micrometer^2 / s,'
                          '{}\n'.format(d_clust))
        elif artifact == 3:
            my_file.write('number of bleached molecules (50% immobile and '
                          '50% mobile),{}\n'.format(nmol * 2))
        # comments expects a str. Otherwise it printed a '# ' in first column
        # header and importing that to pandas made it an 'object' dtype which
        # uses a lot of memory
        np.savetxt(my_file,
                   traces_array,
                   delimiter=',',
                   header=header,
                   comments='')


def produce_training_data(folder,
                          file_name,
                          number_of_sets,
                          traces_per_set,
                          total_sim_time,
                          artifact,
                          d_mol_arr,
                          label_for=0):
    """Save multiple .csv files containing simulations of Fluorescence
    Correlation Spectroscopy measurements including labelled artifacts.

    Parameters
    ----------
    folder : str
        Folder used for saving
    file_name : str
        Name of files. Extension of style '_setXXX.csv' will be automatically
        created.
    number_of_sets : int
        Number of csv files to generate
    traces_per_set : int
        Traces per file to generate
    total_sim_time : int
        Length of simulated trace in ms
    artifact : {0, 1, 2, 3}
        0 = no artifact, 1 = bright clusters, 2 = detector dropout,
        3 = photobleaching
    d_mol_arr : list or tuple
        Diffusion coefficients in mm^2 / s used for simulation. For each set,
        one will be drawn using random.choice()

    Returns
    -------
    save desired number of csv files (= sets) to desired folder with desired
    number of traces per file

    Raises
    ------
    NotADirectoryError if folder is not a directory

    Notes
    -----
    - The main underlying physical property of the diversity of the
      fluorescence traces is the diffusion constant D of the molecules. This
      means for the correction to work on as much data as possible, the model
      has to be trained on diverse diffusion constants
    - common diffusion constants in fluorescence correlation spectroscopy
      range from 10^{-3} to 10^{2} um^2 / s
    - this is why the dataset's sub-folders are stratified along the given
      diffusion constants in `d_mol_arr`. `number_of_sets` gives the amount of
      .csv files for each sub-folder - this way, an equal distribution of
      diffusion constants is guaranteed
    """
    p = Path(folder)
    if not p.is_dir():
        raise NotADirectoryError('Parameter folder should be a directory.')

    foci_array = np.array([250])
    foci_distance = 4000
    time_step = 1.
    width = 3000.0
    height = 3000.0
    for d_mol in d_mol_arr:
        print('D = {} um^2 / s'.format(d_mol))
        # define the name of the directory to be created
        pdir = p / '{}'.format(d_mol)

        try:
            os.mkdir(pdir)
        except OSError:
            print("Creation of the directory {} failed".format(pdir))
        else:
            print("Successfully created the directory {} ".format(pdir))
        for idx in range(number_of_sets):
            print('Set {} ------------------------'.format(idx + 1))
            nmol = random.choice([500, 1000, 1500, 2000, 2500, 3000, 3500])

            file_name_ext = '_D{}_set{:0>3}.csv'.format(d_mol, idx + 1)
            file = ''.join([file_name, file_name_ext])
            f = Path(file)
            path_and_file_name = pdir / f
            col_per_example = 2

            if artifact == 1:
                # bright clusters
                nclust = 10
                d_clust = [0.005, 0.01, 0.02]
                d_clust = random.choice(d_clust)
                traces = simulate_trace_array(artifact=artifact,
                                              nsamples=traces_per_set,
                                              foci_array=foci_array,
                                              foci_distance=foci_distance,
                                              total_sim_time=total_sim_time,
                                              time_step=time_step,
                                              nmol=nmol,
                                              d_mol=d_mol,
                                              width=width,
                                              height=height,
                                              nclust=nclust,
                                              d_clust=d_clust,
                                              label_for=label_for)
                savetrace_csv(artifact=artifact,
                              path_and_file_name=path_and_file_name,
                              traces_array=traces,
                              col_per_example=col_per_example,
                              foci_array=foci_array,
                              foci_distance=foci_distance,
                              total_sim_time=total_sim_time,
                              time_step=time_step,
                              nmol=nmol,
                              d_mol=d_mol,
                              width=width,
                              height=height,
                              nclust=nclust,
                              d_clust=d_clust)

            elif artifact == 0 or 2 or 3:
                traces = simulate_trace_array(artifact=artifact,
                                              nsamples=traces_per_set,
                                              foci_array=foci_array,
                                              foci_distance=foci_distance,
                                              total_sim_time=total_sim_time,
                                              time_step=time_step,
                                              nmol=nmol,
                                              d_mol=d_mol,
                                              width=width,
                                              height=height,
                                              label_for=label_for)
                savetrace_csv(artifact=artifact,
                              path_and_file_name=path_and_file_name,
                              traces_array=traces,
                              col_per_example=col_per_example,
                              foci_array=foci_array,
                              foci_distance=foci_distance,
                              total_sim_time=total_sim_time,
                              time_step=time_step,
                              nmol=nmol,
                              d_mol=d_mol,
                              width=width,
                              height=height)
            else:
                raise ValueError('artifact must be 0, 1, 2 or 3')

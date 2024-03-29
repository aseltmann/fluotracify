# This module copies functions of Dominic Waithe's nanosimpy module
# https://github.com/dwaithe/nanosimpy
import copy
import sys

import numpy as np
import scipy.stats as sst


def brownian_only_numpy(total_sim_time, time_step, num_of_mol, D, width,
                        height):
    """Simulate brownian motion / random walk of a given number of molecules

    Parameters
    ----------
    total_simulation_time : int
        Total simulation time in ms.
    time_step : int
        The duration of each time step ms.
    num_mol : int
        The number of molecules in the simulation.
    D : float
        The diffusion rate in {mu m^2}{s}
    width : int
        The width of the simulation area
    height : int
        The height of the simulation area

    Returns
    -------
    track_arr : dict of list of numpy arrays
        A dictionary where each track number (e.g. track_arr[0]) contains the
        track data with y-coordinates [0,:] and x-coordinates [1,:]
    """

    # Number of steps.
    num_of_steps = int(round(float(total_sim_time) / float(time_step), 0))

    print('num_of_steps', num_of_steps)
    # Calculates length scales
    scale_in = np.sqrt(2.0 * (float(D) * 1e3) * float(time_step))

    # Randomly generates start locations
    start_coord_x = (np.random.uniform(0.0, 1.0, num_of_mol)) * width
    start_coord_y = (np.random.uniform(0.0, 1.0, num_of_mol)) * height

    track_arr = {}
    # This can be done as one big matrix, but can crash system if large so
    # I break it up by molecule.
    for b in range(0, num_of_mol):
        per = int((float(b) / float(num_of_mol)) * 100)
        sys.stdout.write("\rProcessing tracks: [{:20}] {}% complete".format(
            '=' * int(per / 5), per))
        sys.stdout.flush()
        track = np.zeros((2, num_of_steps))
        track[0, 0] = start_coord_y[b]
        track[1, 0] = start_coord_x[b]
        rand_in = sst.norm.rvs(size=[2, num_of_steps]) * scale_in
        track[:, 1:] += rand_in[:, 1:]
        track = np.cumsum(track, 1)
        out = track
        mod = np.zeros((out.shape))
        mod[0, :] = np.floor(track[0, :].astype(np.float64) / height)
        mod[1, :] = np.floor(track[1, :].astype(np.float64) / width)
        track_arr[b] = np.array(out -
                                ([mod[0, :] * height, mod[1, :] * width]))

        # We go through and make sure our particles wrap around.
        # for b in range(0,num_of_mol):
        # print 'wrapping tracks: ', (float(b) / float(num_of_mol)) * 100, '%'
        # bool_to_adapt = (track_arr[b][0, :] -
        #                  offset)**2 + (track_arr[b][1, :] - offset)**2 >= R2
        # while np.sum(bool_to_adapt) > 0:
        #     ind = np.argmax(bool_to_adapt > 0)
        #     phi = np.arctan2((track_arr[b][0, ind] - offset),
        #                      (track_arr[b][1, ind] - offset))
        #     track_arr[b][1, ind:] = np.round(
        #         ((track_arr[b][1, ind:] - offset) -
        #          (2.0 * (R - 2) * np.cos(phi))) + offset, 0).astype(np.int32)
        #     track_arr[b][0, ind:] = np.round(
        #         ((track_arr[b][0, ind:] - offset) -
        #          (2.0 * (R - 2) * np.sin(phi))) + offset, 0).astype(np.int32)
        #     bool_to_adapt = (track_arr[b][0, :] - offset)**2 + (
        #         track_arr[b][1, :] - offset)**2 >= R2
    return track_arr


def calculate_psf(fwhms, distance):
    """Calculates Gaussian of particular FWHM

    Parameters
    ----------
    fwhms : list of ints
        List of Full Width Half Maximum (FWHMs) of Point Spread Functions
        (PSFs) of excitation laser to simulate particles under a fluorescence
        microscope
    distance : int
        Length of simulated PSF (from maximum radially outwards).

    Returns
    -------
    psf : dict
        'FWHMs' : list of ints
        'pixel_size' : 1.0
        'ri' : dict of lists of ints
            Length values of simulated PSF
        'number_FWHMs' : int
        'V' : dict of lists of floats
            Intensity values of simulated PSF at length 'ri'

    Notes
    -----
    # FWHM to sigma conversion
    FWHM = 2*np.sqrt(2*np.log(2))*sigma
    # Sigma from FWHM
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))
    # is conventional Gaussian
    G = np.exp(-x**2/(2*sigma**2))
    # substitute FWHM for sigma
    G = np.exp(-x**2/(2*(FWHM/(2*np.sqrt(2*np.log(2))))**2))
    # open the brackets and square contents
    G = np.exp(-x**2/(2*(FWHM**2/(4*2*np.log(2)))))
    # decompose fraction
    G = np.exp((-x**2/(2*(FWHM**2)/8.))*(np.log(2)))
    # power law decomposition
    G = np.exp((np.log(2.)))**(-x**2/((FWHM**2)/4.0))
    # e^(ln2) = 2 indentity.
    G = 2.**(-x**2/(FWHM/2.0)**2)
    """

    psf = {}
    psf['FWHMs'] = fwhms
    psf['pixel_size'] = 1.0
    psf['ri'] = np.meshgrid(np.arange(0, distance, psf['pixel_size']))[0]
    psf['number_FWHMs'] = psf['FWHMs'].__len__()
    psf['V'] = {}
    for ki in range(0, psf['number_FWHMs']):
        psf['V'][ki] = 2.0**(-psf['ri']**2 / (psf['FWHMs'][ki] / 2.0)**2)
    return psf


def integrate_over_psf(psf, track_arr, num_of_mol, psy, psx):
    """Pass an array of Brownian motion tracks through the PSF function

    Parameters
    ----------
    psf : dict
        'FWHMs' : list of ints
        'pixel_size' : 1.0
        'ri' : dict of lists of ints
            Length values of simulated PSF
        'number_FWHMs' : int
        'V' : dict of lists of float
            Intensity values of simulated PSF at length 'ri'
    track_arr : dict of lists of numpy arrays
        A dictionary where each track number (e.g. track_arr[0]) contains the
        track data with y-coordinates [0,:] and x-coordinates [1,:]
    num_of_mol : int
        The number of molecules in the simulation.
    psy, psx : int
        The location of the focal volume in the simulated area.

    Returns
    -------
    psf : dict
        'FWHMs' : list of ints
        'pixel_size' : 1.0
        'ri' : dict of lists of ints
            Length values of simulated PSF
        'number_FWHMs' : int
        'V' : dict of lists of float
            Intensity values of simulated PSF at length 'ri'
        'trace' : dict of lists of floats
            Gives non-physical relative intensity values (= emitted photons)
            for each molecule during its track through the simulated area.

    Notes
    -----
    - basic algorithm for each molecule:
      1. Get Euclidian distances from each position of the molecule to the
         location of the focal volume
      2. Choose relative intensity value for each position of the molecule
         according to the simulated psf from `calculate_psf`
    """
    psf['trace'] = {}
    sys.stdout.write('\n')
    for ki in range(0, psf['number_FWHMs']):
        sys.stdout.write("\rProcessing FWHM {}, ".format(psf['FWHMs'][ki]))
        sys.stdout.flush()
        trace = 0
        for b in range(0, num_of_mol):
            b_dist = np.round(
                np.sqrt((track_arr[b][1] - psx)**2 +
                        (track_arr[b][0] - psy)**2), 0).astype(np.int32)
            b_trace = psf['V'][ki][b_dist]
            trace += b_trace
        psf['trace'][ki] = copy.deepcopy(trace)
    return psf

"""This module contains functions to perform autocorrelation analysis on one-
dimensional fluorescence traces. Largely based on Dominic Waithe's FOCUSpoint
"""

import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmfit import Parameters, fit_report, minimize
from multipletau import autocorrelate
from pathlib import Path

from fluotracify.applications import (equations_to_fit as eq,
                                      correlate_cython as ccy,
                                      corr_fit_object as cfo)

logging.basicConfig(format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def correlate_timetrace_and_save(df, out_path, out_txt):
    """Correlate FCS timetraces ordered columnwise and save each correlation as a
    single .csv file which can then be fitted by fitting software such as
    D. Waithe's FOCUSpoint or FCSfitJS (https://dwaithe.github.io/FCSfitJS/)

    Paramters
    ---------
    df : pandas DataFrame
        FCS traces ordered columnwise
    out_path : str or Path
        Path where .csv files are saved
    out_txt : str
        Used for custom filename, the final file is saved as
        <out_path>/<datetime>_multipletau_<out_txt>_<df-column-name>_correlation.csv

    Returns
    -------
    Nothing, but saves files as described

    Raises
    ------
    ValueError
        if df is not a pd.DataFrame
    ValueError
        if out_path is not convertible to a pathlib.Path
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df has to be a pandas DataFrame')
    try:
        out_path = Path(out_path)
    except TypeError as exc:
        raise ValueError(
            'out_path has to be convertible to a pathlib.Path') from exc

    for idx, col in enumerate(df.columns):
        trace = df.iloc[:, idx].dropna()
        corr_fn = autocorrelate(
            a=trace,
            m=16,
            deltat=1,
            normalize=True)
        # there was a bug with the 0th step when loading the file in FCSfitJS
        # so we just skip it as a workaround
        autotime = corr_fn[1:, 0]
        autonorm = corr_fn[1:, 1]
        out_file_txt = f'{out_txt}_{col.replace(".", "dot")}'
        out_file = Path(f'{datetime.date.today()}_multipletau_'
                        f'{out_file_txt}_{idx:04}_correlation.csv')
        out_file = out_path / out_file

        with open(out_file, 'w', encoding='utf-8') as out:
            out.write('version,3.0\n')
            out.write('numOfCh,1\n')
            out.write('type,point\n')
            out.write(f'parent_name,{out_txt}\n')
            out.write('ch_type,1_1\n')
            out.write('kcount,1\n')  # arbitrary value
            out.write('numberNandB,1\n')  # arbitrary value
            out.write('brightnessNandB,1\n')  # arbitrary value
            out.write('carpet pos,0\n')
            out.write('pc,0\n')
            out.write('Time (ms),CH1 Auto-Correlation\n')
            for i in range(autotime.shape[0]):
                out.write(f'{autotime[i]},{autonorm[i]}\n')
            out.write('end\n')
        log.debug('correlate_timetrace_and_save: Finished saving of file %s',
                  out_file)


def correlate_ptu_and_save(in_path, out_path):
    """Helper function for corr_fit_object.py. Autocorrelate FCS TCSPC data
    using the tttr2xfcs algorithm and save each correlation as a single .csv
    file which can then be fitted by fitting software such as D. Waithe's
    FOCUSpoint or FCSfitJS (https://dwaithe.github.io/FCSfitJS/)

    Parameters
    ----------
    in_path : str or pathlib.Path
        Folder containing .ptu files
    out_path : str or pathlib.Path
        Folder for output files

    Returns
    -------
    Nothing, but saves a single .csv file for each input file as described.

    Raises
    ------
    ValueError
        if in_path or out_path are not convertible to a pathlib.Path

    Notes
    -----
    - To tune tttr2xfcs correlation parameters, instantiate an own
    ParameterClass()
    - For tuning different correlation algorithms, or fluorescence trace
    artifact prediction and correction, directly use the functions provided
    in corr_fit_object.py.
    """
    try:
        in_path = Path(in_path)
        out_path = Path(out_path)
    except TypeError as exc:
        raise ValueError('in_path and out_path have to be convertible to a'
                         ' pathlib.Path') from exc

    files = [in_path / f for f in os.listdir(in_path) if f.endswith('.ptu')]

    class ParameterClass():
        """Stores parameters for correlation """
        def __init__(self):
            # Where the data is stored.
            self.data = []
            self.objectRef = []
            self.subObjectRef = []
            self.colors = ['blue', 'green', 'red', 'cyan', 'magenta',
                           'yellow', 'black']
            self.numOfLoaded = 0
            # very fast from Ncasc ~ 14 onwards
            self.NcascStart = 0
            self.NcascEnd = 30  # 25
            self.Nsub = 6  # 6
            self.photonLifetimeBin = 10  # used for photon decay
            self.photonCountBin = 1  # used for time series

    par_obj = ParameterClass()

    for _, myfile in enumerate(files):
        ptufile = cfo.PicoObject(myfile, par_obj)
        # init calls the following methods:
        # self.importData() - imports .ptu data to memory
        # self.prepareChannels() - checks how many channels are used
        # self.getPhotonDecay() - gets photon decay curve
        # self.getTimeSeries() - gets time series
        # self.getPhotonCountingStats() - gets photon counting statistics
        for key in list(ptufile.trueTimeArr.keys()):
            ptufile.get_autocorrelation(method='tttr2xfcs', name=key)

        for m in ['multipletau', 'tttr2xfcs', 'tttr2xfcs_with_weights']:
            if m in list(ptufile.autoNorm.keys()):
                for key, _ in list(ptufile.autoNorm[m].items()):
                    ptufile.save_autocorrelation(name=key, method=m,
                                                 output_path=out_path)


def save_correlations(df, out_path, out_txt):
    """Take FCS autocorrelations ordered columnwise with the index being the
    correlation times and save each correlation as a single .csv file which
    can then be fitted by fitting software such as D. Waithe's FOCUSpoint or
    FCSfitJS (https://dwaithe.github.io/FCSfitJS/)

    Paramters
    ---------
    df : pandas DataFrame
        FCS correlations ordered columnwise
    out_path : str or Path
        Path where .csv files are saved
    out_txt : str
        Used for custom filename, the final file is saved as
        <out_path>/<datetime>_multipletau_<out_txt>_<df-column-name>_correlation.csv

    Returns
    -------
    Nothing, but saves files as described

    Raises
    ------
    ValueError
        if df is not a pd.DataFrame
    ValueError
        if out_path is not convertible to a pathlib.Path
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df has to be a pandas DataFrame')
    try:
        out_path = Path(out_path)
    except TypeError as exc:
        raise ValueError(
            'out_path has to be convertible to a pathlib.Path') from exc

    for idx, col in enumerate(df.columns):
        corr = df.iloc[:, idx].dropna()
        autotime = corr.index
        autonorm = corr.values
        out_file_txt = f'{out_txt}_{col.replace(".", "dot")}'
        out_file = Path(f'{datetime.date.today()}_multipletau_'
                        f'{out_file_txt}_{idx:04}_correlation.csv')
        out_file = out_path / out_file

        with open(out_file, 'w', encoding='utf-8') as out:
            out.write('version,3.0\n')
            out.write('numOfCh,1\n')
            out.write('type,point\n')
            out.write(f'parent_name,{out_txt}\n')
            out.write('ch_type,1_1\n')
            out.write('kcount,1\n')  # arbitrary value
            out.write('numberNandB,1\n')  # arbitrary value
            out.write('brightnessNandB,1\n')  # arbitrary value
            out.write('carpet pos,0\n')
            out.write('pc,0\n')
            out.write('Time (ms),CH1 Auto-Correlation\n')
            for i in range(autotime.shape[0]):
                out.write(f'{autotime[i]},{autonorm[i]}\n')
            out.write('end\n')
        log.debug('save_correlations: Finished saving of file %s',
                  out_file)


                    
def correlate_and_fit(trace, fwhm, diffrate=None, time_step=1., verbose=True):
    """autocorrelate an FCS trace and return the calculated diffusion rate and
    transit_time. If verbose=True, plot the correlation curve and fit, and
    print out the derived values for transit time and diffusion constant.

    Parameters
    ----------
    trace : numpy array, pandas DataFrame, dtype has to be float64
        A 1D time trace
    fwhm : float
        The full width half maximum of the excitation beam in nm. Used for
        Calculation of the diffusion coefficient.
    diffrate : float
        The simulated diffusion rate. Insert 'None', if no simulation was done.
    time_step : int, float
        The time step size of the fluorescence trace in ms. Is used for
        positioning the x-axis of the  correlation plot and does not influence
        the correlation itself
    verbose : bool
        If True, does a plot of correlation and fit and prints out the fit
        report, transit time from fit,  diffusion rate calculated from transit
        time(, and simulated diffusion rate, if applicable)

    Returns
    -------
    diffrate_calc : float
        Diffusion rate in um^2 / s calculated from transit time
    transit_time : flaot
        Transit time in ms calculated from fit
    out : numpy arrays
        - out[0] gives tau in ms
        - out[1] gives G(tau)
        - out[2] gives correlation (x, y), fit, and residuals
    """
    # otherwise alpha1-variation does not seem to work
    assert trace.dtype == np.float64, 'Error: The datatype should be float64'
    # Paremeters for fitting.
    param = Parameters()
    param.add('offset', value=0.01, min=-0.5, max=1.5, vary=True)
    param.add('GN0', value=1.000, min=-0.0001, max=3000.0, vary=True)
    param.add('A1', value=1.000, min=0.0001, max=1.0000, vary=False)
    param.add('txy1', value=0.10, min=0.001, max=2000.0, vary=True)
    param.add('alpha1', value=0.75, min=0.600, max=2.0, vary=True)
    options = {'Dimen': 1, 'Diff_eq': 1, 'Triplet_eq': 1, 'Diff_species': 1}
    # Correlation
    try:
        out = autocorrelate(trace, m=16, normalize=True, deltat=time_step)
    except (ValueError, IndexError):
        # if correlation fails, e.g. because len(trace) < 2*m (ValueError)
        # or because a trace of length 0 is given (IndexError)
        return np.nan, np.nan, np.nan
    # Fit
    res = minimize(eq.residual, param, args=(out[:, 0], out[:, 1], options))
    fit = eq.equation_(res.params, out[:, 0], options)
    residual_var = res.residual
    output = fit_report(res.params)
    transit_time = res.params['txy1'].value
    diffrate_calc = (float(fwhm) / 1000)**2 / (8 * np.log(2.0) * transit_time /
                                               1000)
    if verbose:
        plt.semilogx(out[:, 0], out[:, 1], 'g', label='correlation')
        plt.semilogx(out[:, 0], fit, 'r', label='fit')
        plt.xlim(left=time_step)
        plt.legend()
        print('fit report: {}\n'.format(output))
        print('transit time derived from fit: {}'.format(transit_time))
        print('diffusion rate calculated from transit time: {}'.format(
            diffrate_calc))
        if diffrate is not None:
            print('simulated diffusion rate: {}'.format(diffrate))
        print('\n')
    return diffrate_calc, transit_time, (out[:, 0], out[:,
                                                        1], fit, residual_var)


def correlation_of_arbitrary_trace(ntraces,
                                   traces_of_interest,
                                   fwhm,
                                   time_step,
                                   length_delimiter=None):
    """Takes pandas DataFrame of fluorescence traces ordered columnwise and
    performs an autocorrelation analysis on each trace
    Parameters
    ----------
    ntraces : int
        Number of ntraces from given DataFrames to choose for correlation
    traces_of_interest : Pandas DataFrame
        Contains the traces columnwise
    fwhm : float
        The full width half maximum of the excitation beam in nm. Used for
        Calculation of the diffusion coefficient.
    """
    if ntraces is None:
        ntraces = len(traces_of_interest.columns)

    diffrates_arb = []
    transit_times_arb = []
    tracelen_arb = []

    for ntraces_index in range(ntraces):
        trace_arb = traces_of_interest.iloc[:length_delimiter, ntraces_index]

        diff_arb, trans_arb, _ = correlate_and_fit(trace=trace_arb,
                                                   fwhm=fwhm,
                                                   diffrate=None,
                                                   time_step=time_step,
                                                   verbose=False)
        diffrates_arb.append(diff_arb)
        transit_times_arb.append(trans_arb)
        tracelen_arb.append(len(trace_arb))
    return diffrates_arb, transit_times_arb, tracelen_arb


"""From here on: FCS Bulk Correlation Software

    Copyright (C) 2015  Dominic Waithe

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""


def tttr2xfcs(y, num, NcascStart, NcascEnd, Nsub):
    """autocorr, autotime = tttr2xfcs(y,num,10,20)
    Translation into python of:
    Fast calculation of fluorescence correlation data with asynchronous
    time-correlated single-photon counting.
    Michael Wahl, Ingo Gregor, Matthias Patting, Jorg Enderlein

    This algorithm is most appropriate to use with time-tag data, whereby the
    photons are recorded individually as they arrive.
    The arrival times are correlated rather than binned intensities (though
    some binning is performed at later cycles).

    for intensity data which is recorded at regular intervals use a
    high-peforming correlation such as multipletau:
    (https://github.com/FCS-analysis/multipletau_)

    or a basic numpy version which can be found amongst others here:
    https://github.com/dwaithe/generalMacros/blob/master/diffusion%20simulations%20/Correlate%20Comparison.ipynb

    --- inputs ---
    y:
        An array of the photon arrival times for both channels.
    num:
        This a 2D boolean array of the photons in each channel. A '1'
        represents a photon at each corresponding time (row) in y in each
        channel (col)
    Ncasc:
        in general refers to the number of logarithmic ranges to calculate the
        correlation function.
    NcascStart:
        This is a feature I added whereby you can start the correlation at a
        later stage.
    NcasEnd:
        This is the last level of correlation
    Nsub:
        This is the number of sub-levels correlated at each casc level. You
        can think of this as the level of detail. The higher the value the more
        noisy

    --- outputs ---
    auto:
        This is the un-normalised  auto and cross-correlation function output.
            auto[:,0,0] = autocorrelation channel 0
            auto[:,1,1] = autocorrelation channel 1
            auto[:,1,0] = crosscorrelation channel 10
            auto[:,0,1] = crosscorrelation channel 01
    autotime:
        This is the associated tau time range.
    """
    dt = np.max(y) - np.min(y)
    y = np.round(y[:], 0)
    # numshape = num.shape[0]
    autotime = np.zeros(((NcascEnd + 1) * (Nsub + 1), 1))
    auto = np.zeros(((NcascEnd + 1) * (Nsub + 1), num.shape[1],
                     num.shape[1])).astype(np.float64)
    shift = float(0)
    delta = float(1)

    for j in range(0, NcascEnd):
        # Finds the unique photon times and their indices. The division of 'y'
        # by '2' each cycle makes this more likely.
        y, k1 = np.unique(y, 1)
        k1shape = k1.shape[0]

        # Sums up the photon times in each bin.
        cs = np.cumsum(num, 0).T

        # Prepares difference array so starts with zero.
        diffArr1 = np.zeros((k1shape + 1))
        diffArr2 = np.zeros((k1shape + 1))

        # Takes the cumulative sum of the unique photon arrivals
        diffArr1[1:] = cs[0, k1].reshape(-1)
        diffArr2[1:] = cs[1, k1].reshape(-1)

        # del k1
        # del cs
        num = np.zeros((y.shape[0], 2))

        # Finds the total photons in each bin. and represents as count.
        # This is achieved because we have the indices of each unique time
        # photon and cumulative total at each point.
        num[:, 0] = np.diff(diffArr1)
        num[:, 1] = np.diff(diffArr2)
        # diffArr1 = []
        # diffArr2 = []
        for k in range(0, Nsub):
            shift = shift + delta
            lag = np.round(shift / delta, 0)
            # Allows the script to be sped up.
            if j >= NcascStart:
                # Old method
                # i1= np.in1d(y,y+lag,assume_unique=True)
                # i2= np.in1d(y+lag,y,assume_unique=True)
                # New method, cython
                i1, i2 = ccy.dividAndConquer(y, y + lag, y.shape[0] + 1)
                # If the weights (num) are one as in the first Ncasc round,
                # then the correlation is equal to np.sum(i1)
                i1 = np.where(i1.astype(np.bool))[0]
                i2 = np.where(i2.astype(np.bool))[0]
                # Now we want to weight each photon correctly.
                # Faster dot product method, faster than converting to matrix.
                if i1.size and i2.size:
                    jin = np.dot((num[i1, :]).T, num[i2, :]) / delta
                    auto[(k + (j) * Nsub), :, :] = jin
                # log.debug(f'tttr2xfcs: finished Nsub {k} of Ncasc {j}')
            autotime[k + (j) * Nsub] = shift

        # Equivalent to matlab round when numbers are %.5
        y = np.ceil(np.array(0.5 * y))
        delta = 2 * delta
        log.debug(f'tttr2xfcs: finished Ncasc {j}')

    for j in range(0, auto.shape[0]):
        auto[j, :, :] = auto[j, :, :] * dt / (dt - autotime[j])
    # FIXME: why divided by 1000000, is this related to self.timeSeriesDividend?
    autotime = autotime / 1000000

    # Removes the trailing zeros.
    idauto = np.where(autotime != 0)[0]
    autotime = autotime[idauto]
    auto = auto[idauto, :, :]
    return auto, autotime


def delayTime2bin(dTimeArr, chanArr, chanNum, winInt):
    decayTime = np.array(dTimeArr)
    # This is the point and which each channel is identified.
    decayTimeCh = decayTime[chanArr == chanNum]
    # Find the first and last entry
    firstDecayTime = 0  # np.min(decayTimeCh).astype(np.int32)
    tempLastDecayTime = np.max(decayTimeCh).astype(np.int32)
    # We floor this as the last bin is always incomplete and so we discard
    # photons.
    numBins = np.floor((tempLastDecayTime - firstDecayTime) / winInt)
    lastDecayTime = numBins * winInt
    bins = np.linspace(firstDecayTime, lastDecayTime, int(numBins) + 1)
    photonsInBin, jnk = np.histogram(decayTimeCh, bins)
    # bins are valued as half their span.
    decayScale = bins[:-1] + (winInt / 2)
    # decayScale =  np.arange(0,decayTimeCh.shape[0])
    log.debug(f'lastDecayTime={lastDecayTime}, numBins={numBins}, bins[:5]={bins[:5]}')
    return list(photonsInBin), list(decayScale)


def dividAndConquer(arr1b, arr2b, arrLength):
    """divide and conquer fast intersection algorithm. Waithe D 2014

    Notes
    -----
    This implementation is very slow. Use correlate_cython.dividAndConquer
    for a fast cython implementation"""
    arr1bool = np.zeros((arrLength - 1))
    arr2bool = np.zeros((arrLength - 1))
    arr1 = arr1b
    arr2 = arr2b
    arrLen = arrLength
    i = 0
    j = 0
    while (i < arrLen - 1 and j < arrLen - 1):
        if (arr1[i] < arr2[j]):
            i += 1
        elif (arr2[j] < arr1[i]):
            j += 1
        elif (arr1[i] == arr2[j]):
            arr1bool[i] = 1
            arr2bool[j] = 1
            i += 1
    return arr1bool, arr2bool


def photonCountingStats(timeSeries, timeSeriesScale):
    """returns counting statistics

    Notes
    -----
    - code is adopted from Dominic Waithe's Focuspoint package:
    https://github.com/dwaithe/FCS_point_correlator/blob/master/focuspoint/correlation_objects.py"""
    unit = timeSeriesScale[-1] / timeSeriesScale.__len__()
    # Converts to counts per
    kcount_CH = np.average(timeSeries)
    # This is the unnormalised intensity count for int_time duration (the first
    # moment)
    raw_count = np.average(timeSeries)
    var_count = np.var(timeSeries)

    brightnessNandBCH = (((var_count - raw_count) / (raw_count)) /
                         (float(unit)))
    if (var_count - raw_count) == 0:
        numberNandBCH = 0
    else:
        numberNandBCH = (raw_count**2 / (var_count - raw_count))
    return kcount_CH, brightnessNandBCH, numberNandBCH


def calc_coincidence_value(timeSeries1, timeSeries2):
    N1 = np.bincount((np.array(timeSeries1)).astype(np.int64))
    N2 = np.bincount((np.array(timeSeries2)).astype(np.int64))

    n = max(N1.shape[0], N2.shape[0])
    NN1 = np.zeros(n)
    NN2 = np.zeros(n)
    NN1[:N1.shape[0]] = N1
    NN2[:N2.shape[0]] = N2
    N1 = NN1
    N2 = NN2

    CV = (np.sum(N1 * N2) / (np.sum(N1) * np.sum(N2))) * n
    return CV

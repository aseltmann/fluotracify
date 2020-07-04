import sys

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, fit_report, minimize
from multipletau import autocorrelate

# from nanosimpy
if True:  # isort workaround
    sys.path.append('../../../src/')
    import nanosimpy.nanosimpy.equations_to_fit as eq


def correlate(trace, fwhm, diffrate, time_step=1., verbose=True):
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
        - out[2] gives residuals from fit
    """
    # otherwise alpha1-variation does not seem to work
    assert trace.dtype == np.float64, 'Error: The datatype should be float64'
    # Paremeters for fitting.
    param = Parameters()
    param.add('offset', value=0.01, min=-0.5, max=1.5, vary=True)
    param.add('GN0', value=1.000, min=-0.0001, max=3000.0, vary=True)
    param.add('offset', value=0.01, min=-0.5, max=1.5, vary=True)
    param.add('A1', value=1.000, min=0.0001, max=1.0000, vary=False)
    param.add('txy1', value=0.10, min=0.001, max=2000.0, vary=True)
    param.add('alpha1', value=0.75, min=0.600, max=2.0, vary=True)
    options = {'Dimen': 1, 'Diff_eq': 1, 'Triplet_eq': 1, 'Diff_species': 1}
    # Correlation
    out = autocorrelate(trace, normalize=True, deltat=time_step)
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
    return diffrate_calc, transit_time, (out[:, 0], out[:, 1],
                                         fit, residual_var)


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
    diffrates_arb = []
    transit_times_arb = []
    tracelen_arb = []

    for ntraces_index in range(ntraces):
        trace_arb = traces_of_interest.iloc[:length_delimiter, ntraces_index]

        diff_arb, trans_arb, _ = correlate(trace=trace_arb,
                                           fwhm=fwhm,
                                           diffrate=None,
                                           time_step=time_step,
                                           verbose=False)
        diffrates_arb.append(diff_arb)
        transit_times_arb.append(trans_arb)
        tracelen_arb.append(len(trace_arb))
    return diffrates_arb, transit_times_arb, tracelen_arb

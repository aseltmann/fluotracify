import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fluotracify.applications.correction import correct_correlation_by_prediction
from fluotracify.applications.correlate import correlation_of_arbitrary_trace
from fluotracify.simulations.plot_simulations import correct_correlation_by_label


def plot_distribution_of_correlations_by_label_thresholds(
        diffrate_of_interest, thresh_arr, xunit, artifact, bins, diffrates,
        features, labels):
    """plot the distribution of correlations after correcting fluorescence
    traces corrupted by photobleaching applying different thresholds

    Parameters
    ----------
    diffrate_of_interest : float
        Check simulated diffusion rates and distribution of diffrates per set
        of simulated traces
    thresh_arr : array / list
        Contains the different thresholds you want to apply on the label data
    xunit : {0, 1}
        0: plot histogram of diffusion rates in um / s,
        1: plot histogram of transit times in ms
    artifact : {0, 1, 2}
        0: bright clusters / bursts
        1: detector dropout,
        2: photobleaching
    bins : int or sequence of scalars or str, optional
        Binning of the histogram of xunit, see
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
    """

    fwhm = 250
    # Calculate expected transit time for title of plot
    transit_time_expected = ((float(fwhm) / 1000)**2 *
                             1000) / (diffrate_of_interest * 8 * np.log(2.0))

    # get pandas Series of diffrates of interest
    diff = diffrates.where(diffrates == diffrate_of_interest).dropna()

    traces_of_interest = pd.DataFrame()
    for diff_idx in diff.index:
        # number of read in files is 100, every 100th trace belongs to the
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

        if artifact == 0:  # bright clusters
            labels_of_interest = labels_of_interest > thresh
        elif artifact == 1:  # detector dropout
            labels_of_interest = labels_of_interest < thresh
        elif artifact == 2:  # photobleaching
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
                 bins=bins,
                 alpha=0.5,
                 label='threshold: {}'.format(thresh))
        ax2 = fig.add_subplot(gs[:, -1])
        ax2.hist(out[2],
                 bins=np.arange(0, 20001, 1000),
                 alpha=0.5,
                 label='threshold: {}'.format(thresh))
    if xunit == 0:
        ax1.set_title('Diffusion rates by correlation with expected value '
                      '{:.2f}$um^2 / s$'.format(diffrate_of_interest),
                      fontsize=20)
        ax1.set_xlabel('Diffusion coefficients in $um^2 / s$', size=16)
        ax1.axvline(x=diffrate_of_interest,
                    color='r',
                    label='simulated diffusion rate')
    elif xunit == 1:
        ax1.set_title('Transit times by correlation with expected value '
                      '{:.2f}$ms$'.format(transit_time_expected),
                      fontsize=20)
        ax1.set_xlabel('Transit times in $ms$', size=16)
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


def plot_distribution_of_correlations_corrected_by_prediction(
        diffrate_of_interest,
        model,
        lab_thresh,
        pred_thresh,
        xunit,
        artifact,
        xunit_bins,
        diffrates,
        features,
        labels,
        plotperfect=True,
        number_of_traces=None):
    """plot the distribution of correlations after correcting fluorescence
    traces corrupted by the given artifact applying different thresholds

    caveat: The parameters fwhm, win_len and zoomvector are currently fixed
    to reduce the number of parameters of this function, but can easily be
    introduced as parameters.

    Parameters
    ----------
    diffrate_of_interest : float
        Check simulated diffusion rates and distribution of diffrates per
        set of simulated traces
    model
        Model used to correct the corrupted traces
    lab_thresh : float
        Threshold you want to apply on the label data
    pred_thresh : float between 0 and 1
        Threshold you want to apply on the predictions
    xunit : {0, 1}
        0: plot histogram of diffusion rates in um / s
        1: plot histogram of transit times in ms
    artifact : {0, 1, 2}
        0: bright clusters / bursts
        1: detector dropout
        2: photobleaching
    xunit_bins : int or sequence of scalars or str, optional
        Binning of the histogram of xunit, see
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
    plotperfect : bool, optional
        If True, the 'perfect' traces (which have no time step with corruption
        above lab_thresh) will be plotted as well. If this parameter is chosen,
        number_of_traces should be 'None'
    number_of_traces : int, None, optional
        Number of traces which shall be chosen. If 'None', the number of
        traces is determined of the number of perfect traces and they will be
        plotted as well.

    Returns
    -------
    lab_out : tuple
        Contains diffusion rates, transit times and trace lengths corrected by
        label
    pred_out : tuple
        Contains diffusion rates, transit times and trace lengths corrected by
        prediciton
    perfect_out : tuple, optional
        Contains diffusion rates, transit times and trace lengths of 'perfect
        traces' which have no time step with corruption above lab_thresh. Only
        returned, if plotperfect=True

    Notes
    -----
    To be implemented
    lognormfit : bool, optional - ,
        lognormfit=False
        If True, use scipy.stats.lognorm.fit routine to display a probability
        density function of a lognorm distribution fitted on the transit time
        histograms and display the mean of the fit as a vertical line.
    """

    # PART 1: Calculation
    # fix constants (could be introduced as variables)
    fwhm = 250
    win_len = 128
    zoomvector = (5, 21)
    # we assume 100 traces per file
    nfiles = int(features.shape[1] / 100)

    # Calculate expected transit time for title of plot
    transit_time_expected = ((float(fwhm) / 1000)**2 *
                             1000) / (diffrate_of_interest * 8 * np.log(2.0))

    # get pandas Series of diffrates of interest
    diff = diffrates.where(diffrates == diffrate_of_interest).dropna()

    # get traces and labels according to the desired diffrate
    traces_of_interest = pd.DataFrame()
    labels_of_interest = pd.DataFrame()
    for diff_idx in diff.index:
        traces_tmp = features.iloc[:, diff_idx::nfiles]
        labels_tmp = labels.iloc[:, diff_idx::nfiles]
        traces_of_interest = pd.concat([traces_of_interest, traces_tmp],
                                       axis=1)
        labels_of_interest = pd.concat([labels_of_interest, labels_tmp],
                                       axis=1)

    if artifact == 0:
        labels_of_interest = labels_of_interest > lab_thresh
        # get symbol for legend in plot below
        label_operation = '<'
    elif artifact == 1:
        labels_of_interest = labels_of_interest < lab_thresh
        label_operation = '>'
    elif artifact == 2:
        labels_of_interest = labels_of_interest > lab_thresh
        label_operation = '<'
    else:
        raise ValueError('value for artifact has to be 0, 1 or 2')

    traces_of_interest = np.array(traces_of_interest).astype(np.float64)
    labels_of_interest = np.array(labels_of_interest)

    if plotperfect:
        labsum = labels_of_interest.sum(axis=0)

        idx_perfect_traces = []
        for idx, lab in enumerate(labsum):
            if lab == 0:
                idx_perfect_traces.append(idx)

        print('number of perfect traces (corruption threshold: {}): {}'.format(
            lab_thresh, len(idx_perfect_traces)))

        traces_perfect = traces_of_interest[:, idx_perfect_traces]
        traces_perfect = pd.DataFrame(traces_perfect)
        traces_corrupted = np.delete(traces_of_interest,
                                     idx_perfect_traces,
                                     axis=1)
        labels_corrupted = np.delete(labels_of_interest,
                                     idx_perfect_traces,
                                     axis=1)
        # has to be DataFrame for prediction preprocessing
        traces_corrupted = pd.DataFrame(traces_corrupted)
        labels_corrupted = pd.DataFrame(labels_corrupted)
    else:
        print('number of traces to plot: {}'.format(number_of_traces))
        traces_corrupted = pd.DataFrame(
            traces_of_interest[:, :number_of_traces])
        labels_corrupted = pd.DataFrame(
            labels_of_interest[:, :number_of_traces])

    ntraces = len(traces_corrupted.columns)
    lab_out = correct_correlation_by_label(ntraces=ntraces,
                                           traces_of_interest=traces_corrupted,
                                           labels_of_interest=labels_corrupted,
                                           fwhm=fwhm)
    print('processed correlation with correction by label')
    pred_out = correct_correlation_by_prediction(
        ntraces=ntraces,
        traces_of_interest=traces_corrupted,
        model=model,
        pred_thresh=pred_thresh,
        win_len=win_len,
        zoomvector=zoomvector,
        fwhm=fwhm)
    print('processed correlation with correction by prediction')
    corrupt_out = correlation_of_arbitrary_trace(
        ntraces=ntraces, traces_of_interest=traces_corrupted, fwhm=fwhm)
    print('processed correlation without correction')
    if plotperfect:
        perfect_out = correlation_of_arbitrary_trace(
            ntraces=ntraces, traces_of_interest=traces_perfect, fwhm=fwhm)

    # PART 2: Plotting
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    sns.set(style='whitegrid')
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    ax1 = fig.add_subplot(gs[0, :-1])
    ax1.hist(lab_out[xunit],
             bins=xunit_bins,
             alpha=0.3,
             label='corrected by label with threshold: {}'.format(lab_thresh),
             color='C2')
    ax1.hist(
        pred_out[xunit],
        bins=xunit_bins,
        alpha=0.5,
        label='corrected by prediction with threshold: {}'.format(pred_thresh),
        color='C1')
    ax1.hist(corrupt_out[xunit],
             bins=xunit_bins,
             alpha=0.3,
             label='corrupted traces without correction',
             color='C0')

    ax2 = fig.add_subplot(gs[0, -1])
    ax2.hist(lab_out[2],
             bins=np.arange(0, 20001, 1000),
             alpha=0.3,
             label='trace length after correction by label',
             color='C2')
    ax2.hist(pred_out[2],
             bins=np.arange(0, 20001, 1000),
             alpha=0.5,
             label='trace length after correction by prediction',
             color='C1')
    ax2.hist(corrupt_out[2],
             bins=np.arange(0, 20001, 1000),
             alpha=0.3,
             label='length of corrupted traces without correction',
             color='C0')
    if plotperfect:
        ax1.hist(perfect_out[xunit],
                 bins=xunit_bins,
                 alpha=0.3,
                 label='"perfect traces" (corruption {} {})'.format(
                     label_operation, lab_thresh))
        ax2.hist(perfect_out[2],
                 bins=np.arange(0, 20001, 1000),
                 alpha=0.3,
                 label='length of "perfect traces" without correction')
        datadf = pd.DataFrame(data=[
            corrupt_out[xunit], pred_out[xunit], lab_out[xunit],
            perfect_out[xunit]
        ],
                              index=[
                                  'corrupted\nwithout\ncorrection',
                                  'corrected\nby predictions',
                                  'corrected\nby labels', 'perfect\ntrace'
                              ]).T
        boxplotcols = 4
    else:
        datadf = pd.DataFrame(
            data=[corrupt_out[xunit], pred_out[xunit], lab_out[xunit]],
            index=[
                'corrupted\nwithout\ncorrection', 'corrected\nby predictions',
                'corrected\nby labels'
            ]).T
        boxplotcols = 3

    ax3 = fig.add_subplot(gs[1, :-1])
    box = sns.boxplot(data=datadf, showfliers=False, orient='h')
    sns.stripplot(data=datadf, orient='h', color='.3')

    if xunit == 0:
        ax1.set_title('Histogram of diffusion rates by correlation',
                      fontsize=20)
        ax1.set_xlabel(r'Diffusion coefficients in $\mu m^2 / s$', size=16)
        ax1.axvline(
            x=diffrate_of_interest,
            color='r',
            label=r'simulated diffusion rate: {:.2f}$\mu m^2 / s$'.format(
                diffrate_of_interest))
        ax3.set_title('Boxplot of diffusion rates by correlation', fontsize=20)
        ax3.set_xlabel(r'Diffusion coefficients in $\mu m^2 / s$', size=16)
        ax3.axvline(
            x=diffrate_of_interest,
            color='r',
            label=r'simulated diffusion rate: {:.2f}$\mu m^2 / s$'.format(
                diffrate_of_interest))
    elif xunit == 1:
        ax1.set_title('Histogram of transit times by correlation', fontsize=20)
        ax1.set_xlabel(r'Transit times in $ms$', size=16)
        ax1.axvline(x=transit_time_expected,
                    color='r',
                    label=r'simulated transit time: {:.2f}$ms$'.format(
                        transit_time_expected))
        ax3.set_title('Boxplot of transit times by correlation', fontsize=20)
        ax3.set_xlabel(r'Transit times in $ms$', size=16)
        ax3.axvline(x=transit_time_expected,
                    color='r',
                    label=r'simulated transit time: {:.2f}$ms$'.format(
                        transit_time_expected))
    else:
        raise ValueError('value for xunit has to be 0 or 1')

    for i in range(boxplotcols):
        median_index = 4 + (i * 4 + i)
        lines = box.axes.get_lines()
        # every 4th line at the interval of 6 is median line
        # 0 -> p25, 1 -> p75, 2 -> lower whisker, 3 -> upper whisker, 4 -> p50
        # 5 -> upper extreme value (not here)
        median = round(lines[median_index].get_xdata()[0], 1)
        ax3.text(x=np.median(xunit_bins),
                 y=i,
                 s='median\n{:.2f}'.format(median),
                 va='center',
                 size=16,
                 fontweight='bold',
                 color='white',
                 bbox=dict(facecolor='#445A64'))
    ax1.set_ylabel('Number of fluorescence traces', fontsize=16)
    ax1.set_xlim((min(xunit_bins), max(xunit_bins)))
    ax1.legend(fontsize=16)
    ax2.set_title('Length of traces\nfor correlation', fontsize=20)
    ax2.set_xlabel(r'Length of traces in $ms$', fontsize=16)
    ax2.set_ylabel('Number of fluorescence traces', fontsize=16)
    ax3.set_xlim((min(xunit_bins), max(xunit_bins)))
    ax3.legend(fontsize=16)
    # ax3.grid(axis='x', color="0.9", linestyle='-', linewidth=1)
    # ax3.tick_params(axis='x', length=0)
    # ax3.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['bottom'].set_visible(False)

    if plotperfect:
        return lab_out, pred_out, perfect_out, corrupt_out
    return lab_out, pred_out, corrupt_out

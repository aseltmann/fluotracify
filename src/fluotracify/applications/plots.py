"""This module contains functions which perform artifact correction on
fluorescence timetraces at scale and plots statistics of the results"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fluotracify.applications import correction, correlate
from fluotracify.imports import ptu_utils as ptu
from fluotracify.simulations import (
    import_simulation_from_csv as isfc,
    plot_simulations as ps,
)


def plot_distribution_of_correlations_by_label_thresholds(
        diffrate_of_interest, thresh_arr, xunit, artifact, xunit_bins,
        experiment_params, nsamples, features, labels_artifact,
        labels_puretrace):
    """Examine ensemble correction of simulated fluorescence traces with artifacts

    The features (=fluorescence traces), labels (=ground truth), and diffrates
    have to be pandas DataFrames and their indeces have to match. Also, this
    function right now only works properly, if you read in the data via
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
    experiment_params : pandas DataFrame
        Contains metadata of the files (is obtained while loading the
        simulated data with the `import_from_csv` function from the
        fluotracify.simulations.import_simulation_from_csv module)
    nsamples : int
        Number of traces per .csv file (is obtained while loading the
        simulated data with the `import_from_csv` function from the
        fluotracify.simulations.import_simulation_from_csv module)
    features : pandas DataFrame
        Contains simulated fluorescence traces
    labels_artifact : pandas DataFrame
        Contains ground truth information of the simulated artifacts for
        each time step
    labels_puretrace : pandas DataFrame
        Contains ground truth information of the simulated trace without
        the addition of artifacts

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
    time_step = 1.
    # Calculate expected transit time for title of plot
    transit_time_expected = ((float(fwhm) / 1000)**2 *
                             1000) / (diffrate_of_interest * 8 * np.log(2.0))

    traces_of_interest = isfc.extract_traces_by_diffrates(
        diffrate_of_interest=diffrate_of_interest,
        traces=features,
        experiment_params=experiment_params,
        nsamples=nsamples)
    labpure_of_interest = isfc.extract_traces_by_diffrates(
        diffrate_of_interest=diffrate_of_interest,
        traces=labels_puretrace,
        experiment_params=experiment_params,
        nsamples=nsamples)
    labart_of_interest = isfc.extract_traces_by_diffrates(
        diffrate_of_interest=diffrate_of_interest,
        traces=labels_artifact,
        experiment_params=experiment_params,
        nsamples=nsamples)
    ntraces = len(traces_of_interest.columns)

    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[:, :-1])

    correlations_puretrace = correlate.correlation_of_arbitrary_trace(
        ntraces=ntraces,
        traces_of_interest=labpure_of_interest,
        fwhm=fwhm,
        time_step=time_step)

    ax1.hist(correlations_puretrace[xunit],
             bins=xunit_bins,
             alpha=0.3,
             label='pure traces',
             color='C0')

    for idx, thresh in enumerate(thresh_arr):
        print(idx, thresh)

        if artifact in (0, 2):
            labart_bool = labart_of_interest > thresh
        elif artifact == 1:
            labart_bool = labart_of_interest < thresh
        else:
            raise ValueError('value for artifact has to be 0 (for bright'
                             'clusters), 1 (for detector dropout) or 2 (for'
                             ' photobleaching)')

        correlations_corrected_by_label = ps.correct_correlation_by_label(
            ntraces=ntraces,
            traces_of_interest=traces_of_interest,
            labels_of_interest=labart_bool,
            fwhm=fwhm)

        ax1.hist(correlations_corrected_by_label[xunit],
                 bins=xunit_bins,
                 alpha=0.3,
                 label='threshold: {}'.format(thresh))

        ax2 = fig.add_subplot(gs[:, -1])
        ax2.hist(correlations_corrected_by_label[2],
                 bins=np.arange(0, 2**14, 1000),
                 alpha=0.3,
                 label='threshold: {}'.format(thresh))
    if xunit == 0:
        ax1.set_title(r'Diffusion rates by correlation with expected value '
                      r'{:.2f}$\mu m^2/s$'.format(diffrate_of_interest),
                      fontsize=20)
        ax1.set_xlabel(r'Diffusion coefficients in $\mu m^2/s$', size=16)
        ax1.axvline(x=diffrate_of_interest,
                    color='r',
                    label='simulated diffusion rate')
    elif xunit == 1:
        ax1.set_title(r'Transit times by correlation with expected value'
                      ' {:.2f}$ms$'.format(transit_time_expected),
                      fontsize=20)
        ax1.set_xlabel(r'Transit times in $ms$', size=16)
        ax1.axvline(x=transit_time_expected,
                    color='r',
                    label='simulated transit time')
    else:
        raise ValueError('value for xunit has to be 0 (for diffusion rates) '
                         'or 1 (for transit times)')
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
        model_type,
        xunit_bins,
        experiment_params,
        nsamples,
        features,
        labels_artifact,
        labels_puretrace,
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
    model_type : {0, 1}
        0: vgg
        1: unet
    xunit_bins : int or sequence of scalars or str, optional
        Binning of the histogram of xunit, see
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
    number_of_traces : int, None, optional
        Number of traces which shall be chosen. If 'None', the number of
        traces is determined of the number of perfect traces and they will be
        plotted as well.
    experiment_params : pandas DataFrame
        Contains metadata of the files (is obtained while loading the
        simulated data with the `import_from_csv` function from the
        fluotracify.simulations.import_simulation_from_csv module)
    nsamples : int
        Number of traces per .csv file (is obtained while loading the
        simulated data with the `import_from_csv` function from the
        fluotracify.simulations.import_simulation_from_csv module)
    features : pandas DataFrame
        Contains simulated fluorescence traces
    labels_artifact : pandas DataFrame
        Contains ground truth information of the simulated artifacts for
        each time step
    labels_puretrace : pandas DataFrame
        Contains ground truth information of the simulated trace without
        the addition of artifacts

    Returns
    -------
    lab_out : tuple
        Contains diffusion rates, transit times and trace lengths corrected by
        label
    pred_out : tuple
        Contains diffusion rates, transit times and trace lengths corrected by
        prediciton
    corrupt_out : tuple
        Contains diffusion rates, transit times and trace lengths of corrupted
        traces without correction
    pure_out : tuple
        Contains diffusion rates, transit times and trace lengths of 'pure
        traces' before the artifacts were added in the simulations

    Notes
    -----
    To be implemented:
    lognormfit : bool, optional - ,
        lognormfit=False
        If True, use scipy.stats.lognorm.fit routine to display a probability
        density function of a lognorm distribution fitted on the transit time
        histograms and display the mean of the fit as a vertical line.

    Commented out:
    plotperfect : bool
        If True, the 'perfect' traces (which have no time step with corruption
        above lab_thresh) will be plotted as well. If this parameter is chosen,
        number_of_traces should be 'None'. (Removed because with newer
        simulations I save out the traces without any artifact influence which
        I can now use directly)
    # if plotperfect:
    #     labsum = lab_art_of_interest.sum(axis=0)

    #     idx_perfect_traces = []
    #     for idx, lab in enumerate(labsum):
    #         if lab == 0:
    #             idx_perfect_traces.append(idx)

    #     print('number of perfect traces (corruption threshold: {}): {}'.format(
    #         lab_thresh, len(idx_perfect_traces)))

    #     traces_perfect = traces_of_interest[:, idx_perfect_traces]
    #     traces_perfect = pd.DataFrame(traces_perfect)
    #     traces_corrupted = np.delete(traces_of_interest,
    #                                  idx_perfect_traces,
    #                                  axis=1)
    #     labels_corrupted = np.delete(lab_art_of_interest,
    #                                  idx_perfect_traces,
    #                                  axis=1)
    #     # has to be DataFrame for prediction preprocessing
    #     traces_corrupted = pd.DataFrame(traces_corrupted)
    #     labels_corrupted = pd.DataFrame(labels_corrupted)
    """

    # PART 1: Calculation
    # fix constants (could be introduced as variables)
    fwhm = 250
    win_len = 128
    zoomvector = (5, 21)
    length_delimiter = 16384

    # Calculate expected transit time for title of plot
    transit_time_expected = ((float(fwhm) / 1000)**2 *
                             1000) / (diffrate_of_interest * 8 * np.log(2.0))

    traces_of_interest = isfc.extract_traces_by_diffrates(
        diffrate_of_interest=diffrate_of_interest,
        traces=features,
        experiment_params=experiment_params,
        nsamples=nsamples)
    labpure_of_interest = isfc.extract_traces_by_diffrates(
        diffrate_of_interest=diffrate_of_interest,
        traces=labels_puretrace,
        experiment_params=experiment_params,
        nsamples=nsamples)
    labart_of_interest = isfc.extract_traces_by_diffrates(
        diffrate_of_interest=diffrate_of_interest,
        traces=labels_artifact,
        experiment_params=experiment_params,
        nsamples=nsamples)

    if artifact in (0, 2):
        labart_of_interest = labart_of_interest > lab_thresh
    elif artifact == 1:
        labart_of_interest = labart_of_interest < lab_thresh
    else:
        raise ValueError('value for artifact has to be 0 (bright clusters), 1'
                         ' (detector dropout) or 2 (photobleaching)')

    traces_of_interest = np.array(traces_of_interest).astype(np.float64)
    labart_of_interest = np.array(labart_of_interest)
    labpure_of_interest = np.array(labpure_of_interest).astype(np.float64)

    print('number of traces to plot: {}'.format(number_of_traces))
    traces_corrupted = pd.DataFrame(traces_of_interest[:, :number_of_traces])
    labels_corrupted = pd.DataFrame(labart_of_interest[:, :number_of_traces])
    traces_pure = pd.DataFrame(labpure_of_interest[:, :number_of_traces])

    ntraces = len(traces_corrupted.columns)

    lab_out = ps.correct_correlation_by_label(
        ntraces=ntraces,
        traces_of_interest=traces_corrupted,
        labels_of_interest=labels_corrupted,
        fwhm=fwhm)
    print('processed correlation with correction by label')
    if model_type == 0:
        pred_out = correction.correct_correlation_by_vgg_prediction(
            ntraces=ntraces,
            traces_of_interest=traces_corrupted,
            model=model,
            pred_thresh=pred_thresh,
            win_len=win_len,
            zoomvector=zoomvector,
            fwhm=fwhm)
    elif model_type == 1:
        pred_out = correction.correct_correlation_by_unet_prediction(
            ntraces=ntraces,
            traces_of_interest=traces_corrupted,
            model=model,
            pred_thresh=pred_thresh,
            length_delimiter=length_delimiter,
            fwhm=fwhm)
    else:
        raise ValueError('value for model_type has to be 0 (vgg) or 1 (unet)')
    print('processed correlation with correction by prediction')

    corrupt_out = correlate.correlation_of_arbitrary_trace(
        ntraces=ntraces,
        traces_of_interest=traces_corrupted,
        fwhm=fwhm,
        time_step=1.)
    print('processed correlation without correction')

    pure_out = correlate.correlation_of_arbitrary_trace(
        ntraces=ntraces,
        traces_of_interest=traces_pure,
        fwhm=fwhm,
        time_step=1.)

    # PART 2: Plotting
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    sns.set(style='whitegrid')
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    ax1 = fig.add_subplot(gs[0, :-1])
    ax1.hist(pure_out[xunit],
             bins=xunit_bins,
             alpha=0.3,
             label='pure traces before simulation of artifact (control)',
             color='C3')
    ax1.hist(lab_out[xunit],
             bins=xunit_bins,
             alpha=0.3,
             label='corrected by label with threshold: {} (control)'.format(
                 lab_thresh),
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
    ax2.hist(pure_out[2],
             bins=np.arange(0, length_delimiter, 1000),
             alpha=0.3,
             label='length of pure traces (control)',
             color='C3')
    ax2.hist(lab_out[2],
             bins=np.arange(0, length_delimiter, 1000),
             alpha=0.3,
             label='trace length after correction by label',
             color='C2')
    ax2.hist(pred_out[2],
             bins=np.arange(0, length_delimiter, 1000),
             alpha=0.5,
             label='trace length after correction by prediction',
             color='C1')
    ax2.hist(corrupt_out[2],
             bins=np.arange(0, length_delimiter, 1000),
             alpha=0.3,
             label='length of corrupted traces without correction',
             color='C0')

    datadf = pd.DataFrame(data=[
        corrupt_out[xunit], pred_out[xunit], lab_out[xunit], pure_out[xunit]
    ],
                          index=[
                              'corrupted\nwithout\ncorrection',
                              'corrected\nby predictions',
                              'corrected\nby labels\n(control)',
                              'pure traces\n(control)'
                          ]).T
    boxplotcols = 4

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
        raise ValueError('value for xunit has to be 0 (diffusion coefficients)'
                         'or 1 (transit times)')

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

    return lab_out, pred_out, corrupt_out, pure_out, fig


def plot_experimental_traces_from_ptu_corrected_by_unet_prediction(
        path,
        model,
        pred_thresh,
        xunit,
        xunit_bins,
        photon_count_bin=1e6,
        additional_path=None,
        number_of_traces=None,
        verbose=False):
    """plot the distribution of correlations after correcting fluorescence
    traces corrupted by the given artifact applying different thresholds
    Parameters
    ----------
    path : str
        Folder which contains .ptu files with data.
    model
        Model used to correct the corrupted traces
    pred_thresh : float between 0 and 1
        Threshold you want to apply on the predictions
    xunit : {0, 1}
        0: plot histogram of diffusion rates in um / s
        1: plot histogram of transit times in ms
    xunit_bins : int or sequence of scalars or str, optional
        Binning of the histogram of xunit, see
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
    photon_count_bin : integer, optional
        Standard is 1e6, which means binning the tcspc data in the .ptu files
        to ms, which is the binning the neural net was trained for. If you want
        the prediction to be applied to a photon trace with *smaller* binning,
        you can choose this here (e.g. 1e5 means binning of 100us, 1e4 means
        binning of 10us etc). *Larger* binning will be ignored.
    number_of_traces : int, None, optional
        Number of traces which shall be chosen. If 'None', the number of
        traces is determined of the number of perfect traces and they will be
        plotted as well.
    additional_path : str, optional
        Folder which contains .ptu files with data. Idea is that you have two
        experiments whoses distributions you want to compare in one plot.
    verbose : bool
        If True, print out each file which was loaded
    Returns
    -------
    out : tuple of tuples
        orig_out1 : tuple of lists
            Contains diffusion rates, transit times and trace lengths of
            original trace
        pred_out1 : tuple of lists
            Contains diffusion rates, transit times and trace lengths corrected
            by prediciton
        ptu_metadata1 : pandas DataFrame
            Contains ptu header metadata + num_of_ch from process_tcspc_data
        if additional_path with .ptu traces if given:
        orig_out2 : tuple, optional
            Contains diffusion rates, transit times and trace lengths of
            original trace
        pred_out2 : tuple, optional
            Contains diffusion rates, transit times and trace lengths corrected
            by prediciton
        ptu_metadata2 : pandas DataFrame
            Contains ptu header metadata + num_of_ch from process_tcspc_data
    """

    # PART 1: Calculation
    # fix constants (could be introduced as variables)
    fwhm = 250
    # data from Pablo_structured_experiment has length of 10s = 10000ms and
    # unet can at the moment only take powers of 2 as input size
    length_delimiter = 8192
    print('Loading first dataset with bin=1e6. This can take a while...')
    try:
        ptu_1ms, ptu_metadata1 = ptu.import_from_ptu(
            path=path,
            file_delimiter=number_of_traces,
            photon_count_bin=1e6,
            verbose=verbose)
    except ValueError:
        raise ValueError('There was an error in loading .ptu files'
                         'from folder {}'.format(path))

    if photon_count_bin >= 1e6:
        print('Processing correlation of unprocessed first dataset')
        orig_out1 = correlate.correlation_of_arbitrary_trace(
            ntraces=number_of_traces,
            traces_of_interest=ptu_1ms.astype(np.float64),
            fwhm=fwhm,
            time_step=1.,
            length_delimiter=length_delimiter)
        print('Processing correlation with correction by prediction '
              'of first dataset')
        pred_out1 = correction.correct_correlation_by_unet_prediction(
            ntraces=number_of_traces,
            traces_of_interest=ptu_1ms.astype(np.float64),
            model=model,
            pred_thresh=pred_thresh,
            length_delimiter=length_delimiter,
            fwhm=fwhm)
    elif photon_count_bin < 1e6:
        # correlate function expects time_step for shifting x-axis
        # time_step_for_correlation = float(photon_count_bin / 1e6)
        # I changed it back to "1." because it gave strange results
        time_step_for_correlation = 1.

        print('Different binning was chosen for correlation. Loading first '
              'dataset with bin={}. This can take a while...'.format(
                  photon_count_bin))
        ptu_cor, _ = ptu.import_from_ptu(path=path,
                                         file_delimiter=number_of_traces,
                                         photon_count_bin=photon_count_bin,
                                         verbose=False)
        print('Processing correlation of unprocessed first dataset')
        orig_out1 = correlate.correlation_of_arbitrary_trace(
            ntraces=number_of_traces,
            traces_of_interest=ptu_cor.astype(np.float64),
            fwhm=fwhm,
            time_step=time_step_for_correlation,
            length_delimiter=length_delimiter)
        print('Processing correlation with correction by prediction '
              'of first dataset')
        pred_out1 = correction.correct_correlation_by_unet_prediction(
            ntraces=number_of_traces,
            traces_of_interest=ptu_1ms,
            model=model,
            pred_thresh=pred_thresh,
            length_delimiter=length_delimiter,
            fwhm=fwhm,
            traces_for_correlation=ptu_cor.astype(np.float64),
            bin_for_correlation=photon_count_bin)
    else:
        raise ValueError('photon_count_bin has to be a positive integer')

    if additional_path:
        print('Loading second dataset with bin=1e6. this can take a while...')
        try:
            ptu_1ms_add, ptu_metadata2 = ptu.import_from_ptu(
                path=additional_path,
                file_delimiter=number_of_traces,
                photon_count_bin=1e6,
                verbose=verbose)
        except ValueError:
            raise ValueError(
                'There was an error in loading .ptu files '
                'from the additional path {}'.format(additional_path))

        if photon_count_bin >= 1e6:
            print('Processing correlation of unprocessed second dataset')
            orig_out2 = correlate.correlation_of_arbitrary_trace(
                ntraces=number_of_traces,
                traces_of_interest=ptu_1ms_add.astype(np.float64),
                fwhm=fwhm,
                time_step=1.,
                length_delimiter=length_delimiter)
            print('Processing correlation with correction by prediction '
                  'of second dataset')
            pred_out2 = correction.correct_correlation_by_unet_prediction(
                ntraces=number_of_traces,
                traces_of_interest=ptu_1ms_add.astype(np.float64),
                model=model,
                pred_thresh=pred_thresh,
                length_delimiter=length_delimiter,
                fwhm=fwhm)
        elif photon_count_bin < 1e6:
            print(
                'Different binning was chosen for correlation. Loading second'
                ' dataset with bin={}. This can take a while...'.format(
                    photon_count_bin))
            ptu_cor2, _ = ptu.import_from_ptu(
                path=additional_path,
                file_delimiter=number_of_traces,
                photon_count_bin=photon_count_bin,
                verbose=False)
            print('Processing correlation of unprocessed second dataset')
            orig_out2 = correlate.correlation_of_arbitrary_trace(
                ntraces=number_of_traces,
                traces_of_interest=ptu_cor2.astype(np.float64),
                fwhm=fwhm,
                time_step=time_step_for_correlation,
                length_delimiter=length_delimiter)
            print('Processing correlation with correction by prediction '
                  'of second dataset')
            pred_out2 = correction.correct_correlation_by_unet_prediction(
                ntraces=number_of_traces,
                traces_of_interest=ptu_1ms_add,
                model=model,
                pred_thresh=pred_thresh,
                length_delimiter=length_delimiter,
                fwhm=fwhm,
                traces_for_correlation=ptu_cor2.astype(np.float64),
                bin_for_correlation=photon_count_bin)

    # PART 2: Plotting
    if additional_path:
        fig = plt.figure(figsize=(16, 16), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)

    gs = fig.add_gridspec(2, 3)
    sns.set(style='whitegrid')
    sns.set_palette('colorblind')
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    ax1 = fig.add_subplot(gs[0, :-1])
    ax1.hist(pred_out1[xunit],
             bins=xunit_bins,
             alpha=0.3,
             label='Tb-PEX5-eGFP, corrected by prediction with '
             'threshold: {}'.format(pred_thresh),
             color='C1')
    ax1.hist(orig_out1[xunit],
             bins=xunit_bins,
             alpha=0.3,
             label='Tb-PEX5-eGFP without correction',
             color='C0')
    if additional_path:
        ax1.hist(pred_out2[xunit],
                 bins=xunit_bins,
                 alpha=0.3,
                 label='Hs-PEX5-eGFP, corrected by prediction with '
                 'threshold: {}'.format(pred_thresh),
                 color='C3')
        ax1.hist(orig_out2[xunit],
                 bins=xunit_bins,
                 alpha=0.3,
                 label='Hs-PEX5-eGFP without correction',
                 color='C2')

    ax2 = fig.add_subplot(gs[0, -1])
    ax2.hist(pred_out1[2],
             bins=np.arange(0, 10001, 500),
             alpha=0.3,
             label='trace length of Tb-PEX5-eGFP after correction',
             color='C1')
    ax2.hist(orig_out1[2],
             bins=np.arange(0, 10001, 500),
             alpha=0.3,
             label='trace length of Tb-PEX5-eGFP without correction',
             color='C0')
    if additional_path:
        ax2.hist(pred_out2[2],
                 bins=np.arange(0, 10001, 500),
                 alpha=0.3,
                 label='trace length of Hs-PEX5-eGFP after correction',
                 color='C3')
        ax2.hist(orig_out2[2],
                 bins=np.arange(0, 10001, 500),
                 alpha=0.3,
                 label='trace length of "clean" PEX5-eGFP without correction',
                 color='C2')

    if additional_path:
        datadf = pd.DataFrame(data=[
            orig_out1[xunit], pred_out1[xunit], orig_out2[xunit],
            pred_out2[xunit]
        ],
                              index=[
                                  'Tb-PEX5-eGFP\nwithout\ncorrection',
                                  'Tb-PEX5-eGFP\ncorrected\nby predictions',
                                  'Hs-PEX5-eGFP\nwithout\ncorrection',
                                  'Hs-PEX5-eGFP\ncorrected\nby predictions',
                              ]).T
        boxplotcols = 4
    else:
        datadf = pd.DataFrame(data=[orig_out1[xunit], pred_out1[xunit]],
                              index=[
                                  'Tb-PEX5-eGFP\nwithout\ncorrection',
                                  'Tb-PEX5-eGFP\ncorrected\nby predictions'
                              ]).T
        boxplotcols = 2

    ax3 = fig.add_subplot(gs[1, :-1])
    box = sns.boxplot(data=datadf, showfliers=False, orient='h')
    sns.stripplot(data=datadf, orient='h', color='.3')

    if xunit == 0:
        ax1.set_title('Histogram of diffusion rates by correlation',
                      fontsize=20)
        ax1.set_xlabel(r'Diffusion coefficients in $\mu m^2 / s$', size=16)
        ax3.set_title('Boxplot of diffusion rates by correlation', fontsize=20)
        ax3.set_xlabel(r'Diffusion coefficients in $\mu m^2 / s$', size=16)

    elif xunit == 1:
        ax1.set_title('Histogram of transit times by correlation', fontsize=20)
        ax1.set_xlabel(r'Transit times in $ms$', size=16)
        ax3.set_title('Boxplot of transit times by correlation', fontsize=20)
        ax3.set_xlabel(r'Transit times in $ms$', size=16)

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

    if additional_path:
        out = (orig_out1, pred_out1, ptu_metadata1, orig_out2, pred_out2,
               ptu_metadata2)
    else:
        out = (orig_out1, pred_out1, ptu_metadata1)
    return out

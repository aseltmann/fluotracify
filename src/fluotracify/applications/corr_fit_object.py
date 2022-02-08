"""Taken from Dominic Waithe's FCS Bulk Correlation Software FOCUSpoint
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

import copy
import datetime
import lmfit
import logging
import multipletau
import os
import time

from pathlib import Path
from typing import Literal, Optional, Union
from fluotracify.applications import correlate, correlate_cython
from fluotracify.applications import (fitting_methods_SE as SE,
                                      fitting_methods_GS as GS,
                                      fitting_methods_VD as VD,
                                      fitting_methods_PB as PB)
from fluotracify.imports import (asc_utils as asc, csv_utils as csvu, pt2_utils
                                 as pt2, pt3_utils as pt3, ptu_utils as ptu,
                                 spc_utils as spc)
from fluotracify.training import preprocess_data as ppd

import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class PicoObject():
    """This is the class which holds the .pt3 data and parameters

        Returns
        -------
        subChanArr : list of int
            Number of channel of arrival for each photon
        trueTimeArr : list of float
            the macro time in ns (absolute time when each photon arrived)
        dtimeArr : list of ...
            the micro time in ns (lifetime of each photon). By default this
            is not saved out.
        resolution : float
            Time resolution in seconds? (e.g. 1.6e-5 means 160 ms(???))

        """
    def __init__(self, input_file, par_obj):
        # parameter object and fit object.
        log.debug('PicoObject: Start CorrObj creation.')
        self.par_obj = par_obj
        self.type = 'mainObject'

        # self.PIE = 0
        self.filepath = Path(input_file)
        self.nameAndExt = os.path.basename(self.filepath).split('.')
        self.name = self.nameAndExt[0]
        self.ext = self.nameAndExt[-1]
        self.par_obj.data.append(input_file)
        self.par_obj.objectRef.append(self)

        # Imports pt3 file format to object.
        self.unqID = self.par_obj.numOfLoaded
        self.objId = []
        self.plotOn = True

        self.NcascStart = self.par_obj.NcascStart
        self.NcascEnd = self.par_obj.NcascEnd
        self.Nsub = self.par_obj.Nsub

        self.timeSeriesDividend = 1000000
        self.CV = []

        # used for photon decay
        self.photonLifetimeBin = self.par_obj.photonLifetimeBin

        # define dictionary variables for methods
        (self.photonDecay, self.decayScale, self.photonDecayMin,
         self.photonDecayNorm, self.kcount, self.brightnessNandB,
         self.numberNandB, self.timeSeries, self.autoNorm, self.autotime,
         self.timeSeriesScale, self.timeSeriesSize, self.predictions,
         self.subChanArr, self.trueTimeArr, self.trueTimeWeights,
         self.photonCountBin) = ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                                 {}, {}, {}, {}, {}, {})

        self.importData()
        self.prepareChannels()
        self.getPhotonDecay()
        self.getTimeSeries()
        self.getPhotonCountingStats()
        # self.getCrossAndAutoCorrelation()
        self.dTimeInfo()

    def importData(self):
        # file import
        key = f'{self.name}'
        if self.ext == 'spc':
            (self.subChanArr[key], self.trueTimeArr[key], self.dTimeArr,
             self.resolution) = spc.spc_file_import(self.filepath)
        elif self.ext == 'asc':
            (self.subChanArr[key], self.trueTimeArr[key], self.dTimeArr,
             self.resolution) = asc.asc_file_import(self.filepath)
        elif self.ext == 'pt2':
            (self.subChanArr[key], self.trueTimeArr[key], self.dTimeArr,
             self.resolution) = pt2.pt2import(self.filepath)
        elif self.ext == 'pt3':
            (self.subChanArr[key], self.trueTimeArr[key], self.dTimeArr,
             self.resolution) = pt3.pt3import(self.filepath)
        elif self.ext == 'ptu':
            (out, self.ptu_tags, self.ptu_num_records,
             self.glob_res) = ptu.import_ptu(self.filepath)
            if out is not False:
                (self.subChanArr[key], self.trueTimeArr[key], self.dTimeArr,
                 self.resolution) = (out["chanArr"], out["trueTimeArr"],
                                     out["dTimeArr"], out["resolution"])
                # Remove Overflow and Markers; they are not handled at the
                # moment.
                self.subChanArr[key] = np.array([
                    i for i in self.subChanArr[key]
                    if not isinstance(i, tuple)
                ])
                self.trueTimeArr[key] = np.array([
                    i for i in self.trueTimeArr[key]
                    if not isinstance(i, tuple)
                ])
                self.dTimeArr = np.array(
                    [i for i in self.dTimeArr if not isinstance(i, tuple)])
            # out = ptuimport(self.filepath)
            # if out is not False:
            #     (self.subChanArr, self.trueTimeArr, self.dTimeArr,
            #      self.resolution) = out
            else:
                self.par_obj.data.pop(-1)
                self.par_obj.objectRef.pop(-1)
                self.exit = True
        elif self.ext == 'csv':
            (self.subChanArr[key], self.trueTimeArr[key], self.dTimeArr,
             self.resolution) = csvu.csvimport(self.filepath)
            # If the file is empty.
            if self.subChanArr[key] is None:
                # Undoes any preparation of resource.
                self.par_obj.data.pop(-1)
                self.par_obj.objectRef.pop(-1)
                self.exit = True
        else:
            self.exit = True
        log.debug('Finished import.')

    def prepareChannels(self):
        if self.type == 'subObject':
            self.subArrayGeneration(self.xmin, self.xmax)

        # Colour assigned to file.
        self.color = self.par_obj.colors[self.unqID % len(self.par_obj.colors)]

        # How many channels there are in the files.
        self.ch_present = np.sort(np.unique(np.array(
            self.subChanArr[f'{self.name}'])))
        for i in range(self.ch_present.__len__() - 1, -1, -1):
            if self.ch_present[i] > 8:
                self.ch_present = np.delete(self.ch_present, i)

        if self.ext in ('pt3', 'ptu', 'pt2'):
            self.numOfCH = self.ch_present.__len__()
        else:
            self.numOfCH = self.ch_present.__len__()

        self.indx_arr = []
        # I order them this way, for systematic ordering in the plotting.
        # All the plots are included. First the auto, then the cross.
        for i in range(self.numOfCH):
            self.indx_arr.append([i, i])
            for j in range(self.numOfCH):
                if i != j:
                    self.indx_arr.append([i, j])

        log.debug('Finished prepareChannels() with %s channels: %s',
                  self.numOfCH, self.ch_present)

    def getPhotonDecay(self, photonLifetimeBin=None, name=None):
        """Gets photon decay curve from TCSPC data, specifically lifetimes

        Parameters
        ----------
        photonLifetimeBin : int
            bin for calculation of photon decay. Photon lifetimes in the
            time interval x to x+photonLifetimeBin are aggregated
        name : optional, str
            The photon decay and scale are added to self via a dictionary with
            the key "name". If None, the current time is taken as a key.

        Returns
        -------
        Nothing, but assigns to self:
            self.photonDecay : dict of dict of list
                - 1st dict a wrapper from name
                - 2nd dict for each given channel:
                    - the original list of binned lifetimes
                    - the list of binned lifetimes, substracted by the minimum
                      lifetime
                    - the list of binned lifetimes, minmax-normalized
            self.decayScale : dict of dict of list
                - 1st dict a wrapper from name
                - 2nd dict for each given channel:
                    - the list of the centered value for each bin
        """
        if photonLifetimeBin is not None:
            self.photonLifetimeBin = int(photonLifetimeBin)
        name = f'{self.name}' if name is None else f'{name}'
        self.photonDecay[name] = phd = {}
        self.decayScale[name] = dsc = {}

        for i in range(self.numOfCH):
            key = f'CH{i}_BIN{self.photonLifetimeBin}'
            photonDecay, decayScale = self.time2bin(
                np.array(self.dTimeArr), np.array(self.subChanArr[name]),
                self.ch_present[i], self.photonLifetimeBin)
            phd[f'Orig_{key}'] = np.array(photonDecay)
            dsc[key] = decayScale

            # Normalisation of the decay functions.
            if np.sum(photonDecay) > 0:
                phd[f'Min_{key}'] = (photonDecay - np.min(photonDecay))
                phd[f'Norm_{key}'] = (phd[f'Min_{key}'] /
                                      np.max(phd[f'Min_{key}']))
            else:
                (phd[f'Min_{key}'], phd[f'Norm_{key}']) = 0, 0
        log.debug('Finished getPhotonDecay() with name %s', name)

    def getTimeSeries(self,
                      photonCountBin=None,
                      truetime_name=None,
                      timeseries_name=None):
        """Gets time series from TCSPC data, specifically photon arrival times

        Parameters
        ----------
        photonCountBin : optional, int
            bin for calculation of time series. Photons arriving in the
            time interval x to x+photonCountBin are aggregated
        truetime_name : optional, str
            the key to self.trueTimeArr which determines the photon arrival
            times to use for the construction of the time series
        timeseries_name : optional, str
            The time series and time series scale are added to self via a
            dictionary with the key "name". If None, self.name is taken
            as the key.
        """
        # update if method is called again with new parameters
        name = f'{self.name}' if truetime_name is None else f'{truetime_name}'
        ts_name = f'{self.name}' if timeseries_name is None else (
            f'{timeseries_name}')
        if name not in self.trueTimeArr:
            raise ValueError(f'key={name} is no valid key to dictionary'
                             ' self.trueTimeArr.')
        self.timeSeries[ts_name] = tser = {}
        self.timeSeriesScale[ts_name] = tss = {}
        if photonCountBin is not None:
            self.photonCountBin[ts_name] = tsb = float(photonCountBin)
        else:
            self.photonCountBin[ts_name] = tsb = float(
                self.par_obj.photonCountBin)

        for i in range(self.numOfCH):
            key = f'CH{self.ch_present[i]}_BIN{tsb}'
            timeSeries, timeSeriesScale = self.time2bin(
                np.array(self.trueTimeArr[name]) / self.timeSeriesDividend,
                np.array(self.subChanArr[name]), self.ch_present[i], tsb)
            tser[key] = timeSeries
            tss[key] = timeSeriesScale
        log.debug(
            'Finished getTimeSeries() with truetime_name %s'
            ', timeseries_name %s', name, ts_name)

    def getPhotonCountingStats(self, name=None):
        """Gets photon counting statistics from time series

        Parameters
        ----------
        name : str
            Key to get time series and time series scale from dictionary
        """
        name = f'{self.name}' if name is None else f'{name}'
        if name not in self.timeSeries:
            raise ValueError(f'key={name} is not a valid key for the'
                             'dictionary self.timeSeries')
        (self.kcount[f'{name}'], self.brightnessNandB[f'{name}'],
         self.numberNandB[f'{name}']) = {}, {}, {}

        for i in range(self.numOfCH):
            key = f'CH{self.ch_present[i]}_BIN{self.photonCountBin[name]}'

            (kcount, brightnessNandB,
             numberNandB) = correlate.photonCountingStats(
                 self.timeSeries[f'{name}'][key],
                 self.timeSeriesScale[f'{name}'][key])
            self.kcount[f'{name}'][key] = kcount
            self.brightnessNandB[f'{name}'][key] = brightnessNandB
            self.numberNandB[f'{name}'][key] = numberNandB
        log.debug('Finished getPhotonCountingStats() with name: %s', name)

    def getCrossAndAutoCorrelation(self, name=None):
        """Gets autocorrelation of photons and crosscorrelation if there are
        more than 1 Channel

        Parameters
        ----------
        name : str
        """
        name = f'{self.name}' if name is None else f'{name}'
        self.autoNorm[name] = an = {}
        # Correlation combinations.
        # Provides ordering of files and reduces repetition.
        corr_array = []
        corr_comb = []
        for i in range(self.numOfCH):
            corr_array.append([])
            for j in range(self.numOfCH):
                if i < j:
                    corr_comb.append([i, j])
                corr_array[i].append([])

        for i, j in corr_comb:
            log.debug(
                'Starting first crossAndAuto() with ch_present[i] %s '
                'and ch_present[j] %s', self.ch_present[i], self.ch_present[j])
            corr_fn, autotime = self.crossAndAuto(
                np.array(self.trueTimeArr[name]),
                np.array(self.subChanArr[name]),
                [self.ch_present[i], self.ch_present[j]])
            if corr_array[i][i] == []:
                an[f'CH{i}_CH{i}'] = corr_fn[:, 0, 0].reshape(-1)
            if corr_array[j][j] == []:
                an[f'CH{j}_CH{j}'] = corr_fn[:, 1, 1].reshape(-1)
            an[f'CH{i}_CH{j}'] = corr_fn[:, 0, 1].reshape(-1)
            an[f'CH{j}_CH{i}'] = corr_fn[:, 1, 0].reshape(-1)

        if self.numOfCH == 1:
            log.debug(
                'Starting first crossAndAuto() with ch_present[i] %s '
                'and ch_present[j] %s', self.ch_present[i], self.ch_present[j])
            # FIXME: What is i and j here??
            corr_fn, autotime = self.crossAndAuto(
                np.array(self.trueTimeArr[name]),
                np.array(self.subChanArr[name]),
                [self.ch_present[i], self.ch_present[j]])
            an['CH0_CH0'] = corr_fn[:, 0, 0].reshape(-1)

        self.autotime[name] = autotime
        log.debug('Finished crossAndAuto()')

    def dTimeInfo(self):
        self.dTimeMin = 0
        self.dTimeMax = np.max(self.dTimeArr)
        self.subDTimeMin = self.dTimeMin
        self.subDTimeMax = self.dTimeMax
        self.exit = False
        # del self.subChanArr
        # del self.trueTimeArr
        del self.dTimeArr
        log.debug('Finished dTimeInfo()')

    def time2bin(self, time_arr, chan_arr, chan_num, win_int):
        """A binning method for arrival times (=photon time trace) or for
        lifetimes (=decay scale)

        Parameters
        ----------
        time_arr : np.array or list
            arrival times or lifetimes to bin
        chan_arr : np.array or list
            the channel for each photon arrival time or lifetime in time_arr
        chan_num : int
            which channel to choose in chan_arr
        win_int : int
            binning window

        Returns
        -------
        photons_in_bin : list
            list of amount of photons in arrival time / lifetime bin
        bins_scale : list
            the centers of the bins corresponding to photons_in_bin
        """
        time_arr = np.array(time_arr)
        # This is the point and which each channel is identified.
        time_ch = time_arr[chan_arr == chan_num]
        # Find the first and last entry
        first_time = 0  # np.min(time_ch).astype(np.int32)
        tmp_last_time = np.max(time_ch).astype(np.int32)
        # We floor this as the last bin is always incomplete and so we discard
        # photons.
        num_bins = np.floor((tmp_last_time - first_time) / win_int)
        last_time = num_bins * win_int
        bins = np.linspace(first_time, last_time, int(num_bins) + 1)
        photons_in_bin, _ = np.histogram(time_ch, bins)
        # bins are valued as half their span.
        bins_scale = bins[:-1] + (win_int / 2)
        # bins_scale =  np.arange(0,decayTimeCh.shape[0])
        log.debug('Finished time2bin. last_time=%s, num_bins=%s', last_time,
                  num_bins)
        return list(photons_in_bin), list(bins_scale)

    def crossAndAuto(self, trueTimeArr, subChanArr, channelsToUse):
        # For each channel we loop through and find only those in the correct
        # time gate.
        # We only want photons in channel 1 or two.
        if self.numOfCH == 1:
            indices = subChanArr == channelsToUse[0]
            y = trueTimeArr[indices]
            validPhotons = subChanArr[indices]
        else:
            indices0 = subChanArr == channelsToUse[0]
            indices1 = subChanArr == channelsToUse[1]
            indices = indices0 + indices1
            y = trueTimeArr[indices]
            validPhotons = subChanArr[indices]

        log.debug('crossAndAuto: sum(indeces)=%s', sum(indices))

        # Creates boolean for photon events in either channel.
        num = np.zeros((validPhotons.shape[0], 2))
        num[:, 0] = (np.array([np.array(validPhotons) == channelsToUse[0]
                               ])).astype(np.int32)
        if self.numOfCH > 1:
            num[:, 1] = (np.array([np.array(validPhotons) == channelsToUse[1]
                                   ])).astype(np.int32)

        self.count0 = np.sum(num[:, 0])
        self.count1 = np.sum(num[:, 1])
        log.debug('crossAndAuto: finished preparation')

        auto, autotime = correlate.tttr2xfcs(y, num, self.NcascStart,
                                             self.NcascEnd, self.Nsub)
        log.debug('Finished crossAndAuto - tttr2xfcs().')

        # Normalisation of the TCSPC data:
        maxY = np.ceil(max(trueTimeArr))
        autoNorm = np.zeros((auto.shape))
        autoNorm[:, 0, 0] = ((auto[:, 0, 0] * maxY) /
                             (self.count0 * self.count0)) - 1

        if self.numOfCH > 1:
            autoNorm[:, 1, 1] = ((auto[:, 1, 1] * maxY) /
                                 (self.count1 * self.count1)) - 1
            autoNorm[:, 1, 0] = ((auto[:, 1, 0] * maxY) /
                                 (self.count1 * self.count0)) - 1
            autoNorm[:, 0, 1] = ((auto[:, 0, 1] * maxY) /
                                 (self.count0 * self.count1)) - 1
        log.debug('Finished crossAndAuto()')
        return autoNorm, autotime

    def subArrayGeneration(self, xmin, xmax):
        if (xmax < xmin):
            xmin1 = xmin
            xmin = xmax
            xmax = xmin1
        # self.subChanArr = np.array(self.chanArr)
        # Finds those photons which arrive above certain time or below certain
        # time
        photonInd = np.logical_and(self.dTimeArr >= xmin,
                                   self.dTimeArr <= xmax).astype(np.bool)
        self.subChanArr[f'{self.name}'][np.invert(photonInd).astype(
            np.bool)] = 16

    def predictTimeSeries(self, model, scaler, name=None):
        """Takes a timetrace, performs preprocessing, and applies a compiled
        unet for artifact detection

        Parameters
        ----------
        model : tf.keras.Functional model
        scaler : ('standard', 'robust', 'maxabs', 'quant_g', 'minmax', l1',
                  'l2')
            Scales / normalizes the input trace. Check with which scaler the
            training data of the model was scaled and use the same one.

        Returns
        -------
        Nothing, but assigns two new variables to self
        self.predictions : numpy ndarray, dtype=float32, shape=(input_size,)
            Predictions between 0 and 1, 0 = no artifact, 1 = artifact
        self.timeSeriesPrepro : numpy ndarray, dtype=float32,
                                shape=(input_size,)
            The preprocessed time trace scaled according to scaler, and
            cropped according to input_size.

        Note
        ----
        - The input size of the model:
            The prediction is made with the trace padded with the median to an
            input size of at least 1024, or if bigger to the size of the next
            biggest power of 2, e.g. 2**13 (8192), 2**14 (16384), ...
            This is necessary to avoid tensorflow throwing a size mismatch
            error. The algorithm was trained on traces with lenghts of 2**14,
            the experimental test data had a length of 2**13, so these sizes
            are known to work well.
        - At the moment we assume one trace (timeSeries[0])
        """
        name = f'{self.name}' if name is None else f'{name}'
        if name not in self.timeSeries:
            raise ValueError(f'name={name} is not a valid key for the'
                             'dictionary self.timeSeries')
        self.predictions[name], self.timeSeriesSize[name] = {}, {}

        for key, trace in self.timeSeries[name].copy().items():
            trace = np.array(trace).astype(float)
            trace_size = trace.size
            if trace_size < 1024:
                input_size = 1024
            else:
                input_size = 2**(np.ceil(np.log2(trace_size))).astype(int)
            pad_size = input_size - trace_size

            # pad trace for unet input
            trace = np.pad(trace, pad_width=(0, pad_size), mode='median')

            # scale trace
            trace = np.reshape(trace, newshape=(-1, 1))
            try:
                trace = ppd.scale_trace(trace, scaler)
            except Exception as ex:
                raise ValueError('Scaling failed.') from ex

            # predict trace
            trace = np.reshape(trace, newshape=(1, -1, 1))
            try:
                predictions = model.predict(trace, verbose=0).flatten()
            except Exception as ex:
                raise ValueError('The prediction failed. Double-check correct'
                                 ' input model, input size..') from ex

            self.predictions[name][f'{key}'] = predictions
            self.timeSeries[name][f'{key}_PREPRO'] = trace.flatten().astype(
                np.float32)
            self.timeSeriesSize[name][f'{key}'] = trace_size
        log.debug('Finished predictTimeSeries() with name=%s', name)

    def correctTCSPC(self,
                     pred_thresh=0.5,
                     method='weights',
                     weight=None,
                     truetime_name=None,
                     timeseries_name=None):
        """Takes the artifact prediction from the time series and removes
        the artifacts in the TCSPC data

        Parameters
        ----------
        pred_thresh : float between 0 and 1
            If prediction is lower than pred_thresh, the time step is assumed
            to show 'no corruption'
        method : ['weights', 'delete', 'delete_and_shift']
            'weights' : give a weight to photons which are classified as
            artifacts, see argument =weight=.
            'delete' : photons classified as artifacts are deleted and a new
            dict in self.trueTimeArr is constructed with the remaining photons)
            The time series constructed from this trueTimeArr will have drops
            to 0 where photons were deleted
            'delete_and_shift' : Like 'delete', but additionally adjust the
            photon arrival times of all photons by shifting each photon by
            the bin size which was deleted before. The time series constructed
            from this trueTimeArr will have no drops, all ends are annealed to
            each other.
        weight = optional, float or None
            Only used if method='weights'. Photons classified as artifacts are
            given this weight. If None (or explicitly set to 0), the weight
            will be set to 0, meaning the photons will not be correlated

        Returns
        -------
        """
        name = f'{self.name}' if truetime_name is None else f'{truetime_name}'
        ts_name = f'{self.name}' if timeseries_name is None else (
            f'{timeseries_name}')
        if name not in self.trueTimeArr:
            raise ValueError(f'key={name} is no valid key to dictionary'
                             ' self.trueTimeArr.')
        if ts_name not in self.timeSeries:
            raise ValueError(f'key={ts_name} is not a valid key for the'
                             'dictionary self.timeSeries. Run method'
                             'getTimeSeries first.')
        if ts_name not in self.predictions:
            raise ValueError(f'key={ts_name} is not a valid key for the'
                             'dictionary self.predictions. Run method'
                             'predictTimeSeries() first.')
        methods = ['weights', 'delete', 'delete_and_shift']
        if method not in methods:
            raise ValueError(f'method has to be in {methods}.')
        if method == 'weight' and (not isinstance(weight, float)
                                   and weight not in [None, 0, 1]):
            raise ValueError('if method == "weight", the argument weight'
                             ' has to be a float, 0, 1 or None')

        for key, trace in self.timeSeries[ts_name].copy().items():
            if 'PREPRO' in key or 'CORRECTED' in key:
                continue
            metadata = key.split('_')
            chan = int(metadata[0].strip('CH'))
            trace = np.array(trace)
            trace_scale = np.array(self.timeSeriesScale[ts_name][f'{key}'])
            predictions = self.predictions[ts_name][f'{key}']
            timeSeriesSize = self.timeSeriesSize[ts_name][f'{key}']
            subChanCorrected = np.array(self.subChanArr[name])
            channelMask = subChanCorrected == chan
            subChanCorrected = subChanCorrected[channelMask]
            trueTimeCorrected = np.array(self.trueTimeArr[name])
            trueTimeCorrected = trueTimeCorrected[channelMask]

            # get prediction as time series mask and photon arrival time mask
            timeSeriesMask = predictions[:timeSeriesSize] > pred_thresh
            photonMask = np.repeat(timeSeriesMask, trace)
            # match trueTimeArr and subChanArr shape to prediction
            subChanCorrected = subChanCorrected[:photonMask.size]
            trueTimeCorrected = trueTimeCorrected[:photonMask.size]
            log.debug(
                'correctTCSPC: some samples: subChan %s, truetime %s,'
                'photonMask %s, channelMask %s', subChanCorrected.size,
                trueTimeCorrected.size, photonMask.size, np.size(channelMask))

            if method in ['delete', 'delete_and_shift']:
                photon_count_bin = int(metadata[1].strip('BIN'))
                # delete photons classified as artifactual
                trueTimeCorrected = np.delete(trueTimeCorrected, photonMask)
                subChanCorrected = np.delete(subChanCorrected, photonMask)
                log.debug('correctTCSPC: deleted %s photons of %s photons.',
                          len(self.trueTimeArr[name]) - len(trueTimeCorrected),
                          len(self.trueTimeArr[name]))

                if method == 'delete_and_shift':
                    # moves the photons as if the deleted bins never existed
                    idxphot = 0
                    for nphot, artifact in zip(trace, timeSeriesMask):
                        if artifact:
                            trueTimeCorrected[idxphot:] -= photon_count_bin
                        else:
                            idxphot += nphot
                    log.debug('correctTCSPC: shifted non-deleted photon '
                              'arrival times by photonCountBin=%s',
                              photon_count_bin)
                    tsCorrected = np.delete(trace, timeSeriesMask)
                    tsScaleCorrected = np.delete(trace_scale, timeSeriesMask)
                else:
                    tsCorrected = np.where(timeSeriesMask == 1, 0, trace)
                    tsScaleCorrected = trace_scale
                self.timeSeries[ts_name][f'{key}_CORRECTED'] = tsCorrected
                self.timeSeriesScale[ts_name][f'{key}_CORRECTED'] = (
                     tsScaleCorrected)
                self.trueTimeArr[f'{metadata[0]}_{ts_name}_CORRECTED'] = (
                    trueTimeCorrected)
                self.subChanArr[f'{metadata[0]}_{ts_name}_CORRECTED'] = (
                    subChanCorrected)

            elif method == 'weights':
                weight = float(0) if weight is None else float(weight)
                photon_weights = np.zeros((subChanCorrected.shape[0], 2))
                # for autocorrelation, only channel [:, 0] is relevant
                photon_weights[:, 0] = np.where(photonMask == 1, weight,
                                                float(1))
                self.trueTimeWeights[f'{metadata[0]}_{ts_name}'] = (
                    photon_weights)
                self.trueTimeArr[f'{metadata[0]}_{ts_name}_FORWEIGHTS'] = (
                    trueTimeCorrected)

        log.debug('Finished correctTCSPC() with name %s, timeseries_name %s',
                  name, ts_name)

    def get_autocorrelation(self, method='tttr2xfcs', name=None):
        """Get Autocorrelation of either TCSPC data or time series data

        Parameters
        ----------
        method : ['tttr2xfcs', 'tttr2xfcs_with_weights', 'multipletau']
        name : string or tuple of strings
            if method='tttr2xfcs', name should be a key for the dictionaries
            self.trueTimeArr and self.subChanArr. If None, self.name is chosen
            as a key
            if method='tttr2xfcs_with_weights', additionally to above there
            should be a dict self.trueTimeWeights from correctTCSPC(
            method='weights')
            if method='multipletau', name should be a tuple with 2 strings:
                the first key to the dictionary self.timeSeries,
                the second key to the dictionary self.timeSeries['first key']
        """
        methods = ['tttr2xfcs', 'tttr2xfcs_with_weights', 'multipletau']
        if method not in methods:
            raise ValueError(f'{method} is not a valid method from {methods}.')

        if method == 'tttr2xfcs_with_weights':
            name_weights = f'{name}'.removesuffix('_FORWEIGHTS')
            if name_weights not in self.trueTimeWeights:
                raise ValueError(f'key={name_weights} is no valid key for the'
                                 'dict self.trueTimeWeights. Run correctTCSPC('
                                 'method=\'weights\') first.')

        if method in ['tttr2xfcs', 'tttr2xfcs_with_weights']:
            if method not in self.autoNorm:
                self.autoNorm[f'{method}'] = {}
                self.autotime[f'{method}'] = {}
            name = f'{self.name}' if name is None else f'{name}'
            if name not in self.trueTimeArr:
                raise ValueError(f'key={name} is no valid key for the dict '
                                 'self.trueTimeArr or self.subChanArr.')
            if name in self.autoNorm[f'{method}']:
                raise ValueError('self.autoNorm[\'tttr2xfcs\'] already has a'
                                 f' key={name} Check if your desired '
                                 'autocorrelation already happened.')
            metadata = name.split('_')
            chan = metadata[0].strip('CH')
            photon_count_bin = metadata[1].strip('BIN')

            log.debug(
                'get_autocorrelation: Starting tttr2xfcs correlation '
                'with name %s', name)

            for i in range(self.numOfCH):
                try:
                    chan = int(chan)
                    if chan != self.ch_present[i]:
                        log.debug(
                            'Skipping because key %s of trueTimeArr does'
                            ' give a hint on which channel was used and '
                            'it does not match channel %s', name, chan)
                        continue
                    photon_count_bin = float(photon_count_bin)
                    # just so that I avoid having two CHx_CHx in front
                    name = '_'.join(metadata[1:])
                    tt_key = f'CH{self.ch_present[i]}_{name}'
                except ValueError:
                    # if int(chan) fails
                    tt_key = name
                    photon_count_bin = self.photonCountBin[name]
                    log.debug(
                        'Given key %s of trueTimeArr does not include a '
                        'hint on which channel was used. Assume all '
                        'channels are used and continue', name)
                if method == 'tttr2xfcs':
                    key = (f'CH{self.ch_present[i]}_BIN{photon_count_bin}'
                           f'_{name.removesuffix("_CORRECTED")}')
                    autonorm, autotime = self.crossAndAuto(
                        self.trueTimeArr[tt_key], self.subChanArr[tt_key],
                        [self.ch_present[i], self.ch_present[i]])

                elif method == 'tttr2xfcs_with_weights':
                    key = (f'CH{chan}_BIN{photon_count_bin}_'
                           f'{name.removesuffix("_FORWEIGHTS")}')
                    tt_arr = self.trueTimeArr[f'CH{chan}_{name}']
                    tt_weights = self.trueTimeWeights[name_weights]
                    auto, autotime = correlate.tttr2xfcs(
                        y=tt_arr,
                        num=tt_weights,
                        NcascStart=self.NcascStart,
                        NcascEnd=self.NcascEnd,
                        Nsub=self.Nsub)
                    # Normalisation of the TCSPC data
                    maxY = np.ceil(max(tt_arr))
                    count = np.sum(tt_weights)
                    autonorm = np.zeros((auto.shape))
                    autonorm[:, 0,
                             0] = ((auto[:, 0, 0] * maxY) / (count**2)) - 1
                self.autoNorm[f'{method}'][key] = autonorm[:, i, i].reshape(
                    1, 1, -1)
                self.autotime[f'{method}'][key] = autotime.reshape(-1, 1)
        elif method == 'multipletau':
            if not isinstance(name, tuple) or len(name) != 2:
                raise ValueError(f'For method={method}, name={name} has to be'
                                 ' a tuple of length 2.')
            if name[1] not in self.timeSeries[name[0]]:
                raise ValueError(f'self.timeSeries[{name[0]}][{name[1]}] does'
                                 ' not exist')
            if 'multipletau' not in self.autoNorm:
                self.autoNorm['multipletau'] = {}
                self.autotime['multipletau'] = {}

            corr_fn = multipletau.autocorrelate(
                a=self.timeSeries[f'{name[0]}'][f'{name[1]}'],
                m=16,
                deltat=self.photonCountBin[f'{name[0]}'],
                normalize=True)
            # multipletau outputs autotime=0 as first correlation step, which
            # leads to problems with focus-fit-js
            self.autotime['multipletau'][f'{name[1]}_{name[0]}'] = (corr_fn[1:,
                                                                            0])
            self.autoNorm['multipletau'][f'{name[1]}_{name[0]}'] = (corr_fn[1:,
                                                                            1])

        log.debug('Finished get_autocorrelation() with method=%s, name=%s',
                  method, name)

    def save_autocorrelation(self, name, method, output_path='pwd'):
        """Save files as .csv"""
        if f'{method}' not in self.autoNorm:
            raise ValueError(f'method={method} is not a valid key for dict '
                             'self.autoNorm. Run get_autocorrelation first.')
        if f'{name}' not in self.autoNorm[f'{method}']:
            raise ValueError(f'name={name} is not a valid key for dict '
                             f'self.autoNorm[{method}].')
        metadata = f'{name}'.split('_')
        timeseries_name = '_'.join(metadata[:2])
        truetime_name = '_'.join(metadata[2:])
        chan = metadata[0].strip('CH')

        if truetime_name not in self.kcount:
            raise ValueError(f'self.kcount[{truetime_name}][{timeseries_name}]'
                             'does not exist. Run getTimeSeries and '
                             'getPhotonCountingStats first.')
        elif timeseries_name not in self.kcount[truetime_name]:
            raise ValueError(f'self.kcount[{truetime_name}][{timeseries_name}]'
                             'does not exist. Run getTimeSeries and '
                             'getPhotonCountingStats first.')

        if output_path == 'pwd':
            output_path = Path().parent.resolve()
        else:
            output_path = Path(output_path)
            if not output_path.is_dir():
                raise NotADirectoryError('output_path should be a directory or'
                                         ' "pwd"')
        output_file = (f'{datetime.date.today()}_{method}_'
                       f'{name.replace(".", "dot")}_correlation.csv')
        output_file = output_path / output_file
        autotime = self.autotime[f'{method}'][f'{name}'].flatten()
        autonorm = self.autoNorm[f'{method}'][f'{name}'].flatten()
        kcount = self.kcount[truetime_name][timeseries_name]
        numberNandB = self.numberNandB[truetime_name][timeseries_name]
        brightnessNandB = self.brightnessNandB[truetime_name][timeseries_name]

        # compatibility with FoCuS-fit-JS:
        # with 'w': utf-16le (doesn't work), utf-8 (works)
        # with 'wb': works with .encode() behind strings
        with open(output_file, 'w', encoding='utf-8') as out:
            out.write('version,3.0\n')
            out.write(f'numOfCH,{self.numOfCH}\n')
            out.write('type,point\n')
            out.write(f'parent_name,{method}\n')
            out.write(f'ch_type,{chan}_{chan}\n')
            out.write(f'kcount,{kcount}\n')
            out.write(f'numberNandB,{numberNandB}\n')
            out.write(f'brightnessNandB,{brightnessNandB}\n')
            out.write('carpet pos,0\n')
            out.write('pc,0\n')
            out.write(f'Time (ms),CH{chan} Auto-Correlation\n')
            for i in range(autotime.shape[0]):
                out.write(f'{autotime[i]},{autonorm[i]}\n')
            out.write('end\n')
        log.debug('Finished save_autocorrelation of file %s', output_file)

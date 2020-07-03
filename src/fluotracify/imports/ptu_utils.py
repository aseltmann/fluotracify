import io
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def import_ptu_to_memory(inputfilepath, outputfilepath=None, verbose=True):
    """imports PicoQuant ptu files and returns the contents as numpy arrays.

    Parameters
    ----------
    inputfilepath : str
        Path to PTU file to read
    outputfilepath : str, optional
        If outputfilepath is given, the metadata is printed in the given file.
    verbose : bool
        if True, prints out information while processing

    Returns
    -------
    out : dict of np.arrays
        out['trueTimeArr']: the macro time in ns (absolute time when each
        photon arrived)
        out['dTimeArr']: the micro time in ns (lifetime of each photon)
        out['chanArr']: number of channel where photon was detected
    tagDataList : list of tuples of tag name and tag value
        metadata from ptu header
    numRecords : int
        number of detected photons
    globRes : float
        resolution in seconds

    Code is adopted from: https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/blob/master/PTU/Python/Read_PTU.py"""
    # Read_PTU.py    Read PicoQuant Unified Histogram Files
    # This is demo code. Use at your own risk. No warranties.
    # Keno Goertz, PicoQUant GmbH, February 2018

    # Note that marker events have a lower time resolution and may therefore
    # appear in the file slightly out of order with respect to regular (photon)
    # event records. This is by design. Markers are designed only for
    # relatively coarse synchronization requirements such as image scanning.

    # T Mode data are written to an output file [filename]
    # We do not keep it in memory because of the huge amout of memory
    # this would take in case of large files. Of course you can change this,
    # e.g. if your files are not too big. Otherwise it is best process the
    # data on the fly and keep only the results.

    # Tag Types
    tyEmpty8 = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
    tyBool8 = struct.unpack(">i", bytes.fromhex("00000008"))[0]
    tyInt8 = struct.unpack(">i", bytes.fromhex("10000008"))[0]
    tyBitSet64 = struct.unpack(">i", bytes.fromhex("11000008"))[0]
    tyColor8 = struct.unpack(">i", bytes.fromhex("12000008"))[0]
    tyFloat8 = struct.unpack(">i", bytes.fromhex("20000008"))[0]
    tyTDateTime = struct.unpack(">i", bytes.fromhex("21000008"))[0]
    tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
    tyAnsiString = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
    tyWideString = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
    tyBinaryBlob = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]

    # Record types
    rtPicoHarpT3 = struct.unpack(">i", bytes.fromhex('00010303'))[0]
    rtPicoHarpT2 = struct.unpack(">i", bytes.fromhex('00010203'))[0]
    rtHydraHarpT3 = struct.unpack(">i", bytes.fromhex('00010304'))[0]
    rtHydraHarpT2 = struct.unpack(">i", bytes.fromhex('00010204'))[0]
    rtHydraHarp2T3 = struct.unpack(">i", bytes.fromhex('01010304'))[0]
    rtHydraHarp2T2 = struct.unpack(">i", bytes.fromhex('01010204'))[0]
    rtTimeHarp260NT3 = struct.unpack(">i", bytes.fromhex('00010305'))[0]
    rtTimeHarp260NT2 = struct.unpack(">i", bytes.fromhex('00010205'))[0]
    rtTimeHarp260PT3 = struct.unpack(">i", bytes.fromhex('00010306'))[0]
    rtTimeHarp260PT2 = struct.unpack(">i", bytes.fromhex('00010206'))[0]
    rtMultiHarpNT3 = struct.unpack(">i", bytes.fromhex('00010307'))[0]
    rtMultiHarpNT2 = struct.unpack(">i", bytes.fromhex('00010207'))[0]

    # global variables
    global inputfile
    global outputfile
    global recNum
    global oflcorrection
    global truensync
    global dlen
    global isT2
    global globRes
    global numRecords

    # if len(sys.argv) != 3:
    #    print("USAGE: Read_PTU.py inputfile.PTU outputfile.txt")
    #    exit(0)

    inputfile = open(inputfilepath, "rb")

    # Check if inputfile is a valid PTU file
    # Python strings don't have terminating NULL characters, so they're
    # stripped
    magic = inputfile.read(8).decode("utf-8").strip('\0')
    if magic != "PQTTTR":
        inputfile.close()
        raise ValueError("ERROR: Magic invalid, this is not a PTU file.")

    version = inputfile.read(8).decode("utf-8").strip('\0')
    if outputfilepath is not None:
        # The following is needed for support of wide strings
        outputfile = io.open(outputfilepath, "w+", encoding="utf-16le")
        outputfile.write("Tag version: %s\n" % version)

    # Write the header data to outputfile and also save it in memory.
    # There's no do ... while in Python, so an if statement inside the while
    # loop breaks out of it
    tagDataList = []  # Contains tuples of (tagName, tagValue)
    while True:
        tagIdent = inputfile.read(32).decode("utf-8").strip('\0')
        tagIdx = struct.unpack("<i", inputfile.read(4))[0]
        tagTyp = struct.unpack("<i", inputfile.read(4))[0]
        if tagIdx > -1:
            evalName = tagIdent + '(' + str(tagIdx) + ')'
        else:
            evalName = tagIdent
        if outputfilepath is not None:
            outputfile.write("\n%-40s" % evalName)
        if tagTyp == tyEmpty8:
            inputfile.read(8)
            if outputfilepath is not None:
                outputfile.write("<empty Tag>")
            tagDataList.append((evalName, "<empty Tag>"))
        elif tagTyp == tyBool8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if tagInt == 0:
                if outputfilepath is not None:
                    outputfile.write("False")
                tagDataList.append((evalName, "False"))
            else:
                if outputfilepath is not None:
                    outputfile.write("True")
                tagDataList.append((evalName, "True"))
        elif tagTyp == tyInt8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if outputfilepath is not None:
                outputfile.write("%d" % tagInt)
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyBitSet64:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if outputfilepath is not None:
                outputfile.write("{0:#0{1}x}".format(tagInt, 18))
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyColor8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if outputfilepath is not None:
                outputfile.write("{0:#0{1}x}".format(tagInt, 18))
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyFloat8:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            if outputfilepath is not None:
                outputfile.write("%-3E" % tagFloat)
            tagDataList.append((evalName, tagFloat))
        elif tagTyp == tyFloat8Array:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if outputfilepath is not None:
                outputfile.write("<Float array with %d entries>" % tagInt / 8)
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyTDateTime:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagTime = int((tagFloat - 25569) * 86400)
            tagTime = time.gmtime(tagTime)
            if outputfilepath is not None:
                outputfile.write(time.strftime("%a %b %d %H:%M:%S %Y",
                                               tagTime))
            tagDataList.append((evalName, tagTime))
        elif tagTyp == tyAnsiString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode("utf-8").strip("\0")
            if outputfilepath is not None:
                outputfile.write("%s" % tagString)
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyWideString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode(
                "utf-16le", errors="ignore").strip("\0")
            if outputfilepath is not None:
                outputfile.write(tagString)
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyBinaryBlob:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if outputfilepath is not None:
                outputfile.write("<Binary blob with %d bytes>" % tagInt)
            tagDataList.append((evalName, tagInt))
        else:
            raise ValueError("ERROR: Unknown tag type")
        if tagIdent == "Header_End":
            break

    # Reformat the saved data for easier access
    tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
    tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]

    # get important variables from headers
    numRecords = tagValues[tagNames.index("TTResult_NumberOfRecords")]
    globRes = tagValues[tagNames.index("MeasDesc_GlobalResolution")]
    if verbose:
        print("Writing %d records, this may take a while..." % numRecords)

    # prepare dictionary as output of function
    out = {}  # contains np.arrays with photons, truetimes and dtimes
    out['trueTimeArr'] = np.zeros(numRecords).astype(np.int64)
    out['dTimeArr'] = np.zeros(numRecords).astype(np.int64)
    out['chanArr'] = np.zeros(numRecords).astype(np.int64)

    def gotOverflow(count):
        global outputfile, recNum
        # outputfile.write("%u OFL * %2x\n" % (recNum, count)

    def gotMarker(timeTag, markers):
        global outputfile, recNum
        # outputfile.write("%u MAR %2x %u\n" % (recNum, markers, timeTag))

    def gotPhoton(timeTag, channel, dtime):
        global outputfile, isT2, recNum
        if isT2:
            # outputfile.write("%u CHN %1x %u %8.0lf\n" % (recNum, channel, \
            # timeTag, (timeTag * globRes * 1e12)))
            truetime = timeTag * globRes * 1e12
            out['trueTimeArr'][recNum] = truetime
            out['dTimeArr'][
                recNum] = dtime  # picoquant demo code does not save out dtime
            # - but for timetrace coversion, it isneeded
            out['chanArr'][recNum] = channel
        else:
            # outputfile.write("%u CHN %1x %u %8.0lf %10u\n" % (recNum,\
            # channel, timeTag, (timeTag * globRes * 1e9), dtime))
            truetime = timeTag * globRes * 1e9
            out['trueTimeArr'][recNum] = truetime
            out['dTimeArr'][recNum] = dtime
            out['chanArr'][recNum] = channel

    def readPT3():
        global inputfile, outputfile, recNum, oflcorrection, dlen, numRecords
        T3WRAPAROUND = 65536
        for recNum in range(0, numRecords):
            # The data is stored in 32 bits that need to be divided into
            # smaller groups of bits, with each group of bits representing a
            # different variable. In this case, channel, dtime and nsync. This
            # can easily be achieved by converting the 32 bits to a string,
            # dividing the groups with simple array slicing, and then
            # converting back into the integers.
            try:
                recordData = "{0:0{1}b}".format(
                    struct.unpack("<I", inputfile.read(4))[0], 32)
            except:
                raise ValueError('The file ended earlier than expected, at'
                                 'record %d/%d.' % (recNum, numRecords))

            channel = int(recordData[0:4], base=2)
            dtime = int(recordData[4:16], base=2)
            nsync = int(recordData[16:32], base=2)
            if channel == 0xF:  # Special record
                if dtime == 0:  # Not a marker, so overflow
                    gotOverflow(1)
                    oflcorrection += T3WRAPAROUND
                else:
                    truensync = oflcorrection + nsync
                    gotMarker(truensync, dtime)
            else:
                if channel == 0 or channel > 4:  # Should not occur
                    print("Illegal Channel: #%1d %1u" % (dlen, channel))
                    if outputfilepath is not None:
                        outputfile.write("\nIllegal channel ")
                truensync = oflcorrection + nsync
                gotPhoton(truensync, channel, dtime)
                dlen += 1
            if recNum % 100000 == 0:
                if verbose:
                    sys.stdout.write("\rProgress: %.1f%%" %
                                     (float(recNum) * 100 / float(numRecords)))
                    sys.stdout.flush()

    def readPT2():
        global inputfile, outputfile, recNum, oflcorrection, numRecords
        T2WRAPAROUND = 210698240
        for recNum in range(0, numRecords):
            try:
                recordData = "{0:0{1}b}".format(
                    struct.unpack("<I", inputfile.read(4))[0], 32)
            except:
                raise ValueError('The file ended earlier than expected, at'
                                 'record %d/%d.' % (recNum, numRecords))

            channel = int(recordData[0:4], base=2)
            dtime = int(recordData[4:32], base=2)
            if channel == 0xF:  # Special record
                # lower 4 bits of time are marker bits
                markers = int(recordData[28:32], base=2)
                if markers == 0:  # Not a marker, so overflow
                    gotOverflow(1)
                    oflcorrection += T2WRAPAROUND
                else:
                    # Actually, the lower 4 bits for the time aren't valid
                    # because they belong to the marker. But the error caused
                    # by them is so small that we can just ignore it.
                    truetime = oflcorrection + dtime
                    gotMarker(truetime, markers)
            else:
                if channel > 4:  # Should not occur
                    print("Illegal Channel: #%1d %1u" % (recNum, channel))
                    if outputfilepath is not None:
                        outputfile.write("\nIllegal channel ")
                truetime = oflcorrection + dtime
                gotPhoton(truetime, channel, dtime)
            if recNum % 100000 == 0:
                if verbose:
                    sys.stdout.write("\rProgress: %.1f%%" %
                                     (float(recNum) * 100 / float(numRecords)))
                    sys.stdout.flush()

    def readHT3(version):
        global inputfile, outputfile, recNum, oflcorrection, numRecords
        T3WRAPAROUND = 1024

        for recNum in range(0, numRecords):
            try:
                recordData = "{0:0{1}b}".format(
                    struct.unpack("<I", inputfile.read(4))[0], 32)
            except:
                raise ValueError('The file ended earlier than expected, at'
                                 'record %d/%d.' % (recNum, numRecords))

            special = int(recordData[0:1], base=2)
            channel = int(recordData[1:7], base=2)
            dtime = int(recordData[7:22], base=2)
            nsync = int(recordData[22:32], base=2)
            if special == 1:
                if channel == 0x3F:  # Overflow
                    # Number of overflows in nsync. If 0 or old version, it's
                    # an old style single overflow
                    if nsync == 0 or version == 1:
                        oflcorrection += T3WRAPAROUND
                        gotOverflow(1)
                    else:
                        oflcorrection += T3WRAPAROUND * nsync
                        gotOverflow(nsync)
                if 1 <= channel <= 15:  # markers
                    truensync = oflcorrection + nsync
                    gotMarker(truensync, channel)
            else:  # regular input channel
                truensync = oflcorrection + nsync
                gotPhoton(timeTag=truensync, channel=channel, dtime=dtime)
            if recNum % 100000 == 0:
                if verbose:
                    sys.stdout.write("\rProgress: %.1f%%" %
                                     (float(recNum) * 100 / float(numRecords)))
                    sys.stdout.flush()

    def readHT2(version):
        global inputfile, outputfile, recNum, oflcorrection, numRecords
        T2WRAPAROUND_V1 = 33552000
        T2WRAPAROUND_V2 = 33554432
        for recNum in range(0, numRecords):
            try:
                recordData = "{0:0{1}b}".format(
                    struct.unpack("<I", inputfile.read(4))[0], 32)
            except:
                raise ValueError('The file ended earlier than expected, at'
                                 'record %d/%d.' % (recNum, numRecords))

            special = int(recordData[0:1], base=2)
            channel = int(recordData[1:7], base=2)
            timetag = int(recordData[7:32], base=2)
            if special == 1:
                if channel == 0x3F:  # Overflow
                    # Number of overflows in nsync. If old version, it's an
                    # old style single overflow
                    if version == 1:
                        oflcorrection += T2WRAPAROUND_V1
                        gotOverflow(1)
                    else:
                        # old style overflow, shouldn't happen
                        if timetag == 0:
                            oflcorrection += T2WRAPAROUND_V2
                            gotOverflow(1)
                        else:
                            oflcorrection += T2WRAPAROUND_V2 * timetag
                if 1 <= channel <= 15:  # markers
                    truetime = oflcorrection + timetag
                    gotMarker(truetime, channel)
                if channel == 0:  # sync
                    truetime = oflcorrection + timetag
                    gotPhoton(truetime, 0, 0)
            else:  # regular input channel
                truetime = oflcorrection + timetag
                gotPhoton(truetime, channel + 1, 0)
            if recNum % 100000 == 0:
                if verbose:
                    sys.stdout.write("\rProgress: %.1f%%" %
                                     (float(recNum) * 100 / float(numRecords)))
                    sys.stdout.flush()

    oflcorrection = 0
    dlen = 0
    if outputfilepath is not None:
        outputfile.write("\n-----------------------\n")
    recordType = tagValues[tagNames.index("TTResultFormat_TTTRRecType")]
    if recordType == rtPicoHarpT2:
        isT2 = True
        if verbose:
            print("PicoHarp T2 data")
        if outputfilepath is not None:
            outputfile.write("PicoHarp T2 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ps\n")
        readPT2()
    elif recordType == rtPicoHarpT3:
        isT2 = False
        if verbose:
            print("PicoHarp T3 data")
        if outputfilepath is not None:
            outputfile.write("PicoHarp T3 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ns dtime\n")
        readPT3()
    elif recordType == rtHydraHarpT2:
        isT2 = True
        if verbose:
            print("HydraHarp V1 T2 data")
        if outputfilepath is not None:
            outputfile.write("HydraHarp V1 T2 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ps\n")
        readHT2(1)
    elif recordType == rtHydraHarpT3:
        isT2 = False
        if verbose:
            print("HydraHarp V1 T3 data")
        if outputfilepath is not None:
            outputfile.write("HydraHarp V1 T3 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ns dtime\n")
        readHT3(1)
    elif recordType == rtHydraHarp2T2:
        isT2 = True
        if verbose:
            print("HydraHarp V2 T2 data")
        if outputfilepath is not None:
            outputfile.write("HydraHarp V2 T2 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ps\n")
        readHT2(2)
    elif recordType == rtHydraHarp2T3:
        isT2 = False
        if verbose:
            print("HydraHarp V2 T3 data")
        if outputfilepath is not None:
            outputfile.write("HydraHarp V2 T3 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ns dtime\n")
        readHT3(2)
    elif recordType == rtTimeHarp260NT3:
        isT2 = False
        if verbose:
            print("TimeHarp260N T3 data")
        if outputfilepath is not None:
            outputfile.write("TimeHarp260N T3 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ns dtime\n")
        readHT3(2)
    elif recordType == rtTimeHarp260NT2:
        isT2 = True
        if verbose:
            print("TimeHarp260N T2 data")
        if outputfilepath is not None:
            outputfile.write("TimeHarp260N T2 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ps\n")
        readHT2(2)
    elif recordType == rtTimeHarp260PT3:
        isT2 = False
        if verbose:
            print("TimeHarp260P T3 data")
        if outputfilepath is not None:
            outputfile.write("TimeHarp260P T3 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ns dtime\n")
        readHT3(2)
    elif recordType == rtTimeHarp260PT2:
        isT2 = True
        if verbose:
            print("TimeHarp260P T2 data")
        if outputfilepath is not None:
            outputfile.write("TimeHarp260P T2 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ps\n")
        readHT2(2)
    elif recordType == rtMultiHarpNT3:
        isT2 = False
        if verbose:
            print("MultiHarp150N T3 data")
        if outputfilepath is not None:
            outputfile.write("MultiHarp150N T3 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ns dtime\n")
        readHT3(2)
    elif recordType == rtMultiHarpNT2:
        isT2 = True
        if verbose:
            print("MultiHarp150N T2 data")
        if outputfilepath is not None:
            outputfile.write("MultiHarp150N T2 data\n")
            outputfile.write("\nrecord# chan   nsync truetime/ps\n")
        readHT2(2)
    else:
        raise ValueError('ERROR: Unknown record type')

    inputfile.close()
    if outputfilepath is not None:
        outputfile.close()
    return out, tagDataList, numRecords, globRes


def time2bin(time_arr, chan_arr, chan_num, win_int):
    """bins tcspc data (either dtime or truetime). win_int gives the binning
    window.

    code is adopted from: https://github.com/dwaithe/FCS_point_correlator/blob/master/focuspoint/correlation_methods/correlation_methods.py#L157"""
    # identify channel
    ch_idx = np.where(chan_arr == chan_num)[0]
    time_ch = time_arr[ch_idx]

    # Find the first and last entry
    first_time = 0
    # np.min(time_ch).astype(np.int32)
    temp_last_time = np.max(time_ch).astype(np.int32)

    # floor this as last bin is always incomplete and so we discard photons.
    num_bins = np.floor((temp_last_time - first_time) / win_int)
    last_time = num_bins * win_int

    bins = np.linspace(first_time, last_time, int(num_bins) + 1)

    photons_in_bin, _ = np.histogram(time_ch, bins)

    # bins are valued as half their span.
    scale = bins[:-1] + (win_int / 2)

    # scale =  np.arange(0,time_ch.shape[0])

    return photons_in_bin, scale


def calc_coincidence_value(time_series1, time_series2):
    """calculates coincidence value

    Notes
    -----
    code is adopted from: https://github.com/dwaithe/FCS_point_correlator/blob/master/focuspoint/correlation_objects.py#L538"""
    N1 = np.bincount(np.array(time_series1).astype(np.int64))
    N2 = np.bincount(np.array(time_series2).astype(np.int64))

    n = max(N1.shape[0], N2.shape[0])
    NN1 = np.zeros(n)
    NN2 = np.zeros(n)
    NN1[:N1.shape[0]] = N1
    NN2[:N2.shape[0]] = N2
    N1 = NN1
    N2 = NN2

    CV = (np.sum(N1 * N2) / (np.sum(N1) * np.sum(N2))) * n
    return CV


def process_tcspc_data(chan_arr, dtime_arr, true_time_arr, verbose=True):
    """Process tcspc data from .ptu files and return a timetrace

    Takes micro and macro times from tcspc data and processes photon decay
    and time series data.

    Parameters
    ----------
    chan_arr : np.ndarray of integers
        Recording channel number for each recorded event
    dtime_arr : np.ndarray of integers
        Micro time of each event in ns (lifetime of each photon)
    true_time_arr : np.ndarray of integers
        Macro time of the whole recording in ns (absolute time when each
        photon arrived)
    verbose : bool
        if True, prints out information while processing

    Returns
    -------
    tcspc : dict of np.arrays
        if num_of_ch == 1 it contains
            tcspc['photon_decay_ch1'] : photon decay function (y)
            tcspc['decay_scale1'] : photon decay function (x)
            tcspc['time_series1'] : time series of trace in ms (y)
            tcspc['time_series_scale1'] : time series of trace in ms (x)
        if num_of_ch == 2 it contains the above + additionally
            tcspc['photon_decay_ch2'] : photon decay function channel 2 (y)
            tcspc['decay_scale2'] : photon decay function channel 2 (x)
            tcspc['time_series2'] : time series of trace in ms ch 2 (y)
            tcspc['time_series_scale2'] : time series of trace ch 2 (x)
    num_of_ch : integer
        number of detected channels

    Raises
    ------
    ValueError
        if number of channels is not 1 or 2

    Notes
    -----
    code is adopted from: https://github.com/dwaithe/FCS_point_correlator/blob/master/focuspoint/correlation_objects.py#L110"""
    # this is used for the photon decay curve
    win_int = 10
    # see below: since true_time_arr is in ns and we divide by 1e6, we get a
    # binning in ms
    photon_count_bin = 1
    tcspc = {}
    # Number of channels there are in the files.
    ch_present = np.sort(np.unique(np.array(chan_arr)))
    # if self.ext == 'pt3' or self.ext == 'ptu'or self.ext == 'pt2':
    # Minus 1 because not interested in channel 15.
    # numOfCH =  ch_present.__len__()-1
    num_of_ch = len(ch_present)

    # if first channel did not count any photons, do not analyze further
    first_ch_idx = np.where(chan_arr == 0)[0]
    first_ch_max = np.max(dtime_arr[first_ch_idx]).astype(np.int32)
    if first_ch_max == 0:
        num_of_ch -= 1
        ch_present += 1

    if verbose:
        print('\nnumber of channels which recorded photon counts: {}\n'.format(
            num_of_ch))

    # Calculates decay function for both channels.
    photon_decay_ch1, decay_scale1 = time2bin(time_arr=np.array(dtime_arr),
                                              chan_arr=np.array(chan_arr),
                                              chan_num=ch_present[0],
                                              win_int=win_int)
    # Time series of photon counts. For visualisation.
    time_series1, time_series_scale1 = time2bin(
        # timeseries in ms (without conversion high computational cost)
        time_arr=np.array(true_time_arr) / 1000000,
        chan_arr=np.array(chan_arr),
        chan_num=ch_present[0],
        win_int=photon_count_bin)
    #################################################
    # FUNCTIONAL, BUT OUTPUT NOT USED AT THE MOMENT #
    #################################################
    # unit = time_series_scale1[-1] / len(time_series_scale1)
    # Converts to counts per
    # kcount_Ch1 = np.average(
    #     time_series1)  # needed for crossAndAuto() in correlation_objects.py

    # unnormalised intensity count for int_time duration (the first moment)
    # raw_count = np.average(time_series1)
    # var_count = np.var(time_series1)

    # brightnessNandBCH and numberNandBCh used in calc_param_fcs() in
    # fitting_methods_SE.py, fitting_methods_PB.py
    # they are also printed in file by saveFile() used in correlation_gui.py
    # brightnessNandBCH0 = (((var_count - raw_count) / (raw_count)) /
    #                       (float(unit)))
    # if (var_count - raw_count) == 0:
    #     numberNandBCH0 = 0
    # else:
    #     numberNandBCH0 = (raw_count**2 / (var_count - raw_count))
    ################################################
    if num_of_ch == 1:
        tcspc['photon_decay_ch1'] = photon_decay_ch1
        tcspc['decay_scale1'] = decay_scale1
        tcspc['time_series1'] = time_series1
        tcspc['time_series_scale1'] = time_series_scale1

    elif num_of_ch == 2:
        photon_decay_ch2, decay_scale2 = time2bin(time_arr=np.array(dtime_arr),
                                                  chan_arr=np.array(chan_arr),
                                                  chan_num=ch_present[1],
                                                  win_int=win_int)
        time_series2, time_series_scale2 = time2bin(
            time_arr=np.array(true_time_arr) / 1000000,
            chan_arr=np.array(chan_arr),
            chan_num=ch_present[1],
            win_int=photon_count_bin)
        #################################################
        # FUNCTIONAL, BUT OUTPUT NOT USED AT THE MOMENT #
        #################################################
        # unit = time_series_scale2[-1] / len(time_series_scale2)
        # kcount_Ch2 = np.average(time_series2)
        # unnormalised intensity count for int_time duration (the first moment)
        # raw_count = np.average(time_series2)
        # var_count = np.var(time_series2)
        # brightnessNandBCH1 = (((var_count - raw_count) / (raw_count)) /
        #                       (float(unit)))
        # if (var_count - raw_count) == 0:
        #     numberNandBCH1 = 0
        # else:
        #     numberNandBCH1 = (raw_count**2 / (var_count - raw_count))
        # CV = calc_coincidence_value(time_series1=time_series1,
        #                             time_series2=time_series2)
        ################################################
        tcspc['photon_decay_ch1'] = photon_decay_ch1
        tcspc['decay_scale1'] = decay_scale1
        tcspc['time_series1'] = time_series1
        tcspc['time_series_scale1'] = time_series_scale1
        tcspc['photon_decay_ch2'] = photon_decay_ch2
        tcspc['decay_scale2'] = decay_scale2
        tcspc['time_series2'] = time_series2
        tcspc['time_series_scale2'] = time_series_scale2
    else:
        raise ValueError(
            '{} is an unsupported number of channels'.format(num_of_ch))

    return tcspc, num_of_ch


def import_from_ptu(path, file_delimiter=None, verbose=False):
    """Import .ptu files containing TCSPC data

    Import a directory of .ptu files containing TCSPC data, convert them to
    time traces and output them in one pandas DataFrame orderd columnwise. This
    pipeline is useful for feeding the data to a machine learning model for
    prediction.

    Parameters
    ----------
    path : str
        Folder which contains .ptu files with data
    file_delimiter : int or None
        If None, read in all files in path. If set to an integer, this is the
        maximum number of .ptu files which will be read in. This is useful for
        test purposes, since reading in even a single file takes quite some
        time.

    Returns
    -------
    ptu_exps_data : pandas DataFrame
        Contains time traces ordered columnwise
    ptu_exps_metadata : pandas DataFrame
        Contains ptu header metadata + num_of_ch from process_tcspc_data

    Raises
    ------
    ValueError
        If import_ptu_to_memory fails (probably the .ptu file is not
        compatible)
    ValueError
        If number of channels is greater 1 (see below)

    Notes
    -----
    - At the moment, the ptu helper functions can read in traces with up to 2
      channels, but I have not yet written a way to decide which photons to
      take, if there are two channels, so only one channel
      (tcspc['time_series1']) is supported.
    - parameters which could be easily implemented:
      + supply outputfilepath to import_ptu_to_memory() to export the header
        metadata to  txt files
    - functions which need some more thinking:
      + how to deal with multiple channels
      + implement different binning windows for the time traces (in function
        process_tcspc_data())
    """

    path = Path(
        '/beegfs/ye53nis/data/Pablo_structured_experiment/all_dirty_ptu/')
    files = [f for f in os.listdir(path) if f.endswith('.ptu')]

    ptu_exps_metadata = pd.DataFrame()
    ptu_exps_data = pd.DataFrame()

    for idx, file in enumerate(files):
        file = Path(file)
        path_and_file = path / file
        if verbose:
            print('{} of {}: {}'.format(idx, len(files), path_and_file))

        try:
            out, tag_data_list, _, __ = import_ptu_to_memory(
                inputfilepath=path_and_file, verbose=False)
        except ValueError:
            raise ValueError('A problem occurred while reading .ptu files')

        processed_tcspc, num_of_ch = process_tcspc_data(
            chan_arr=out['chanArr'],
            dtime_arr=out['dTimeArr'],
            true_time_arr=out['trueTimeArr'],
            verbose=False)

        if num_of_ch > 1:
            raise ValueError('Recordings with more than one input channel'
                             'are currently not supported.')

        ptu_exp_data = pd.DataFrame(processed_tcspc['time_series1'])
        ptu_exps_data = pd.concat([ptu_exps_data, ptu_exp_data],
                                  axis=1,
                                  ignore_index=True,
                                  sort=False)

        ptu_exp_numofch = pd.DataFrame([['Number of Channels', num_of_ch]])
        ptu_exp_metadata = pd.DataFrame(tag_data_list)
        ptu_exp_metadata = pd.concat([ptu_exp_metadata, ptu_exp_numofch],
                                     axis=0,
                                     ignore_index=True,
                                     sort=False)
        ptu_exps_metadata = pd.concat([ptu_exps_metadata, ptu_exp_metadata],
                                      axis=1,
                                      ignore_index=True,
                                      sort=False)

        if file_delimiter is not None and (idx + 1) > file_delimiter:
            break

    return ptu_exps_data, ptu_exps_metadata

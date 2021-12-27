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

import struct

import numpy as np


def pt2import(filepath):
    """The file import for the .pt3 file"""
    f = open(filepath, 'rb')
    Ident = f.read(16)
    FormatVersion = f.read(6)
    CreatorName = f.read(18)
    CreatorVersion = f.read(12)
    FileTime = f.read(18)
    CRLF = f.read(2)
    CommentField = f.read(256)
    Curves = struct.unpack('i', f.read(4))[0]
    BitsPerRecord = struct.unpack('i', f.read(4))[0]

    RoutingChannels = struct.unpack('i', f.read(4))[0]
    NumberOfBoards = struct.unpack('i', f.read(4))[0]
    ActiveCurve = struct.unpack('i', f.read(4))[0]
    MeasurementMode = struct.unpack('i', f.read(4))[0]
    SubMode = struct.unpack('i', f.read(4))[0]
    RangeNo = struct.unpack('i', f.read(4))[0]
    Offset = struct.unpack('i', f.read(4))[0]
    AcquisitionTime = struct.unpack('i', f.read(4))[0]
    StopAt = struct.unpack('i', f.read(4))[0]
    StopOnOvfl = struct.unpack('i', f.read(4))[0]
    Restart = struct.unpack('i', f.read(4))[0]
    DispLinLog = struct.unpack('i', f.read(4))[0]
    DispTimeFrom = struct.unpack('i', f.read(4))[0]
    DispTimeTo = struct.unpack('i', f.read(4))[0]
    DispCountFrom = struct.unpack('i', f.read(4))[0]
    DispCountTo = struct.unpack('i', f.read(4))[0]

    DispCurveMapTo = []
    DispCurveShow = []
    for i in range(0, 8):
        DispCurveMapTo.append(struct.unpack('i', f.read(4))[0])
        DispCurveShow.append(struct.unpack('i', f.read(4))[0])
    ParamStart = []
    ParamStep = []
    ParamEnd = []
    for i in range(0, 3):
        ParamStart.append(struct.unpack('i', f.read(4))[0])
        ParamStep.append(struct.unpack('i', f.read(4))[0])
        ParamEnd.append(struct.unpack('i', f.read(4))[0])

    RepeatMode = struct.unpack('i', f.read(4))[0]
    RepeatsPerCurve = struct.unpack('i', f.read(4))[0]
    RepeatTime = struct.unpack('i', f.read(4))[0]
    RepeatWait = struct.unpack('i', f.read(4))[0]
    ScriptName = f.read(20)

    # The next is a board specific header
    HardwareIdent = f.read(16)
    HardwareVersion = f.read(8)
    HardwareSerial = struct.unpack('i', f.read(4))[0]
    SyncDivider = struct.unpack('i', f.read(4))[0]
    CFDZeroCross0 = struct.unpack('i', f.read(4))[0]
    CFDLevel0 = struct.unpack('i', f.read(4))[0]
    CFDZeroCross1 = struct.unpack('i', f.read(4))[0]
    CFDLevel1 = struct.unpack('i', f.read(4))[0]
    Resolution = struct.unpack('f', f.read(4))[0]

    # below is new in format version 2.0
    RouterModelCode = struct.unpack('i', f.read(4))[0]
    RouterEnabled = struct.unpack('i', f.read(4))[0]

    # Router Ch1
    RtChan1_InputType = struct.unpack('i', f.read(4))[0]
    RtChan1_InputLevel = struct.unpack('i', f.read(4))[0]
    RtChan1_InputEdge = struct.unpack('i', f.read(4))[0]
    RtChan1_CFDPresent = struct.unpack('i', f.read(4))[0]
    RtChan1_CFDLevel = struct.unpack('i', f.read(4))[0]
    RtChan1_CFDZeroCross = struct.unpack('i', f.read(4))[0]

    # Router Ch2
    RtChan2_InputType = struct.unpack('i', f.read(4))[0]
    RtChan2_InputLevel = struct.unpack('i', f.read(4))[0]
    RtChan2_InputEdge = struct.unpack('i', f.read(4))[0]
    RtChan2_CFDPresent = struct.unpack('i', f.read(4))[0]
    RtChan2_CFDLevel = struct.unpack('i', f.read(4))[0]
    RtChan2_CFDZeroCross = struct.unpack('i', f.read(4))[0]

    # Router Ch3
    RtChan3_InputType = struct.unpack('i', f.read(4))[0]
    RtChan3_InputLevel = struct.unpack('i', f.read(4))[0]
    RtChan3_InputEdge = struct.unpack('i', f.read(4))[0]
    RtChan3_CFDPresent = struct.unpack('i', f.read(4))[0]
    RtChan3_CFDLevel = struct.unpack('i', f.read(4))[0]
    RtChan3_CFDZeroCross = struct.unpack('i', f.read(4))[0]

    # Router Ch4
    RtChan4_InputType = struct.unpack('i', f.read(4))[0]
    RtChan4_InputLevel = struct.unpack('i', f.read(4))[0]
    RtChan4_InputEdge = struct.unpack('i', f.read(4))[0]
    RtChan4_CFDPresent = struct.unpack('i', f.read(4))[0]
    RtChan4_CFDLevel = struct.unpack('i', f.read(4))[0]
    RtChan4_CFDZeroCross = struct.unpack('i', f.read(4))[0]

    # The next is a T3 mode specific header.
    ExtDevices = struct.unpack('i', f.read(4))[0]
    Reserved1 = struct.unpack('i', f.read(4))[0]
    Reserved2 = struct.unpack('i', f.read(4))[0]
    CntRate0 = struct.unpack('i', f.read(4))[0]
    CntRate1 = struct.unpack('i', f.read(4))[0]
    StopAfter = struct.unpack('i', f.read(4))[0]
    StopReason = struct.unpack('i', f.read(4))[0]
    Records = struct.unpack('i', f.read(4))[0]
    ImgHdrSize = struct.unpack('i', f.read(4))[0]

    # Special Header for imaging.
    if ImgHdrSize > 0:
        ImgHdr = struct.unpack('i', f.read(ImgHdrSize))[0]
    ofltime = 0
    cnt_0, cnt_1, cnt_2, cnt_3, cnt_4, cnt_Ofl, cnt_M, cnt_Err = (0, 0, 0, 0,
                                                                  0, 0, 0)
    RESOL = 4E-12  # 4ps
    WRAPAROUND = 210698240
    chanArr = [0] * Records
    trueTimeArr = [0] * Records
    dTimeArr = [0] * Records

    for b in range(0, Records):
        T2Record = struct.unpack('i', f.read(4))[0]
        T2time = T2Record & 268435455
        chan = ((T2Record >> 28) & 15)
        chanArr[b] = chan

        if chan == 0:
            cnt_0 += 1
        elif chan == 1:
            cnt_1 += 1
        elif chan == 2:
            cnt_2 += 1
        elif chan == 3:
            cnt_3 += 1
        elif chan == 4:
            cnt_4 += 1
        elif chan == 15:
            markers = T2Record & 15
            if markers == 0:
                ofltime = ofltime + WRAPAROUND
                cnt_Ofl += 1
            else:
                cnt_M += 1
        else:
            cnt_Err += 1
        time = T2time + ofltime
        trueTimeArr[b] = time * RESOL

    return (np.array(chanArr) + 1, np.array(trueTimeArr) * 1000000000,
            np.array(dTimeArr), Resolution)

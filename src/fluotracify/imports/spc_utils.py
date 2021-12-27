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

import numpy as np


def spc_file_import(file_path):
    f = open(file_path, 'rb')
    macro_time = float(int(bin(ord(f.read(1)))[2:].rjust(2, '0'), 2)) * 0.1

    int(bin(ord(f.read(1)))[2:].rjust(2, '0'), 2)
    int(bin(ord(f.read(1)))[2:].rjust(2, '0'), 2)
    bin(ord(f.read(1)))[2:].rjust(2, '0')

    overflow = 0
    count1 = 0
    count0 = 0
    chan_arr = []
    true_time_arr = []
    dtime_arr = []
    while True:
        byte = f.read(1)
        if byte.__len__() == 0:
            break
        byte0 = bin(ord(byte))[2:].rjust(8, '0')
        byte1 = bin(ord(f.read(1)))[2:].rjust(8, '0')
        byte2 = bin(ord(f.read(1)))[2:].rjust(8, '0')
        byte3 = bin(ord(f.read(1)))[2:].rjust(8, '0')

        INVALID = int(byte3[0])
        MTOV = int(byte3[1])
        GAP = int(byte3[2])

        if MTOV == 1:
            count0 += 1
            overflow += 4096
        if INVALID == 1:
            count1 += 1
        else:
            chan_arr.append(int(byte1[0:4], 2))
            true_time_arr.append(int(byte1[4:8] + byte0, 2) + overflow)
            dtime_arr.append(4095 - int(byte3[4:8] + byte2, 2))
            # file_out.write(str(int(byte1[4:8]+byte0,2)+overflow)+'\t'+str(
            # 4095 - int(byte3[4:8]+byte2,2))+'\t'+str(
            # int(byte1[0:4],2))+'\t'+str(byte3[0:4])+'\n')

    return (np.array(chan_arr), np.array(true_time_arr) * (macro_time),
            np.array(dtime_arr), None)

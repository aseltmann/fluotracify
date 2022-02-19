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


def asc_file_import(file_path):
    f = open(file_path, 'rb')
    count = 0
    chan_arr = []
    true_time_arr = []
    dtime_arr = []
    read_header = True
    for line in iter(f.readline, b''):
        count += 1
        if read_header is True:
            if line[0:5] == 'Macro':
                macro_time = float(line.split(':')[1].split(',')[0])
            if line[0:5] == 'Micro':
                micro_time = float(line.split(':')[1])
            # print line
            count += 1
            if line[0:18] == 'End of info header':
                read_header = False
                f.readline()  # Skips blank line.
                continue
        if read_header is False:
            # Main file reading loop.
            var = line.split(" ")
            true_time_arr.append(int(var[0]))
            dtime_arr.append(int(var[1]))
            chan_arr.append(int(var[3]))

    return (np.array(chan_arr), np.array(true_time_arr) * (macro_time),
            np.array(dtime_arr), micro_time)

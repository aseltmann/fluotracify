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

import csv

import numpy as np


def csvimport(filepath):
    """Function for importing time-tag data directly into FCS point software.
    """
    r_obj = csv.reader(open(filepath, 'r'))
    line_one = next(r_obj)
    if line_one.__len__() > 1:
        if float(line_one[1]) == 2:
            version = 2
        else:
            print('version not known:', line_one[1])
    if version == 2:
        type = str(next(r_obj)[1])
        if type == "pt uncorrelated":
            Resolution = float(r_obj.next()[1])
            chanArr = []
            trueTimeArr = []
            dTimeArr = []
            line = r_obj.next()
            while line[0] != 'end':
                chanArr.append(int(line[0]))
                trueTimeArr.append(float(line[1]))
                dTimeArr.append(int(line[2]))
                line = r_obj.next()
            return (np.array(chanArr), np.array(trueTimeArr),
                    np.array(dTimeArr), Resolution)
        else:
            print('type not recognised')
            return None, None, None, None

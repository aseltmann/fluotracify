import uuid

import numpy as np


def savetrace_csv(artifact,
                  path_and_file_name,
                  traces_array,
                  col_per_example,
                  foci_array,
                  foci_distance,
                  total_sim_time,
                  time_step,
                  nmol,
                  d_mol,
                  width,
                  height,
                  nclust=None,
                  d_clust=None):
    """save out a series of simulated fluorescence traces and labels indluding
    metadata of the simulations

    Parameters
    ----------
    artifact : {0, 1, 2, 3}
        0 = no artifact, 1 = bright clusters, 2 = detector dropout,
        3 = photobleaching. For 1 and 3, additional metadata is saved out.
        If 1 is chosen, the parameters nclust and d_clust have to be given.
    path_and_file_name : str
        Destination path and file name
    traces_array : np.array
        fluorescence traces and labels as columns (trace A, label A1, label A2,
        trace B, label B1, label B2, ...)
    col_per_example : int
        Number of columns per example, first column being a trace, and then
        one or multiple labels
    ...

    Returns
    -------
    Saves a .csv file
    """
    unique = uuid.uuid4()

    header = ''

    for idx, trace in enumerate(
            range(0, len(traces_array[0, :]), col_per_example)):
        header += 'trace{:0>3},'.format(idx + 1)
        for jdx in range(col_per_example - 1):
            header += 'label{:0>3}_{},'.format(idx + 1, jdx + 1)

    with open(path_and_file_name, 'w') as myFile:
        myFile.write('unique identifier,{}\n'.format(unique))
        myFile.write('path and file name,{}\n'.format(path_and_file_name))
        myFile.write(
            'FWHMs of excitation PSFs used in nm,{}\n'.format(foci_array))
        myFile.write(
            'Extent of simulated PSF (distance to center of Gaussian) in nm,{}\n'
            .format(foci_distance))
        myFile.write('total simulation time in ms,{}\n'.format(total_sim_time))
        myFile.write('time step in ms,{}\n'.format(time_step))
        myFile.write('number of fast molecules,{}\n'.format(nmol))
        myFile.write(
            'diffusion rate of molecules in micrometer^2 / s,{}\n'.format(
                d_mol))
        myFile.write('width of the simulation in nm,{}\n'.format(width))
        myFile.write('height of the simulation in nm,{}\n'.format(height))
        if artifact == 1:
            myFile.write('number of slow clusters,{}\n'.format(nclust))
            myFile.write(
                'diffusion rate of clusters in micrometer^2 / s,{}\n'.format(
                    d_clust))
        elif artifact == 3:
            myFile.write(
                'number of bleached molecules (50% immobile and 50% mobile),{}\n'.
                format(nmol * 2))
        # comments expects a str. Otherwise it printed a '# ' in first column
        # header and importing that to pandas made it an 'object' dtype which
        # uses a lot of memory
        np.savetxt(myFile,
                   traces_array,
                   delimiter=',',
                   header=header,
                   comments='')

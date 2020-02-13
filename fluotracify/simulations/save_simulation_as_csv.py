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

    for idx, _ in enumerate(traces_array[0, ::col_per_example], start=1):
        header += 'trace{:0>3},'.format(idx)
        for jdx in range(1, col_per_example):
            header += 'label{:0>3}_{},'.format(idx, jdx)
    # Remove trailing comma
    header = header.strip(',')

    # TODO: include path handling with pathlib to make code work independent
    # of OS
    with open(path_and_file_name, 'w') as my_file:
        my_file.write('unique identifier,{}\n'.format(unique))
        my_file.write('path and file name,{}\n'.format(path_and_file_name))
        my_file.write('FWHMs of excitation PSFs used in nm,'
                      '{}\n'.format(foci_array))
        my_file.write('Extent of simulated PSF (distance to center of '
                      'Gaussian) in nm,{}\n'.format(foci_distance))
        my_file.write('total simulation time in ms,'
                      '{}\n'.format(total_sim_time))
        my_file.write('time step in ms,{}\n'.format(time_step))
        my_file.write('number of fast molecules,{}\n'.format(nmol))
        my_file.write('diffusion rate of molecules in micrometer^2 / s,'
                      '{}\n'.format(d_mol))
        my_file.write('width of the simulation in nm,{}\n'.format(width))
        my_file.write('height of the simulation in nm,{}\n'.format(height))
        if artifact == 1:
            my_file.write('number of slow clusters,{}\n'.format(nclust))
            my_file.write('diffusion rate of clusters in micrometer^2 / s,'
                          '{}\n'.format(d_clust))
        elif artifact == 3:
            my_file.write('number of bleached molecules (50% immobile and '
                          '50% mobile),{}\n'.format(nmol * 2))
        # comments expects a str. Otherwise it printed a '# ' in first column
        # header and importing that to pandas made it an 'object' dtype which
        # uses a lot of memory
        np.savetxt(my_file,
                   traces_array,
                   delimiter=',',
                   header=header,
                   comments='')

#!/usr/bin/env python3
import h5py
import numpy as np

import logging
logger = logging.getLogger(__name__)

def write_energies_to_h5(path, parameters, energies, eigenvectors=None, metadata=None):
    """
    Write spectral data (parameters, energies, and optionally eigenvectors)
    to an HDF5 file.

    Parameters
    ----------
    path : str
        Destination filepath. 
    parameters : array-like
        1D sequence of parameter values (e.g., L values). Will be stored as
        an HDF5 dataset named "parameters".
    energies : array-like
        2D array of shape (num_parameters, k_num) containing k_num eigenvalues
        for each parameter value. Stored as dataset "energies".
    eigenvectors : array-like, optional
        Optional 3D array containing eigenvectors of shape 
        (num_parameters, k_num, vector_dimension). If provided, stored as
        dataset "eigenvectors".
    metadata : dict, optional
        Dictionary of lightweight metadata to attach as HDF5 attributes.
        All values must be convertible to strings.

    Notes
    -----
    - All arrays are written using `f.create_dataset(...)`.
    - Metadata entries are stored as file-level attributes via `f.attrs[...]`.

    Returns
    -------
    None
        The function writes to disk and returns nothing.
    """
    with h5py.File(path, "w") as f:
        # save eigenvalues and eigenvectors if requested
        f.create_dataset("parameters", data=parameters)
        f.create_dataset("energies", data=energies)
        if eigenvectors is not None:
            eigenvectors = np.asarray(eigenvectors)
            f.create_dataset("eigenvectors", data=eigenvectors)

        # add metadata
        metadata = metadata or {}
        for key, val in metadata.items():
            f.attrs[key] = val

def write_energies_to_dat(path, parameters, energies, metadata=None):
    """
    Write a set of (parameter, energy) pairs to a .dat file with optional metadata.

    The output file begins with metadata entries (one per line) prefixed by '#',
    followed by a (1 + k_num)-column table containing the parameter values and the
    corresponding k_num energies per parameter value.

    Parameters
    ----------
    path : str or pathlib.Path
        The output file path. If the file exists, it will be overwritten.
    parameters : array_like
        1D array of parameter values (e.g., couplings, masses, etc.).
        Must be the same length as `energies`.
    energies : array_like
        2D array of energy values corresponding to `parameters`; shape (len(`parameters`), k_num)
        for k_num energies per parameter value.
    metadata : dict, optional
        A mapping of string keys to values. Each entry is written as a header line
        in the format ``# key : value`` before the data table. Defaults to an empty
        dictionary.

    Notes
    -----
    The data are written using ``numpy.savetxt`` with tab delimiters and a fixed
    ``"%.8f"`` floating-point format for consistency across runs.
    """
    with open(path, "w") as f:
        # add metadata
        metadata = metadata or {}
        for key, val in metadata.items():
            f.write(f"# {key} : {val}\n")
        # add parameters, energies columns
        np.savetxt(f, np.column_stack([parameters, energies]), fmt="%.8f", delimiter="\t")

def load_energies_from_h5(path):
    """
    Load spectral data (parameters, energies, optional eigenvectors, and metadata)
    from an HDF5 (.h5) file.

    This function expects an HDF5 file containing at least the datasets
    ``"parameters"`` and ``"energies"``. If the optional dataset
    ``"eigenvectors"`` is present, it will also be loaded. Any file-level
    attributes are returned as metadata.

    Parameters
    ----------
    path : str
        Path to the HDF5 file to read.

    Returns
    -------
    parameters : numpy.ndarray
        Array containing the parameter values stored under the dataset
        ``"parameters"``. No shape is enforced; it is returned exactly as stored.
    energies : numpy.ndarray
        Array containing spectral energy data from the dataset ``"energies"``.
        Returned exactly as stored.
    eigenvectors : numpy.ndarray or None
        If the file contains a dataset named ``"eigenvectors"``, this is returned
        as a NumPy array. If not present, ``None`` is returned.
    metadata : dict
        Dictionary of metadata read from the file's HDF5 attributes (``f.attrs``).
        Keys are strings and values are whatever types were originally stored.

    Raises
    ------
    RuntimeError
        If either the ``"parameters"`` or ``"energies"`` datasets are missing.
    
    Notes
    -----
    - This function does not enforce any shape or type constraints on the
      returned datasets. It simply reads whatever is present in the file.
      Consistency checks should be handled by the calling code or by the
      corresponding writer function.
    - The returned metadata is a shallow copy of the file attributes and may
      contain strings, numbers, or NumPy scalar types depending on the writer.
    """
    with h5py.File(path, "r") as f:
        try:
            parameters = f["parameters"][:]
            energies = f["energies"][:]
        except KeyError as e:
            raise RuntimeError(f"{path} is not formatted correctly. It must be a .h5 file with datasets 'parameters' and 'energies'")
        
        # eigenvectors may or may not exist
        eigenvectors = f.get("eigenvectors")
        if eigenvectors is not None:
            eigenvectors = eigenvectors[:]

        # validate data to ensure compatibility with program
        parameters, energies, eigenvectors = _validate_eigenpair_data(parameters, energies, eigenvectors)

        # load metadata
        metadata = dict(f.attrs)

        return parameters, energies, eigenvectors, metadata

def load_energies_from_dat(path):
    data = np.loadtxt(path, delimiter="\t", comments="#")
    parameters = data[:, 0]
    energies = data[:, 1:]
    
    # validate data to ensure compatibility with program
    parameters, energies, _ = _validate_eigenpair_data(parameters, energies, None)

    # load metadata
    metadata = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                line = line[1:].strip()
                if " : " in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip() 
    return parameters, energies, metadata
 
def _validate_eigenpair_data(parameters, energies, eigenvectors):
    logger.debug(f"Validating parameters, energies, and eigenvectors from I/O.")
    parameters, energies = np.atleast_1d(np.asarray(parameters)), np.atleast_1d(np.asarray(energies))
    
    # handle parameters
    if parameters.ndim > 1: 
        raise ValueError(f"parameters can't be more than 1d, got {parameters.ndim}.")

    # handle energies
    if parameters.shape[0] != energies.shape[0]:
        raise ValueError(f"parameters and energies need to have the same 1st dimension, got "
                         f"{parameters.shape[0]} vs {energies.shape[0]}.")
    if energies.ndim == 1:
        logger.debug(f"energies has shape {energies.shape}. Broadcasting to ({len(energies)}, 1).")
        energies = energies[:, None]
    elif energies.ndim > 2:
        raise ValueError(f"energies can't be more than 2d, got {energies.ndim}.") 

    # handle eigenvectors
    if eigenvectors is not None:
        eigenvectors = np.atleast_1d(np.asarray(eigenvectors))
        if eigenvectors.ndim == 1:
            # interpret as one vector given for one parameter
            logger.debug(f"eigenvectors has shape {eigenvectors.shape}, broadcasting to (1, 1, {len(eigenvectors)}).")
            eigenvectors = eigenvectors[None, None, :]
        elif eigenvectors.ndim == 2:
            # interpret as one vector for multiple parameters
            logger.debug(f"eigenvectors has shape {eigenvectors.shape}, broadcasting to ({eigenvectors.shape[0]}, 1, {eigenvectors.shape[1]}).")
            eigenvectors = eigenvectors[:, None, :]
        elif eigenvectors.ndim > 3:
            raise ValueError(f"eigenvectors can't be more than 3d, got {eigenvectors.ndim}.")
        if eigenvectors.shape[:2] != energies.shape:
            raise ValueError(f"energies and eigenvectors must have the same shape along the first 2 dimensions, got "
                             f"{energies.shape} vs {eigenvectors.shape[1:]}") 

    logger.debug(f"Validated I/O parameters, energies, and eigenvectors to shape {parameters.shape}, {energies.shape}, {eigenvectors.shape if eigenvectors is not None else None}.")
    return parameters, energies, eigenvectors

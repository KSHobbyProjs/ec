"""
ec.py

Defines a class for eigenvector continuation (EC) predictions built on a BaseModel.

This module provides a reusable interface for eigenvector continuation computations where
the Hamiltonian depends on an external parameter L. The EC class handles sampling, projecting, 
dilating, and prediction of eigenvalues and eigenvectors.

Classes
-------
EC
    Class that performs eigenvector continuation for BaseModel subclasses.
"""

import numpy as np
import scipy.sparse as ss
import scipy.ndimage as sn
import scipy

import logging
logger = logging.getLogger(__name__)

class EC: 
    """
    Class for handling eigenvector continuation computations.

    This class defines common methods to sample and predict eigenvalues and eigenvectors
    for Hamiltonians H(L) acting on an N-site lattice.

    Attributes 
    ----------
    _base_coods : dict
        Dictionary storing lattice coordinates for different lattice sizes N.
        Automatically populated upon calling `_get_base_coods()`.
    _model : BaseModel
        Model used for sampling (must subclass `BaseModel`).
    _sample_vectors : ndarray
        Sampled eigenvectors at various parameter values L.
        Automatically populated upon calling `sample()`.
    _sample_Ls : ndarray
        Parameter values where the Hamiltonian H(L) is sampled.
        Automatically populated upon calling `sample()`.
    _S : ndarray
        Overlap matrix between sample eigenvectors.
        Automatically populated upon calling `sample()`.
    """

    _base_coods = {}

    def __init__(self, model):
        """
        Initialize the eigenvector continuation setup with a specified model.

        Parameters
        ----------
        model : BaseModel
            Model used for sampling.
        """

        self._model = model
    
        self._sample_Ls = None
        self._sample_energies = None
        self._sample_vectors = None
        self._S = None

    def sample_from_given_data(self, sample_Ls, sample_energies, sample_eigenvectors):
        """
        Store given eigenpairs of H(L) for each parameter L in `sample_Ls`.

        Parameters
        ----------
        sample_Ls : ndarray
            List of parameters to sample the Hamiltonian at.
        sample_energies : ndarray
            List of k_num energies at each parameter in `sample_Ls`. 
            Shape (len(`sample_Ls`), k_num).
        sample_eigenvectors : ndarray
            List of k_num eigenvectors at each parameter in `sample_Ls`.
            Shape (len(`sample_Ls`), k_num, vector_dimension).

        Returns
        -------
            None.

        Notes
        -------
        This method populates the following attributes:
        
        - `self._sample_Ls` : ndarray
            Array of sampled L values.
        - `self._sample_energies` : ndarray
            Array of sampled energies.
        - `self._sample_vectors` : ndarray
            Array of sampled eigenvectors, equal to `eigvecs`.
        - `self._S` : ndarray
            Overlap matrix between sampled eigenvectors, shape (m, m),
            where m = len(eigvecs).
        """
        self._sample_Ls = sample_Ls
        self._sample_energies = sample_energies
        eigvecs = sample_eigenvectors.reshape(-1, sample_eigenvectors.shape[2])  # flatten eigvecs
        self._sample_vectors = eigvecs 

        self._S = eigvecs.conj() @ eigvecs.T
        if np.linalg.cond(self._S) > 1e10:
            logger.warning("Ill-conditioned overlap matrix S; check sample vectors.") # warn user if S is near singular
        

    def sample_from_model(self, sample_Ls, k_num=1):
        """
        Compute and store the lowest `k_num` eigenpairs of H(L) for each parameter L in `sample_Ls`.

        Parameters
        ----------
        sample_Ls : float or array_like
            Parameter (or list of parameters) to sample the Hamiltonian at.
        k_num : int, optional
            Number of lowest eigenpairs to compute. Default is 1.

        Returns
        -------
        eigvecs : ndarray
            Sampled eigenvectors, shape (len(sample_Ls) * k_num, n).

        Notes
        -------
        This method populates the following attributes:
        
        - `self._sample_Ls` : ndarray
            Array of sampled L values.
        - `self._sample_vectors` : ndarray
            Array of sampled eigenvectors, equal to `eigvecs`.
        - `self._S` : ndarray
            Overlap matrix between sampled eigenvectors, shape (m, m),
            where m = len(eigvecs).
        """

        sample_Ls = np.atleast_1d(sample_Ls)
        energies, eigvecs = self._model.get_eigenvectors(sample_Ls, k_num)
        eigvecs = eigvecs.reshape(-1, eigvecs.shape[2])
        self._sample_energies = energies
        self._sample_vectors = eigvecs
        self._S = eigvecs.conj() @ eigvecs.T
        if np.linalg.cond(self._S) > 1e10: 
            logger.warning("Ill-conditioned overlap matrix S; check sample vectors.") # warn user if S is near singular
        self._sample_Ls = sample_Ls
        return eigvecs

    def ec_predict(self, target_Ls, k_num=None, dilate=False):
        """
        Predict lowest `k_num` eigenpairs at each target L value in `target_Ls` using EC projection.

        Parameters
        ----------
        target_Ls : float or array_like
            Parameter (or list of parameters) to estimate eigenpairs at.
        k_num : int, optional
            Number of lowest eigenpairs to predict. Default is all.
        dilate : bool, optional
            If True, dilates the sampled eigenvectors to match the target `target_Ls`
            before computing the projected matrices. This option only makes sense when
            the target L values correspond to a length or volume scale; it should be left as False for 
            dimensionless coupling parameters. Default is False.

        Returns
        -------
        eigenvalues : ndarray
            Predicted eigenvalues at `target_Ls`. Shape (len(target_Ls), k_num).
        eigenvectors : ndarray
            Predicted eigenvectors at `target_Ls`. Shape (len(target_Ls), k_num, n).

        Raises
        ------
        RuntimeError
            If `sample()` has not been called prior to prediction.
        """
        if dilate:
            logger.warning(f"dilate only set up for 3d (flattened) eigenvectors. Do not use if eigenvector is not 3d.")
        
        if self._sample_vectors is None or self._S is None or self._sample_Ls is None:
            raise RuntimeError("No sampled vectors found. Run `sample()` first.")
        
        Ls = np.atleast_1d(target_Ls)
        
        if k_num is None: k_num = self._sample_vectors.shape[0]
        eigenvalues = np.zeros((len(Ls), k_num), dtype=np.float64)
        eigenvectors = np.zeros((len(Ls), k_num, self._model.construct_H(Ls[0]).shape[0]), dtype=np.complex128)
        for i, L in enumerate(Ls):
            if dilate:
                reshaped_sample_vectors = self._sample_vectors.reshape(len(self._sample_Ls), -1, self._sample_vectors.shape[1])
                sample_vectors, S = EC.get_dilated_basis(self._sample_Ls, reshaped_sample_vectors, L)
            else:
                sample_vectors, S = self._sample_vectors, self._S
            H = self._model.construct_H(L)
            H_proj = sample_vectors.conj() @ H @ sample_vectors.T
            eigval, eigvec = scipy.linalg.eigh(H_proj, S)
            eigenvalues[i] = eigval[:k_num] # eigenvalues are already sorted in ascending order, so no need for argsort
            eigenvectors[i] = eigvec[:, :k_num].T @ sample_vectors # eigenvectors have to be dotted with sample vectors since they're coordinate vectors 
        return eigenvalues, eigenvectors


    def run_ec(self, sample_Ls, target_Ls, k_num_sample=1, k_num_predict=None, dilate=False):
        """
        Wrapper for sampling and predicting eigenpairs at target L values.

        This method calls `sample()` to compute the basis at `sample_Ls` and then calls
        `ec_predict()` to estimate eigenpairs at `target_Ls`. The attributes of the class
        are not updated after this method returns.
        
        Parameters
        ----------
        sample_Ls : float or array_like
            L values at which to sample the Hamiltonian.
        target_Ls : float or array_like
            L values at which to predict eigenpairs.
        k_num_sample : int, optional
            Number of lowest eigenpairs to sample. Default is 1.
        k_num_predict : int, optional
            Number of lowest eigenpairs to predict. Default is None (which returns all eigenvalues available).
        dilate : bool, optional
            If True, dilates the sampled eigenvectors to match the target `target_Ls`
            before computing the projected matrices. This option only makes sense when
            the target L values correspond to a length or volume scale; it should be left as False for 
            dimensionless coupling parameters. Default is False.

        Returns
        -------
        eigenvalues : ndarray
            Predicted eigenvalues at `target_Ls`, exactly as returned by `ec_predict()`.
        eigenvectors : ndarray
            Predicted eigenvectors, exactly as returned by `ec_predict()`.
        """

        temp_vecs = self._sample_vectors
        temp_S = self._S

        self.sample_from_model(sample_Ls, k_num_sample)
        eigenvalues, eigenvectors = self.ec_predict(target_Ls, k_num_predict, dilate)

        self._sample_vectors = temp_vecs
        self._S = temp_S
        return eigenvalues, eigenvectors

    def get_state(self):
        state = {
                "model" : self._model,
                "sample_Ls" : self._sample_Ls,
                "sample_energies" : self._sample_energies,
                "sample_vectors" : self._sample_vectors,
                "S" : self._S
                }
        return state

    def set_state(self, state):
        self._model = state["model"]
        self._sample_Ls = state["sample_Ls"]
        self._sample_energies = state["sample_energies"]
        self._sample_vectors = state["sample_vectors"]
        self._S = state["S"]

    @classmethod
    def _get_base_coods(cls, N):
        """
        Compute, store, and return the lattice coordinates for a given lattice size.

        Parameters
        ----------
        N : int
            Number of lattice points along each dimension.

        Returns
        -------
        base_coods : ndarray 
            Meshgrid of lattice coordinates with shape (3, N, N, N).

        Notes
        -------
        Automatically populates the class attribute `_base_coods` for the given N. 

        """

        if N in cls._base_coods:
            return cls._base_coods[N]

        xs = (np.arange(N) - N / 2).astype(np.float64)
        xs, ys, zs = np.meshgrid(xs, xs, xs, indexing='ij')
        base_coods = np.stack([xs, ys, zs], axis=0)
        cls._base_coods[N] = base_coods
        return base_coods
   
    @staticmethod
    def get_dilated_basis(sample_Ls, sample_vectors, target_L):
        """
        Dilate a set of sampled eigenvectors to a target system size.

        This method rescales the sampled eigenvectors to the target L value `target_L`
        (physical length or volume scale) and computes their overlap matrix.

        Parameters
        ----------
        sample_Ls : ndarray
            L values at which the eigenvectors were originally sampled. Shape (len(sample_Ls),)
        sample_vectors : ndarray
            Sample eigenvectors, shape (len(sample_Ls), k_num, n), where k_num is the number of 
            eigenpairs sampled per L and n is the Hamiltonian dimension.
        target_L : float
            Target L value to which the eigenvectors are dilated.

        Returns 
        -------
        dilated_basis : ndarray
            Dilated eigenvectors, shape (len(sample_Ls) * k_num, n).
        S : ndarray
            Overlap matrix of the dilated basis, shape (m, m) where m=len(sample_Ls).

        Notes
        -------
        Calls `dilate()`, which in turn calls `_get_base_coods()`. The class attribute `_base_coods`
        is updated automatically.
        """

        Ls_length, k_num, N3 = sample_vectors.shape
        N = round(N3**(1/3))

        if Ls_length != len(sample_Ls):
            raise RuntimeError("the first axis of sample_vectors needs to match the length of sample_Ls in `get_dilated_basis(sample_Ls, sample_vectors, target_L)`")

        dilated_basis = np.empty_like(sample_vectors)
        for i, sample_L in enumerate(sample_Ls):
            dilated_basis[i] = EC.dilate(sample_L, target_L, sample_vectors[i])

        # reflatten
        dilated_basis = dilated_basis.reshape(-1, dilated_basis.shape[2])
        S = dilated_basis.conj() @ dilated_basis.T
        return dilated_basis, S
    
    # dilate a state ket psi into a new volume & renormalize
    # given target volume Lprime, old volume L, and wavefunction psi
    @staticmethod
    def dilate(L, L_target, psi):
        """ 
        Dilate a wavefunction / eigenvector from one system size to another.
        
        The wavefunction `psi` is interpolated to the target volume `L_target` and renormalized.
        Only physically meaningful when L represents a length or volume scale.

        Parameters
        ----------
        L : float
            Original system size of `psi`.
        L_target : float
            Target system size for interpolation.
        psi : ndarray
            Eigenvector(s) at system size `L`. Shape (n,) for one vector or (k_num, n) for k_num vectors.

        Returns
        -------
        psi_dilated : ndarray
            Wavefunction(s) dilated to `L_target`. Shape matches input: (n,) or (k_num, n).

        Notes
        -------
        Calls `_get_base_coods()`, updating `_base_coods` if necessary.
        """
        if psi.ndim == 1:
            psi = psi[None, :] # make it (1, N^3)

        k_num, N3 = psi.shape
        N = round(N3**(1/3))
        psi_3d = psi.reshape(k_num, N, N, N)

        # define dilation factor
        s = L / L_target

        # physical coordinates are dilated, not the indices, so create physical cood arrays
        base_coods = EC._get_base_coods(N) 

        # dilate physical coods by L / L'
        dilated_coods = s * base_coods

        # map the physical coods from [-N/2,N/2) back into [0, N) indices
        dilated_coods += N / 2 

        # interpolate psi to psi*
        psi_dilated = np.empty_like(psi)
        for k in range(k_num):
            psi_dilated_3d = s**3/2 * sn.map_coordinates(psi_3d[k], dilated_coods, mode='wrap', order=3)
            psi_dilated[k] = psi_dilated_3d.reshape(-1)

        # reflatten and normalize
        psi_dilated = psi_dilated.reshape(k_num, -1)
        norms = np.linalg.norm(psi_dilated, axis=1, keepdims=True)
        psi_dilated /= norms
        
        # squeeze if k_num=1
        if k_num == 1:
            return psi_dilated[0]
        return psi_dilated

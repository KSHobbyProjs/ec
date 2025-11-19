import numpy as np
import scipy.sparse as ss
from . import base_model

class Ising(base_model.BaseModel):
    def __init__(self, N, J=1.0):
        super().__init__(N)
        self._J = J

    # construct the 3 Pauli matrices at every point i given the number of lattice points N
    @staticmethod
    def _construct_Pauli_matrices(N):
        sigmas = []
        for i in range(N):
            sigmas_at_site_i = []

            # construct z-component of Pauli matrices
            zs = (-1)**( (np.arange(2**N) >> i) & 1 )
            sigmas_at_site_i.append(ss.diags(zs, format='csr', dtype=np.complex128))

            # construct x-component of Pauli matrices
            data = np.ones(2**N, dtype=np.complex128)
            rows = np.arange(2**N)
            cols = rows ^ (1 << i)
            sigmas_at_site_i.append( ss.coo_matrix((data, (rows, cols)), shape=(2**N, 2**N)).tocsr() )

            # construct y-component of Pauli matrices
            rows = np.arange(2**N)
            data = (-1)**( (rows >> i) & 1 ) * 1j
            cols = rows ^ (1 << i)
            sigmas_at_site_i.append( ss.coo_matrix((data, (rows, cols)), shape=(2**N, 2**N)).tocsr() )

            # add [sigma_z,i, sigma_x,i, sigma_y,i] to sigmas
            sigmas.append(sigmas_at_site_i)
        return sigmas

    # construct the Ising Hamiltonian in 1D
    def construct_H(self, g):
        sigmas = Ising._construct_Pauli_matrices(self._N)
        H1 = ss.csr_matrix((2**self._N, 2**self._N), dtype=np.complex128)
        H2 = ss.csr_matrix((2**self._N, 2**self._N), dtype=np.complex128)
        for i, sigma in enumerate(sigmas):
            H1 += sigma[0] @ sigmas[(i + 1) % self._N][0]
            H2 += sigma[1]
        return -self._J * (H1 + g * H2)


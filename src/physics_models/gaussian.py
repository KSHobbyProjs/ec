import numpy as np
import scipy.sparse as ss
from . import base_model

class Gaussian3d(base_model.BaseModel):
    def __init__(self, N, V0=-4.0, R=2.0):
        super().__init__(N)
        self._R = R
        self._V0 = V0

    # grab relative positions from index i
    @staticmethod
    def _get_relative_coods(i, N):
        if np.any(i < 0) or np.any(i > N**3 - 1):
            raise ValueError(f"i must be between 0 and {N**3 - 1}")
        x = (i % N) - N / 2
        y = ((i % N**2) // N) - N / 2
        z = (i // N**2) - N / 2
        return x, y, z

    # calculate relative distance given index i
    @staticmethod
    def _get_distance(i, N):
        x, y, z = Gaussian3d._get_relative_coods(i, N)
        return np.sqrt(x**2 + y**2 + z**2)

    def construct_H(self, L):
        a = L / self._N
        N_tot = self._N**3
        indices = np.arange(N_tot, dtype=np.int32)

        # construct T
        # initialize data, rows, and cols arrays for coo matrix form
        total_entries = N_tot * 7                           # N_tot entries for the diagonal and each neighbor
        data = np.empty(total_entries, dtype=np.complex128)
        rows = np.empty(total_entries, dtype=np.int32)
        cols = np.empty(total_entries, dtype=np.int32)
        
        # fill diagonal 
        data[:N_tot] = -6 * np.ones(N_tot)
        rows[:N_tot] = indices
        cols[:N_tot] = indices

        # fill neighbors
        offset = N_tot
        x, y, z = Gaussian3d._get_relative_coods(indices, self._N)
        neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for dx, dy, dz in neighbors:
            nx = ((x + dx) + self._N // 2) % self._N - self._N // 2                                                   # x-neighbor physical cood
            ny = ((y + dy) + self._N // 2) % self._N - self._N // 2                                                   # y-neighbor physical cood
            nz = ((z + dz) + self._N // 2) % self._N - self._N // 2                                                   # z-neighbor physical cood
            neighbor_indices = (nx + self._N // 2) + self._N * (ny + self._N // 2) + self._N**2 * (nz + self._N // 2) # index of neighbor
            rows[offset:offset + N_tot] = indices
            cols[offset:offset + N_tot] = neighbor_indices
            data[offset:offset + N_tot] = np.ones(N_tot)
            offset += N_tot

        T = -1 / a**2 * ss.coo_matrix((data, (rows, cols)), shape=(N_tot, N_tot), dtype=np.complex128).tocsr()

        # construct V
        distances = Gaussian3d._get_distance(indices, self._N)
        vs = self._V0 * np.exp(-(distances * a)**2 / self._R**2)
        V = ss.diags(vs, format='csr', dtype=np.complex128)

        H = T + V
        return H


class Gaussian1d(base_model.BaseModel):
    def __init__(self, N, V0=-4.0, R=2.0):
        super().__init__(N)
        self._R = R
        self._V0 = V0

    def construct_H(self, L):
        a = L / self._N
        indices = np.arange(self._N)

        # construct T
        # make diagonal portion
        diags = np.ones(self._N)
        off_diags = diags[1:]
        T = ss.diags([off_diags, -2 * diags, off_diags], [-1, 0, 1], format='lil')
        # fill corners because of BCs
        T[0, self._N - 1] = 1
        T[self._N - 1, 0] = 1
        # multiply by lattice spacing factor and format as csr
        T = (-1 / a**2 * T).tocsr().astype(np.complex128)

        # construct V
        # sum over nearest images if want V(x+L) = V(x) (V(L/2 + 1) = V(-L/2)), 
        # but this condition is nearly met if R << L since V(x + L) approx V(x) 
        # near the edges. In general, V(x) = sum_n V0 exp(-(x_i - nL)**2 / R**2)
        # is needed to ensure wrapped V(x), but you can see that, if R << L,
        # terms at moderate n are already suppressed
        distances = np.abs(indices - self._N // 2)
        vs = self._V0 * np.exp(-(distances * a)**2 / self._R**2)
        V = ss.diags(vs, format='csr', dtype=np.complex128)

        H = T + V
        return H

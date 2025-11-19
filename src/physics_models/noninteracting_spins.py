import numpy as np
from . import base_model

class NoninteractingSpins(base_model.BaseModel):
    def __init__(self, N):
        super().__init__(N)

    def construct_H(self, c):
        pauli_z = np.array([[1, 0], [0, -1]])
        pauli_x = np.array([[0, 1], [1, 0]])
        H = 0
        for i in range(self._N):
            H += (pauli_z + c * pauli_x)
        H = 1 / (2 * self._N) * H
        return H.astype(np.complex128)


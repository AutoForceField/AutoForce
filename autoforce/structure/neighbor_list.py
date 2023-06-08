import abc

import numpy as np

from .structure import Structure


class NeighborList(Structure, abc.ABC):
    # Since a neighbor list is associated with a structure,
    # it should inherit from the structure class.
    @abc.abstractmethod
    def get_cutoff(self) -> float:
        ...

    @abc.abstractmethod
    def get_neighbors_indices(self, index: int) -> np.ndarray:  # [:] int array
        ...

    @abc.abstractmethod
    def get_neighbors_offsets(
        self, index: int
    ) -> np.ndarray:  # [:, 3] int array
        ...

    # Drived methods
    def get_neighbors_types(self, index: int) -> np.ndarray:  # [:] int array
        types = self.get_types()
        neighbors = self.get_neighbors_indices(index)
        return types[neighbors]

    def get_neighbors_displacements(
        self, index: int
    ) -> np.ndarray:  # [:, 3] float array
        offsets = self.get_neighbors_offsets(index)
        cell = self.get_cell()
        shifts = (cell.T @ offsets.T).T
        # Or: shifts = (cell * offsets[:, None, :]).sum(axis=1)
        neighbors = self.get_neighbors_indices(index)
        displacements = (
            self.get_positions()[neighbors]
            - self.get_positions()[index]
            + shifts
        )
        return displacements

    def get_available_indices(self) -> np.ndarray:
        return np.arange(self.get_positions().shape[0])

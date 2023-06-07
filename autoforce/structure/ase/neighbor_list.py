import ase
import numpy as np
from ase.neighborlist import NeighborList as _ase_NeighborList

from ..neighbor_list import NeighborList
from .structure import ASEStructureMixin


class ASENeighborListMixin:
    _nl: _ase_NeighborList

    def get_neighbors(self, index: int) -> np.ndarray:  # 1D int array
        neighbors: np.ndarray
        neighbors, offsets = self._nl.get_neighbors(index)
        return neighbors

    def get_neighbors_offsets(self, index: int) -> np.ndarray:  # 2D int array
        neighbors, offsets = self._nl.get_neighbors(index)
        return offsets


class ASENeighborList(ASENeighborListMixin, ASEStructureMixin, NeighborList):
    def __init__(self, atoms: ase.Atoms, cutoff: float) -> None:
        self._atoms = atoms
        self._nl = _ase_NeighborList(
            cutoffs=[cutoff / 2] * len(self._atoms),
            skin=0.0,
            sorted=False,
            self_interaction=False,
            bothways=True,
        )
        self._cutoff = cutoff
        self._nl.update(self._atoms)

    def get_cutoff(self) -> float:
        return self._nl.cutoff

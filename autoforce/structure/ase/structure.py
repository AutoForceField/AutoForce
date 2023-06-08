import ase
import numpy as np

from ..structure import Structure


class ASEStructureMixin:
    _atoms: ase.Atoms

    def get_types(self) -> np.ndarray:
        return self._atoms.get_atomic_numbers()

    def get_positions(self) -> np.ndarray:
        return self._atoms.get_positions()

    def get_cell(self) -> np.ndarray:
        return self._atoms.get_cell()

    def get_pbc(self) -> tuple[bool, bool, bool]:
        return self._atoms.get_pbc()


class ASEStructure(ASEStructureMixin, Structure):
    def __init__(self, atoms: ase.Atoms):
        self._atoms = atoms

"""
Structure and MPI-parallelism interface.

"""
from __future__ import annotations

import abc

import ase
import numpy as np

from autoforce.mpi import Distributed


class Structure(abc.ABC):
    @abc.abstractmethod
    def get_chemical_symbols(self) -> tuple[str, ...]:
        ...

    @abc.abstractmethod
    def get_positions(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_cell(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_pbc(self) -> tuple[bool, bool, bool]:
        ...

    # Derived properties
    def get_atomic_numbers(self) -> np.ndarray:
        return np.array(
            [ase.data.atomic_numbers[s] for s in self.get_chemical_symbols()]
        )


class DistributedStructure(Distributed, Structure):
    @abc.abstractmethod
    def get_local_indices(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_local_chemical_symbols(self) -> tuple[str, ...]:
        ...

    @abc.abstractmethod
    def get_local_positions(self) -> np.ndarray:
        ...

    # Gather local properties

    def get_chemical_symbols(self) -> tuple[str, ...]:
        raise NotImplementedError

    def get_positions(self) -> np.ndarray:
        raise NotImplementedError


class ASEStructure(Structure):
    def __init__(self, atoms: ase.Atoms) -> None:
        self._atoms = atoms

    def get_chemical_symbols(self) -> tuple[str, ...]:
        return tuple(self._atoms.get_chemical_symbols())

    def get_positions(self) -> np.ndarray:
        return self._atoms.get_positions()

    def get_cell(self) -> np.ndarray:
        return self._atoms.get_cell()

    def get_pbc(self) -> tuple[bool, bool, bool]:
        x, y, z = self._atoms.get_pbc()
        return x, y, z

    def get_atomic_numbers(self) -> np.ndarray:
        return self._atoms.get_atomic_numbers()

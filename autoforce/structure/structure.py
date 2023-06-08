"""
Structure and MPI-parallelism interface.

"""
from __future__ import annotations

import abc

import numpy as np

import autoforce.mpi as mpi


class Structure(abc.ABC):
    @abc.abstractmethod
    def get_types(self) -> np.ndarray:  # [:] int array
        ...

    @abc.abstractmethod
    def get_positions(self) -> np.ndarray:  # [:, 3] float array
        ...

    @abc.abstractmethod
    def get_cell(self) -> np.ndarray:  # [3, 3] float array
        ...

    @abc.abstractmethod
    def get_pbc(self) -> tuple[bool, bool, bool]:
        ...


class DistributedStructure(mpi.Distributed, Structure):
    @abc.abstractmethod
    def get_local_indices(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_local_types(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_local_positions(self) -> np.ndarray:
        ...

    # Gather local properties

    def get_types(self) -> np.ndarray:
        # TODO: MPI gather
        raise NotImplementedError

    def get_positions(self) -> np.ndarray:
        # TODO: MPI gather
        raise NotImplementedError

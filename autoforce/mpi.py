"""
MPI-parallelism interface.

"""
from __future__ import annotations

import abc
import typing

import mpi4py.MPI as MPI


class Distributable(abc.ABC):
    @abc.abstractmethod
    def distribute(self, comm: MPI.Intracomm) -> Distributed:
        ...


class Distributed(abc.ABC):
    @abc.abstractmethod
    def get_mpi_comm(self) -> MPI.Intracomm:
        ...


def is_distributed(obj: typing.Any) -> bool:
    return isinstance(obj, Distributed)


def is_distributable(obj: typing.Any) -> bool:
    return isinstance(obj, Distributable)


def distribute(obj: Distributable, comm: MPI.Intracomm) -> Distributed:
    return obj.distribute(comm)

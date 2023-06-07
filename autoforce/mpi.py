"""
MPI-parallelism interface.

The "Distributed" class is an abstract base class for all classes
that implement MPI-parallelism. Any class that is not a subclass
of "Distributed" is assumed to be serial.

"""
import abc

import mpi4py.MPI as MPI


class Distributed(abc.ABC):
    @abc.abstractmethod
    def get_mpi_comm(self) -> MPI.Intracomm:
        ...


def is_distributed(obj) -> bool:
    return isinstance(obj, Distributed)

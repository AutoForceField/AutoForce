import numpy as np
from ase.build import bulk

from autoforce.structure.ase import ASENeighborList


def test_ASENeighborList() -> None:
    atoms = bulk("Cu", cubic=True).repeat(3)
    cutoff = 5.0
    nl = ASENeighborList(atoms, cutoff)
    for i in range(len(atoms)):
        displacements = nl.get_neighbors_displacements(i)
        neighbors = nl.get_neighbors_indices(i)
        types = nl.get_neighbors_types(i)
        offsets = nl.get_neighbors_offsets(i)
        distances = np.linalg.norm(displacements, axis=1)
        assert (
            len(neighbors) == len(displacements) == len(offsets) == len(types)
        )
        assert np.all(distances < cutoff)


if __name__ == "__main__":
    test_ASENeighborList()

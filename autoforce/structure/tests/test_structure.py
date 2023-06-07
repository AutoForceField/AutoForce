from ase.build import bulk

from autoforce.structure.structure import ASEStructure


def test_ASEStructure() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    structure = ASEStructure(atoms)
    structure.get_chemical_symbols()
    structure.get_positions()
    structure.get_cell()
    structure.get_pbc()
    structure.get_atomic_numbers()


if __name__ == "__main__":
    test_ASEStructure()

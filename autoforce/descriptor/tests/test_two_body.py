import torch

from autoforce.descriptor.smooth_cutoff import PolynomialCutoff
from autoforce.descriptor.two_body import GaussianKernel, RadialBasis


def arange_displacements(rmin, rmax, dr) -> tuple[torch.Tensor, torch.Tensor]:
    dx = torch.arange(rmin, rmax, dr)
    dy = torch.zeros_like(dx)
    dz = torch.zeros_like(dx)
    types = torch.tensor(len(dx) * [1])
    return torch.stack([dx, dy, dz]).T, types


def continuity(w: float | None = None) -> None:
    cutf = PolynomialCutoff(10.0)
    radii = torch.arange(0.5, cutf.cutoff, 0.1)
    descriptor = RadialBasis(
        radii, GaussianKernel(0.5), smooth_cutoff=cutf, radii_weights=w
    )
    displacements, types = arange_displacements(1.0, cutf.cutoff, 0.1)
    (d1,), _ = descriptor(displacements, types)
    displacements, types = arange_displacements(1.0, 2 * cutf.cutoff, 0.1)
    (d2,), _ = descriptor(displacements, types)
    assert d1.allclose(d2)


def test_continuity():
    continuity(None)
    continuity(1.0)

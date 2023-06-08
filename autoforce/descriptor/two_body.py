import abc
import typing
from math import pi

import torch

from .descriptor import DescriptorFunction, DescriptorType
from .smooth_cutoff import SmoothCutoff


class RadialKernel(abc.ABC):
    sigma: torch.Tensor

    @abc.abstractmethod
    def __call__(
        self,
        x: torch.Tensor,  # [:] float Tensor
        y: torch.Tensor,  # [:] float Tensor
    ) -> torch.Tensor:  # [x.shape[0], y.shape[0]] float Tensor
        pass


class GaussianKernel(RadialKernel):
    def __init__(self, sigma: float):
        self.sigma = torch.as_tensor(sigma)

    def __call__(
        self,
        x: torch.Tensor,  # [:] float Tensor
        y: torch.Tensor,  # [:] float Tensor
    ) -> torch.Tensor:  # [x.shape[0], y.shape[0]] float Tensor
        return (x[:, None] - y[None]).pow(2).div(-2 * self.sigma.pow(2)).exp()


class RadialBasis(DescriptorFunction):
    def __init__(
        self,
        radii: typing.Sequence[float],
        kernel: RadialKernel,
        smooth_cutoff: SmoothCutoff | None = None,
        radii_weights: float | typing.Sequence[float] | None = None,
    ):
        self.radii = torch.as_tensor(radii)
        self.kernel = kernel
        self.smooth_cutoff = smooth_cutoff
        if radii_weights is None:
            # Default to 1 / (4 * pi * r^2 * sigma)
            assert (self.radii > torch.finfo(self.radii.dtype).eps).all()
            self.radii_weights = (
                1 / (4 * pi * self.radii**2 * self.kernel.sigma)
            ).view(-1, 1)
        else:
            self.radii_weights = torch.as_tensor(radii_weights).view(-1, 1)

    def function(
        self,
        displacements: torch.Tensor,
        types: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> DescriptorType:
        distances = torch.norm(displacements, dim=-1)
        aux = []
        out = []
        for t in torch.unique(types):
            mask = types == t
            aux.append(t)
            k = self.radii_weights * self.kernel(self.radii, distances[mask])
            if weights is not None:
                k = k * weights[mask]
            v = k.sum(dim=1)
            out.append(v)
        return tuple(out), tuple(aux)

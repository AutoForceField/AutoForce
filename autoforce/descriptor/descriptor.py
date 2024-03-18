import abc

import torch

from .smooth_cutoff import SmoothCutoff

# Type aliases
# - A descriptor is a tuple of "Tensor" and a tuple of auxiliary "Tensor"s.
DescriptorType = tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]


class DescriptorFunction(abc.ABC):
    smooth_cutoff: SmoothCutoff | None = None

    def __call__(
        self,
        displacements: torch.Tensor,  # [:, 3] float Tensor
        types: torch.Tensor,  # [:] int Tensor
        weights: torch.Tensor | None = None,  # scalar or [:] float Tensor
    ) -> DescriptorType:
        if self.smooth_cutoff is not None:
            distances = displacements.norm(dim=1)
            cut = self.smooth_cutoff(distances)
            weights = cut if weights is None else cut * weights
        return self.function(displacements, types, weights)

    @abc.abstractmethod
    def function(
        self,
        displacements: torch.Tensor,  # [:, 3] float Tensor
        types: torch.Tensor,  # [:] int Tensor
        weights: torch.Tensor | None = None,  # scalar or [:] float Tensor
    ) -> DescriptorType: ...

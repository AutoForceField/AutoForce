# +
from __future__ import annotations

import abc
from math import pi

import torch
from torch import Tensor


class SmoothCutoff(abc.ABC):
    cutoff: float

    def __call__(
        self,
        d: Tensor,  # [:] float Tensor
    ) -> Tensor:
        return torch.where(
            d < self.cutoff,
            self.function(1 - d / self.cutoff),
            torch.zeros_like(d),
        )

    @abc.abstractmethod
    def function(self, x: Tensor) -> Tensor:
        """
        A class of functions that satisfy the following properties:
            - f(0) = 0
            - f(1) = 1
            - 0 < x < 1 --> 0 < f(x) < 1
            - f'(0) = 0
        The main property is that both its value f(x) and
        its derivative f'(x) should be zero at x = 0.

        """
        ...


class PolynomialCutoff(SmoothCutoff):
    def __init__(self, cutoff: float, n: int = 2):
        assert n > 1
        self.cutoff = cutoff
        self.n = n

    def function(self, x: Tensor) -> Tensor:
        return x**self.n


class SineCutoff(SmoothCutoff):
    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def function(self, x: Tensor) -> Tensor:
        return torch.sin(x * pi / 2)

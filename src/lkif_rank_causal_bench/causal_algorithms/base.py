"""The interface every causal algorithm implements, and its result type."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass
class Estimate:
    """One direction of one algorithm over a batch of grid points."""

    values: np.ndarray  # (n_points,)
    per_realisation: np.ndarray | None = None  # (n_points, n_realisations), CCC and CCM
    p_values: np.ndarray | None = None  # (n_points,), PCMCI only


class CausalAlgorithm(ABC):
    """A causality measure source -> target, evaluated on data of shape (n_points, R, 4, T)."""

    name: ClassVar[str]

    @abstractmethod
    def estimate(self, X: np.ndarray, source: int, target: int, indices: np.ndarray) -> Estimate:
        """Estimate source -> target at each grid point; indices are the points' global grid indices."""

    def estimate_both(
        self, X: np.ndarray, source: int, target: int, indices: np.ndarray, reverse: bool = True
    ) -> tuple[Estimate, Estimate | None]:
        """Forward and (optionally) reverse estimates; override when one pass gives both."""
        forward = self.estimate(X, source, target, indices)
        backward = self.estimate(X, target, source, indices) if reverse else None
        return forward, backward

    def derived(self, forward: Estimate) -> dict[str, np.ndarray]:
        """Extra result arrays computed from the forward estimate (none by default)."""
        return {}

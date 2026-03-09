"""
Alpha Signal Framework — base dataclass and ABC for all alpha strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AlphaSignal:
    """A trading signal from an alpha source."""
    pair: str                           # e.g. "eurusd"
    direction: float                    # -1.0 (short) to +1.0 (long)
    confidence: float                   # 0.0 to 1.0
    alpha_name: str                     # e.g. "rl", "momentum", "mean_reversion"
    regime: Optional[str] = None        # e.g. "trending", "ranging", "volatile"
    metadata: dict = field(default_factory=dict)

    @property
    def weighted_signal(self):
        """Direction * confidence — the effective signal strength."""
        return self.direction * self.confidence


class BaseAlpha(ABC):
    """Abstract base class for all alpha strategies."""

    def __init__(self, name: str, pairs: List[str]):
        self.name = name
        self.pairs = pairs

    @abstractmethod
    def generate(self, features: dict, **kwargs) -> List[AlphaSignal]:
        """Generate signals for all pairs.

        Args:
            features: {pair: feature_data} dict of per-pair feature data.

        Returns:
            List of AlphaSignal objects.
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, pairs={len(self.pairs)})"

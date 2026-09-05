"""Data shapes shared across the analyzer and the clarifier."""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Interpretation:
    """A single generated interpretation with its embedding."""
    text: str
    embedding: np.ndarray


@dataclass
class AmbiguityResult:
    """Result of analyzing a set of interpretations against a spec."""
    v1: Interpretation
    v2: Interpretation
    ambiguity_score: float          # PC1 explained variance
    semantic_volume: float          # log det of regularized Gram (centered)
    pc1_variance_ratio: float
    spec_projection: float          # where the spec itself falls on PC1
    all_interpretations: List[Interpretation] = field(default_factory=list)

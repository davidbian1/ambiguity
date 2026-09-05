"""AmbiguityAnalyzer: the embedding-space geometry, no validity filtering."""

from typing import Callable, List

import numpy as np
from sklearn.decomposition import PCA

from .models import AmbiguityResult, Interpretation


class AmbiguityAnalyzer:
    """
    Analyzes ambiguity by examining the geometry of a spec's interpretations
    in embedding space. No validity filtering — every interpretation that
    comes back from generation is used.
    """

    def __init__(self, embed_fn: Callable[[str], np.ndarray]):
        self.embed_fn = embed_fn

    def embed_interpretations(self, texts: List[str]) -> List[Interpretation]:
        return [Interpretation(text=t, embedding=self.embed_fn(t)) for t in texts]

    def find_greatest_ambiguity(self, interpretations: List[Interpretation], spec_emb: np.ndarray) -> AmbiguityResult:
        n = len(interpretations)
        reg = 1e-5

        if n == 0:
            raise ValueError("no interpretations to analyze")

        if n == 1:
            only = interpretations[0]
            # Centered matrix collapses to a single zero vector; the Gram
            # matrix is just [[reg]], so semantic_volume is log(reg).
            semantic_volume = float(np.log(reg))
            return AmbiguityResult(
                v1=only,
                v2=only,
                ambiguity_score=0.0,
                semantic_volume=semantic_volume,
                pc1_variance_ratio=0.0,
                spec_projection=0.0,
                all_interpretations=interpretations,
            )

        X = np.stack([i.embedding for i in interpretations])  # (n, d)
        
        mean = X.mean(axis=0)
        X_centered = X - mean

        pca = PCA(n_components=1)
        pca.fit(X_centered)
        axis = pca.components_[0]

        projections = X_centered @ axis
        min_idx = int(np.argmin(projections))
        max_idx = int(np.argmax(projections))

        gram = X_centered @ X_centered.T + reg * np.eye(n)
        _, logdet = np.linalg.slogdet(gram)
        semantic_volume = float(logdet)

        spec_projection = float((spec_emb - mean) @ axis)

        return AmbiguityResult(
            v1=interpretations[min_idx],
            v2=interpretations[max_idx],
            ambiguity_score=float(pca.explained_variance_[0]),
            semantic_volume=semantic_volume,
            pc1_variance_ratio=float(pca.explained_variance_ratio_[0]),
            spec_projection=spec_projection,
            all_interpretations=interpretations,
        )

    # =========================================================================
    # PROPOSED REPLACEMENT (not active — sketch only, per ticket 004).
    # Replaces find_greatest_ambiguity() (lines 24-73) and analyze() below.
    #
    # from collections import Counter
    # from dataclasses import dataclass, field
    # from math import log2
    # from typing import Dict, List, Optional
    #
    # Particle = Dict[str, Optional[str]]  # {axis_name: value_or_None}, flat — no prose
    #
    # @dataclass
    # class SpecState:
    #     seed_text: str
    #     particles: List[Particle] = field(default_factory=list)
    #     weights: List[float] = field(default_factory=list)      # sums to 1
    #     ledger: Dict[str, str] = field(default_factory=dict)     # axis -> "confirmed" | "accepted-default"
    #     axis_class: Dict[str, str] = field(default_factory=dict) # axis -> "decision" | "detail"
    #
    # def axis_entropy(particles: List[Particle], weights: List[float], axis: str) -> float:
    #     """Shannon entropy of this axis's weighted value distribution across
    #     particles — a real, counted quantity, NOT an LLM self-judgment.
    #     Replaces PCA's step 4-5 (center + fit PC1) as the priority signal."""
    #     value_weight: Dict[Optional[str], float] = {}
    #     for particle, w in zip(particles, weights):
    #         v = particle.get(axis)
    #         value_weight[v] = value_weight.get(v, 0.0) + w
    #     total = sum(value_weight.values()) or 1.0
    #     h = 0.0
    #     for w in value_weight.values():
    #         p = w / total
    #         if p > 0:
    #             h -= p * log2(p)
    #     return h
    #
    # def pick_next_axis(state: SpecState) -> Optional[str]:
    #     """Highest-entropy OPEN decision-class axis. Replaces PCA's
    #     steps 5-6 (fit PC1, project, pick min/max) — the 'question' is now
    #     an axis's real candidate values, not two PCA-extremal texts."""
    #     open_decision_axes = [
    #         a for a in state.axis_class
    #         if state.axis_class[a] == "decision" and a not in state.ledger
    #     ]
    #     if not open_decision_axes:
    #         return None
    #     return max(open_decision_axes, key=lambda a: axis_entropy(state.particles, state.weights, a))
    #
    # def compute_metrics(state: SpecState) -> Dict[str, float]:
    #     """Maps the ORIGINAL two required outputs onto the new state, so
    #     callers relying on these two field names still get something
    #     meaningful. Replaces steps 7-8 (PC1 eigenvalue; log-det Gram)."""
    #     open_axes = [a for a in state.axis_class if a not in state.ledger]
    #     ambiguity_score = sum(axis_entropy(state.particles, state.weights, a) for a in open_axes)
    #     combos = Counter(tuple(sorted(p.items())) for p in state.particles)
    #     # exp(entropy over combination distribution) — a smooth analog to a
    #     # discrete "how many effectively-distinct specs remain" count, closer
    #     # in spirit to the original's continuous log-volume than a raw count.
    #     combo_weight: Dict[tuple, float] = {}
    #     for p, w in zip(state.particles, state.weights):
    #         key = tuple(sorted(p.items()))
    #         combo_weight[key] = combo_weight.get(key, 0.0) + w
    #     total = sum(combo_weight.values()) or 1.0
    #     h_combo = -sum((w / total) * log2(w / total) for w in combo_weight.values() if w > 0)
    #     semantic_volume = float(2 ** h_combo)  # "effective number of distinct surviving specs"
    #     return {"ambiguity_score": ambiguity_score, "semantic_volume": semantic_volume}
    #
    # from .prompts import extract_axes, generate_axis_particle  # -> prompts.yaml sketch
    # from .prompts import score_consistency, expand_axes         # -> prompts.yaml sketch
    #
    # class AmbiguityAnalyzer:  # embed_fn param dropped — unused by this design
    #     def __init__(self):
    #         pass
    # =========================================================================

    def analyze(self, spec: str, interpretations: List[str]) -> AmbiguityResult:
        """Full pipeline: embed spec -> embed interpretations -> find ambiguity."""
        spec_emb = self.embed_fn(spec)
        interp_objs = self.embed_interpretations(interpretations)
        return self.find_greatest_ambiguity(interp_objs, spec_emb)

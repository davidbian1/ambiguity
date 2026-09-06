# Ambiguity Clarifier

Resolves ambiguity in a single evolving spec by generating candidate interpretations, scoring how spread apart they are, and looping a human through the widest gap until the spec converges.

## Language

**Interpretation**:
One generated reading of the current spec, carried with its embedding.

**Ambiguity score**:
PC1 explained variance across a spec's generated interpretations — how much of their spread is explained by a single axis of disagreement. High score means the interpretations mostly disagree about one thing.
_Avoid_: "ambiguity" alone — always qualify as this spec-level score. See [[Wayfinder Graph]]'s CONTEXT.md, which scores a different thing under the same word.

**Semantic volume**:
Log-determinant of the (regularized, centered) Gram matrix of interpretation embeddings — how spread out the interpretations are as a whole, not just along one axis.

**Spec projection**:
Where the current spec itself falls on the interpretations' dominant axis of disagreement (PC1).

**Particle**:
One candidate `{axis: value}` assignment in the particles-only redesign — a full guess at every open decision axis at once, not a single interpretation of prose.
_Avoid_: Interpretation (particles replace interpretations in that redesign; the two never coexist in one graph run).

**Axis**:
One named dimension of the spec that can vary (e.g. "platform", "auth strategy") — a particle assigns a value to each axis.

**Ledger**:
The record of which axes have been confirmed by the human, in the particles-only redesign.

# Context Map

## Contexts

- [Ambiguity Clarifier](./src/ambiguity/CONTEXT.md) — resolves ambiguity in a spec by generating and scoring interpretations against a human
- [Wayfinder Graph](./src/wayfinder_graph/CONTEXT.md) — resolves open decisions in a body of work by working a chart of tickets against a human

## Relationships

- **Wayfinder Graph → Ambiguity Clarifier**: no runtime dependency. Wayfinder Graph may reuse Ambiguity Clarifier's embedding/PCA machinery (`AmbiguityAnalyzer`) as an optional, observation-only scorer — it never shares state, storage, or a graph with it.
- Both contexts use the word "ambiguity" for different things — see each context's `_Avoid_` notes below before assuming a shared meaning.

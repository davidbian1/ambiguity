"""
================================================================================
AMBIGUITY CLARIFIER
================================================================================

Iteratively clarifies a vague project idea by finding the "greatest
ambiguity" among plausible interpretations (via embeddings + PCA) and
asking a human to resolve it, one round at a time.
"""

import json
import re
from typing import List, Dict, Callable, Optional, Any
import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA


@dataclass
class Interpretation:
    """A single interpretation with its embedding and validity score."""
    text: str
    embedding: np.ndarray
    validity: float  # cosine similarity to original idea


@dataclass
class AmbiguityResult:
    """Result of analyzing a set of interpretations."""
    v1: Interpretation
    v2: Interpretation
    ambiguity_score: float      # PC1 variance
    semantic_volume: float      # log det of Gram matrix
    pc1_variance_ratio: float   # explained variance ratio of PC1
    all_valid: List[Interpretation]


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    return text


class AmbiguityAnalyzer:
    """
    Analyzes ambiguity in a specification by examining the geometry of
    its interpretations in embedding space.

    Core algorithm:
    1. Embed the original idea
    2. Embed all interpretations, compute validity (cos_sim to original)
    3. Filter to valid interpretations (validity >= threshold)
    4. Center the valid embedding matrix
    5. Run PCA. PC1 is the "ambiguity axis" — the direction of maximum
       systematic divergence among valid interpretations.
    6. The two interpretations with the most extreme projections onto PC1
       are v1 and v2 — the "greatest ambiguity".
    7. Ambiguity score = eigenvalue of PC1 (variance along ambiguity axis)
    8. Semantic volume = log_det(V^T V) of the valid embedding matrix

    NOTE: Do NOT use raw pairwise distance (diameter) to find v1/v2.
    Raw diameter picks outliers. PCA finds the systematic divergence axis.
    """

    def __init__(self, embed_fn: Callable[[str], np.ndarray],
                 validity_threshold: float = 0.55):
        self.embed_fn = embed_fn
        self.validity_threshold = validity_threshold

    def embed_idea(self, idea: str) -> np.ndarray:
        """Embed the original idea and return a normalized vector."""
        vec = np.asarray(self.embed_fn(idea), dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_interpretations(self, texts: List[str],
                               idea_emb: np.ndarray) -> List[Interpretation]:
        """
        Embed each interpretation text and compute its validity
        (cosine similarity to the original idea embedding).
        """
        interps = []
        for text in texts:
            vec = np.asarray(self.embed_fn(text), dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            validity = _cos_sim(vec, idea_emb)
            interps.append(Interpretation(text=text, embedding=vec, validity=validity))
        return interps

    def filter_valid(self, interps: List[Interpretation]) -> List[Interpretation]:
        """
        Filter interpretations to only those with validity >= threshold.

        EDGE CASE: If fewer than 2 interpretations pass the threshold,
        fall back to the top 2 by validity (or all if fewer than 2 exist).
        """
        valid = [i for i in interps if i.validity >= self.validity_threshold]
        if len(valid) >= 2:
            return valid
        ranked = sorted(interps, key=lambda i: i.validity, reverse=True)
        return ranked[:2]

    def find_greatest_ambiguity(self, valid_interps: List[Interpretation]) -> AmbiguityResult:
        """
        Find the greatest ambiguity among valid interpretations via PCA.
        """
        n = len(valid_interps)
        if n == 0:
            raise ValueError("find_greatest_ambiguity requires at least one interpretation")

        if n == 1:
            only = valid_interps[0]
            X = only.embedding.reshape(1, -1)
            gram = X @ X.T + 1e-5 * np.eye(1)
            _, logdet = np.linalg.slogdet(gram)
            return AmbiguityResult(
                v1=only,
                v2=only,
                ambiguity_score=0.0,
                semantic_volume=float(logdet),
                pc1_variance_ratio=0.0,
                all_valid=valid_interps,
            )

        X = np.stack([i.embedding for i in valid_interps])  # (n, d)
        mean = X.mean(axis=0)
        X_centered = X - mean

        n_components = max(1, min(n - 1, X_centered.shape[1]))
        pca = PCA(n_components=n_components)
        pca.fit(X_centered)

        scores = X_centered @ pca.components_[0]  # projection onto PC1

        min_idx = int(np.argmin(scores))
        max_idx = int(np.argmax(scores))

        gram = X_centered @ X_centered.T + 1e-5 * np.eye(n)
        _, logdet = np.linalg.slogdet(gram)

        return AmbiguityResult(
            v1=valid_interps[min_idx],
            v2=valid_interps[max_idx],
            ambiguity_score=float(pca.explained_variance_[0]),
            semantic_volume=float(logdet),
            pc1_variance_ratio=float(pca.explained_variance_ratio_[0]),
            all_valid=valid_interps,
        )

    def analyze(self, idea: str, interpretations: List[str]) -> AmbiguityResult:
        """
        Full pipeline: embed idea → embed interpretations → filter valid
        → find greatest ambiguity.
        """
        idea_emb = self.embed_idea(idea)
        interps = self.embed_interpretations(interpretations, idea_emb)
        valid = self.filter_valid(interps)
        return self.find_greatest_ambiguity(valid)


class IterativeClarifier:
    """
    Runs the full clarification loop:

    While human hasn't stopped and round < max_rounds:
        1. Generate N interpretations of current_spec via LLM
        2. Analyze ambiguity → get v1, v2, scores
        3. Present to human → get choice (v1, v2, tolerate, stop)
        4. Integrate choice into current_spec via LLM
        5. Record metrics
    """

    def __init__(self, llm_client: Any,
                 embed_fn: Callable[[str], np.ndarray],
                 max_rounds: int = 5,
                 n_interpretations: int = 8):
        self.llm_client = llm_client
        self.analyzer = AmbiguityAnalyzer(embed_fn)
        self.max_rounds = max_rounds
        self.n_interpretations = n_interpretations
        self.history: List[Dict] = []

    def _chat(self, messages: List[Dict], model: str, temperature: float,
              max_tokens: int) -> str:
        resp = self.llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def generate_interpretations(self, spec: str) -> List[str]:
        """
        Prompt the LLM to generate N distinct but valid interpretations
        of the current specification.
        """
        prompt = (
            f"Here is a project specification:\n\n\"\"\"\n{spec}\n\"\"\"\n\n"
            f"Generate exactly {self.n_interpretations} distinct, plausible "
            f"interpretations of this specification. Each interpretation should "
            f"be a concrete, specific reading of what the spec could mean in "
            f"practice, while still being a faithful reading of it (do not "
            f"invent unrelated projects).\n\n"
            f"Return ONLY a JSON array of {self.n_interpretations} strings, "
            f"with no other text. Example format:\n"
            f'["interpretation 1", "interpretation 2", ...]'
        )
        messages = [{"role": "user", "content": prompt}]
        raw = self._chat(messages, model="gpt-4o-mini", temperature=0.9, max_tokens=2000)

        cleaned = _strip_json_fences(raw)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (json.JSONDecodeError, TypeError):
            pass

        lines = [line.strip(" -*\t") for line in raw.splitlines()]
        return [line for line in lines if line]

    def summarize_v1v2(self, spec: str, v1_text: str, v2_text: str) -> Dict[str, str]:
        """
        Prompt the LLM to produce concise, neutral summaries of two
        interpretation extremes, plus a short name for the ambiguity topic.
        """
        prompt = (
            f"Specification:\n\"\"\"\n{spec}\n\"\"\"\n\n"
            f"Two extreme interpretations of this specification were found:\n\n"
            f"Interpretation A:\n\"\"\"\n{v1_text}\n\"\"\"\n\n"
            f"Interpretation B:\n\"\"\"\n{v2_text}\n\"\"\"\n\n"
            f"Produce concise, neutral summaries of each extreme, and a short "
            f"name for the underlying design decision they disagree about.\n\n"
            f"Return ONLY JSON in this exact shape, with no other text:\n"
            f'{{"v1_summary": "2-sentence concrete description of A", '
            f'"v2_summary": "2-sentence concrete description of B", '
            f'"ambiguity_topic": "short name for this design decision"}}'
        )
        messages = [{"role": "user", "content": prompt}]
        raw = self._chat(messages, model="gpt-4o-mini", temperature=0.3, max_tokens=500)

        cleaned = _strip_json_fences(raw)
        try:
            parsed = json.loads(cleaned)
            return {
                "v1_summary": str(parsed.get("v1_summary", v1_text)).strip(),
                "v2_summary": str(parsed.get("v2_summary", v2_text)).strip(),
                "ambiguity_topic": str(parsed.get("ambiguity_topic", "Unnamed ambiguity")).strip(),
            }
        except (json.JSONDecodeError, TypeError, AttributeError):
            return {
                "v1_summary": v1_text,
                "v2_summary": v2_text,
                "ambiguity_topic": "Unnamed ambiguity",
            }

    def integrate_choice(self, spec: str, ambiguity_topic: str,
                          v1_summary: str, v2_summary: str,
                          choice: str, prior_choices: List[Dict]) -> str:
        """
        Prompt the LLM to rewrite the specification to incorporate the
        human's choice.
        """
        if choice == "v1":
            decision = f"Use approach: {v1_summary}"
        elif choice == "v2":
            decision = f"Use approach: {v2_summary}"
        elif choice == "tolerate":
            decision = f"Either {v1_summary} OR {v2_summary} is acceptable."
        else:
            decision = ""

        prior_text = ""
        if prior_choices:
            lines = [
                f"- {pc.get('topic', 'Unnamed ambiguity')}: {pc.get('choice', '')}"
                for pc in prior_choices
            ]
            prior_text = "Prior decisions made in earlier rounds:\n" + "\n".join(lines) + "\n\n"

        prompt = (
            f"Current specification:\n\"\"\"\n{spec}\n\"\"\"\n\n"
            f"{prior_text}"
            f"A new decision was just made about \"{ambiguity_topic}\":\n"
            f"{decision}\n\n"
            f"Rewrite the full specification to naturally integrate this decision "
            f"into the text — do not just append a note or bullet point at the "
            f"end. Keep everything else about the spec that isn't affected by "
            f"this decision. Return ONLY the full rewritten specification text, "
            f"with no other commentary."
        )
        messages = [{"role": "user", "content": prompt}]
        raw = self._chat(messages, model="gpt-4o-mini", temperature=0.3, max_tokens=1500)
        return raw.strip()

    def human_input(self, round_info: Dict) -> str:
        """
        CLI interface. Display v1, v2, metrics, and prompt for choice.
        """
        print()
        print(f"=== Round {round_info['round']} - {round_info['topic']} ===")
        print(f"[1] v1: {round_info['v1']}")
        print(f"[2] v2: {round_info['v2']}")
        print(
            f"ambiguity_score={round_info['ambiguity_score']:.4f}  "
            f"semantic_volume={round_info['semantic_volume']:.4f}  "
            f"pc1_variance_ratio={round_info['pc1_variance_ratio']:.4f}"
        )
        print("[3] tolerate this ambiguity")
        print("[q] stop and finalize")

        raw = input("Your choice: ").strip().lower()
        if raw in ("1", "v1"):
            return "v1"
        if raw in ("2", "v2"):
            return "v2"
        if raw in ("3", "tolerate"):
            return "tolerate"
        if raw in ("q", "stop"):
            return "stop"
        return "stop"

    def clarify(self, idea: str) -> Dict:
        """MAIN LOOP."""
        current_spec = idea
        self.history = []

        for round_num in range(1, self.max_rounds + 1):
            interpretations = self.generate_interpretations(current_spec)
            result = self.analyzer.analyze(current_spec, interpretations)
            descriptions = self.summarize_v1v2(current_spec, result.v1.text, result.v2.text)

            choice = self.human_input({
                "round": round_num,
                "topic": descriptions["ambiguity_topic"],
                "v1": descriptions["v1_summary"],
                "v2": descriptions["v2_summary"],
                "ambiguity_score": result.ambiguity_score,
                "semantic_volume": result.semantic_volume,
                "pc1_variance_ratio": result.pc1_variance_ratio,
                "spec": current_spec,
            })

            if choice == "stop":
                break

            prior_choices = [
                {"topic": h["topic"], "choice": h["choice"]}
                for h in self.history
            ]

            current_spec = self.integrate_choice(
                spec=current_spec,
                ambiguity_topic=descriptions["ambiguity_topic"],
                v1_summary=descriptions["v1_summary"],
                v2_summary=descriptions["v2_summary"],
                choice=choice,
                prior_choices=prior_choices,
            )

            self.history.append({
                "round": round_num,
                "topic": descriptions["ambiguity_topic"],
                "v1": descriptions["v1_summary"],
                "v2": descriptions["v2_summary"],
                "choice": choice,
                "ambiguity_score": result.ambiguity_score,
                "semantic_volume": result.semantic_volume,
            })

        return {
            "original_idea": idea,
            "final_spec": current_spec,
            "rounds": self.history,
            "total_rounds": len(self.history),
        }


# ================================================================================
# EXAMPLE USAGE
# ================================================================================
if __name__ == "__main__":
    from openai import OpenAI

    client = OpenAI()

    def embed(text: str) -> np.ndarray:
        resp = client.embeddings.create(input=text, model="text-embedding-3-small")
        vec = np.array(resp.data[0].embedding, dtype=np.float64)
        return vec / np.linalg.norm(vec)

    clarifier = IterativeClarifier(llm_client=client, embed_fn=embed)
    result = clarifier.clarify("Build a social app where people share recommendations")

    print(result["final_spec"])

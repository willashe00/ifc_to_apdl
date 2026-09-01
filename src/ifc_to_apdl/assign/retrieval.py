"""Dense-retrieval material matching (manuscript Phase 2) with graceful
degradation: if the sentence-transformer backend is unavailable the resolver
falls back to a lexical scorer, records which backend ran, and applies the
backend-appropriate acceptance threshold."""

from __future__ import annotations

import difflib

from .material_library import LIBRARY, LibraryEntry

_MODEL_NAME = "all-MiniLM-L6-v2"

#: qualifier tokens: a library entry carrying one is eligible ONLY when the
#: material name carries it too — an unqualified name defaults to the
#: unqualified entry (industry convention: unqualified concrete is
#: normal-weight)
_QUALIFIERS = ("lightweight",)


class Retriever:
    def __init__(self, backend: str = "auto"):
        """backend: 'auto' (embedding with lexical fallback) or 'lexical'
        (forced — used by degradation experiments)."""
        self.backend = "lexical"
        self._model = None
        self._embeddings = None
        if backend == "lexical":
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(_MODEL_NAME)
            corpus = [e.descriptor for e in LIBRARY]
            self._embeddings = self._model.encode(corpus, normalize_embeddings=True)
            self.backend = "embedding"
        except Exception:
            self._model = None

    def query(self, descriptor: str,
              prefer_terms: tuple[str, ...] = ()) -> tuple[LibraryEntry, float]:
        """Best library entry and its similarity score.

        prefer_terms: exact designation tokens (e.g. 'c30/37', 's355',
        'a992', '4000 psi') extracted from the material name. Regulatory
        designations are exact identifiers, so when at least one library
        descriptor carries a term verbatim, selection is restricted to those
        entries — the similarity score of the selected entry is still the
        one gated against the acceptance threshold, so semantically weak
        contexts flag as before.
        """
        pool = self._preferred(prefer_terms) or range(len(LIBRARY))
        pool = self._apply_qualifiers(pool, prefer_terms)
        if self.backend == "embedding":
            q = self._model.encode([descriptor], normalize_embeddings=True)[0]
            sims = self._embeddings @ q
            best = max(pool, key=lambda i: sims[i])
            return LIBRARY[best], float(sims[best])
        return self._lexical(descriptor, pool)

    @staticmethod
    def _preferred(terms: tuple[str, ...]) -> list[int]:
        import re

        out = []
        for i, e in enumerate(LIBRARY):
            d = e.descriptor.lower()
            if any(re.search(rf"(?<![a-z0-9]){re.escape(t.lower())}(?![a-z0-9])", d)
                   for t in terms):
                out.append(i)
        return out

    @staticmethod
    def _apply_qualifiers(pool, terms: tuple[str, ...]):
        for q in _QUALIFIERS:
            if q in terms:
                continue
            unqualified = [i for i in pool
                           if q not in LIBRARY[i].descriptor.lower()]
            if unqualified:
                pool = unqualified
        return pool

    @staticmethod
    def _lexical(descriptor: str, pool) -> tuple[LibraryEntry, float]:
        d = descriptor.lower()
        d_tokens = set(d.split())
        best, best_s = None, -1.0
        for i in pool:
            e = LIBRARY[i]
            et = set(e.descriptor.lower().split())
            jacc = len(d_tokens & et) / max(1, len(d_tokens | et))
            seq = difflib.SequenceMatcher(None, d, e.descriptor.lower()).ratio()
            s = 0.6 * jacc + 0.4 * seq
            if s > best_s:
                best, best_s = e, s
        return best, best_s

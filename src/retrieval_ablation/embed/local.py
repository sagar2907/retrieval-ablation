"""Local GPU embedding models: BGE-M3, E5, and a domain-adapted option.

Imported lazily. `sentence-transformers` pulls in torch and downloads weights, so
importing it at module scope would make the offline test suite depend on both.
Nothing here is touched unless a real embedding run is requested.

VRAM is the binding constraint on the development machine: an RTX 4050 with
6,141 MiB. BGE-M3 is a 568M-parameter XLM-RoBERTa-large derivative, roughly
2.2 GB in fp16, and the cross-encoder reranker is a similar size. Both fit
individually; both resident at once plus activation memory does not, reliably. So
embedding and reranking run as separate passes over the corpus rather than being
pipelined, and models are released between passes. That ordering is load-bearing,
not incidental.

BGE-M3's multi-vector (ColBERT) output is deliberately not used. It emits one
vector per token, which for a 37,000-chunk corpus at 512 tokens is on the order of
19 million vectors instead of 37,000 -- a 500x storage increase for a component
that is not on any axis of this ablation. Dense output only; the lexical arm is
BM25, which is what the study actually compares against.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from ..config import MODEL_DIR
from .base import Embedder, l2_normalize

log = logging.getLogger(__name__)

#: Model identifiers and the query/passage prefixes each one requires.
#:
#: The prefixes are not cosmetic. E5 was trained with literal "query: " and
#: "passage: " markers and loses a large amount of retrieval accuracy without
#: them, with no error to indicate anything is wrong -- a silent accuracy loss is
#: exactly the kind of bug an ablation would misattribute to the model itself.
MODEL_SPECS: dict[str, dict[str, str]] = {
    "bge-m3": {
        "repo": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5-base": {
        "repo": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    # English-only E5 v2, paired with multilingual-e5-base above so the third
    # embedding arm varies one thing: whether the model spends its capacity on
    # other languages. Same family, same size, same prefixes, English-specialised.
    #
    # This entry was previously named "finance-e5" and commented as domain-adapted
    # to financial text. That was simply false -- intfloat/e5-base-v2 is trained on
    # general web data like the rest of the family, and no amount of naming makes
    # it otherwise. The arm was never measured, so nothing rested on the claim, but
    # a label asserting a property the weights do not have is the kind of thing a
    # reader has no way to check.
    "e5-base-v2": {
        "repo": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}


class SentenceTransformerEmbedder(Embedder):
    """Wraps a sentence-transformers model with the right asymmetric prefixes."""

    def __init__(
        self,
        model_key: str = "bge-m3",
        device: str | None = None,
        batch_size: int = 16,
        max_seq_length: int | None = 512,
        fp16: bool = True,
    ) -> None:
        if model_key not in MODEL_SPECS:
            raise ValueError(f"unknown model {model_key!r}; known: {sorted(MODEL_SPECS)}")
        self.name = model_key
        self._spec = MODEL_SPECS[model_key]
        self.batch_size = batch_size
        self._model = None
        self._device = device
        self._max_seq_length = max_seq_length
        self._fp16 = fp16
        self.dimension = 0  # set on load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from sentence_transformers import SentenceTransformer

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading %s on %s", self._spec["repo"], device)

        self._model = SentenceTransformer(
            self._spec["repo"],
            device=device,
            cache_folder=str(MODEL_DIR),
        )
        if self._max_seq_length:
            # Capped well below the model's 8,192 limit. Attention memory grows
            # with the square of sequence length, and a 6 GB card cannot hold the
            # activations for a full-length batch. Chunks are ~512 tokens by
            # design, so a higher limit would buy nothing but out-of-memory risk.
            self._model.max_seq_length = self._max_seq_length
        if self._fp16 and device == "cuda":
            self._model = self._model.half()

        # sentence-transformers 5.x renamed this; 3.x and 4.x only have the old
        # name. Preferring the new one and falling back keeps a single code path
        # working across the version the development machine pins and whatever a
        # borrowed GPU environment happens to ship.
        getter = getattr(
            self._model,
            "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension", None),
        )
        self.dimension = int(getter()) if getter else 0

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None

        prefix = self._spec["query_prefix" if is_query else "passage_prefix"]
        prepared = [prefix + t for t in texts] if prefix else list(texts)

        vectors = self._model.encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        # Normalised here rather than by the model so every embedder in the
        # project returns unit vectors under the same code path, which means
        # cosine similarity is a plain dot product everywhere downstream.
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def release(self) -> None:
        """Free the model and its VRAM.

        Called explicitly between passes. On 6 GB, leaving an embedding model
        resident while loading the cross-encoder is the difference between a run
        that completes and one that dies partway with a CUDA out-of-memory error
        after hours of work.
        """
        if self._model is None:
            return
        self._model = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def gpu_report() -> dict[str, object]:
    """What torch actually sees. Reported rather than assumed."""
    try:
        import torch
    except ImportError:
        return {"torch": "not installed"}

    info: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device"] = props.name
        info["total_vram_mib"] = props.total_memory // (1024 * 1024)
        info["capability"] = f"{props.major}.{props.minor}"
    return info

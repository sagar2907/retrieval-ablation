"""Embedding models. Real models live behind the optional `gpu` extra."""

from .base import Embedder, HashingEmbedder, l2_normalize

__all__ = ["Embedder", "HashingEmbedder", "l2_normalize"]

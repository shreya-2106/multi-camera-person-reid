"""Re-identification embedding module (OSNet). Not yet implemented (Phase 3)."""
"""Phase 6 pretrained OSNet appearance embeddings (no identity matching)."""

from src.reid.types import EmbeddingExtractionResult, EmbeddingMetadata

__all__ = ["ReIDEmbedder", "ReIDError", "EmbeddingExtractionResult", "EmbeddingMetadata"]


def __getattr__(name):
    """Keep importing the package side-effect free until inference is requested."""
    if name in {"ReIDEmbedder", "ReIDError"}:
        from src.reid.embedder import ReIDEmbedder, ReIDError
        return {"ReIDEmbedder": ReIDEmbedder, "ReIDError": ReIDError}[name]
    raise AttributeError(name)

"""Built-in document source providers."""

from offline_rag.sources.filesystem import DiscoveryOptions, FileSystemSourceProvider
from offline_rag.sources.manifest import ManifestSourceProvider, apply_sidecar

__all__ = [
    "DiscoveryOptions",
    "FileSystemSourceProvider",
    "ManifestSourceProvider",
    "apply_sidecar",
]

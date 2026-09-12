"""Built-in document source providers."""

from corpuscore.sources.filesystem import DiscoveryOptions, FileSystemSourceProvider
from corpuscore.sources.manifest import ManifestSourceProvider, apply_sidecar

__all__ = [
    "DiscoveryOptions",
    "FileSystemSourceProvider",
    "ManifestSourceProvider",
    "apply_sidecar",
]

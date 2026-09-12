"""Strict, versioned configuration loading and merging."""

from corpuscore.config.loader import deep_merge, load_config, parse_cli_overrides
from corpuscore.config.models import CorpusConfig

__all__ = ["CorpusConfig", "deep_merge", "load_config", "parse_cli_overrides"]

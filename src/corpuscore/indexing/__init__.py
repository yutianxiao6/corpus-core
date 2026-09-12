"""Index construction and compatibility helpers."""

from corpuscore.indexing.compatibility import assert_index_compatible, compare_index_specs
from corpuscore.indexing.journal import IndexedSource, IngestionJournal
from corpuscore.indexing.sync import SyncPlan, plan_sync

__all__ = [
    "IndexedSource",
    "IngestionJournal",
    "SyncPlan",
    "assert_index_compatible",
    "compare_index_specs",
    "plan_sync",
]

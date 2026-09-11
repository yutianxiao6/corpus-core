"""Index construction and compatibility helpers."""

from offline_rag.indexing.compatibility import assert_index_compatible, compare_index_specs
from offline_rag.indexing.journal import IndexedSource, IngestionJournal
from offline_rag.indexing.sync import SyncPlan, plan_sync

__all__ = [
    "IndexedSource",
    "IngestionJournal",
    "SyncPlan",
    "assert_index_compatible",
    "compare_index_specs",
    "plan_sync",
]

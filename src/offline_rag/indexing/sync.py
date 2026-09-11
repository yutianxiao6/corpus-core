"""Deterministic incremental synchronization planning."""

from __future__ import annotations

from dataclasses import dataclass

from offline_rag.contracts.documents import SourceDescriptor
from offline_rag.indexing.journal import IndexedSource


@dataclass(frozen=True, slots=True)
class SyncPlan:
    new: tuple[SourceDescriptor, ...]
    modified: tuple[tuple[SourceDescriptor, IndexedSource], ...]
    unchanged: tuple[tuple[SourceDescriptor, IndexedSource], ...]
    missing: tuple[IndexedSource, ...]


def plan_sync(
    discovered: tuple[SourceDescriptor, ...], indexed: tuple[IndexedSource, ...]
) -> SyncPlan:
    discovered_by_id = {source.source_id: source for source in discovered}
    indexed_by_id = {source.source_id: source for source in indexed}
    new: list[SourceDescriptor] = []
    modified: list[tuple[SourceDescriptor, IndexedSource]] = []
    unchanged: list[tuple[SourceDescriptor, IndexedSource]] = []
    for source in discovered:
        previous = indexed_by_id.get(source.source_id)
        if previous is None:
            new.append(source)
        elif previous.content_hash == source.content_hash:
            unchanged.append((source, previous))
        else:
            modified.append((source, previous))
    missing = tuple(
        indexed_by_id[source_id]
        for source_id in sorted(indexed_by_id.keys() - discovered_by_id.keys())
    )
    return SyncPlan(tuple(new), tuple(modified), tuple(unchanged), missing)

"""Retrieval result organizers."""

from offline_rag.organizers.context import ContextOrganizer
from offline_rag.organizers.debug import DebugOrganizer
from offline_rag.organizers.diverse import DiverseOrganizer
from offline_rag.organizers.flat import FlatOrganizer
from offline_rag.organizers.grouped import GroupByDocumentOrganizer
from offline_rag.organizers.structured import MergeNeighborsOrganizer, ParentOrganizer

__all__ = [
    "ContextOrganizer",
    "DebugOrganizer",
    "DiverseOrganizer",
    "FlatOrganizer",
    "GroupByDocumentOrganizer",
    "MergeNeighborsOrganizer",
    "ParentOrganizer",
]

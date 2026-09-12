"""Retrieval result organizers."""

from corpuscore.organizers.context import ContextOrganizer
from corpuscore.organizers.debug import DebugOrganizer
from corpuscore.organizers.diverse import DiverseOrganizer
from corpuscore.organizers.flat import FlatOrganizer
from corpuscore.organizers.grouped import GroupByDocumentOrganizer
from corpuscore.organizers.structured import MergeNeighborsOrganizer, ParentOrganizer

__all__ = [
    "ContextOrganizer",
    "DebugOrganizer",
    "DiverseOrganizer",
    "FlatOrganizer",
    "GroupByDocumentOrganizer",
    "MergeNeighborsOrganizer",
    "ParentOrganizer",
]

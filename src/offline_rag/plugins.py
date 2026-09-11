"""Explicit and auditable discovery of installed component plugins."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from typing import Protocol

from offline_rag.exceptions import RegistryError
from offline_rag.registry import ComponentRegistry

PLUGIN_GROUPS: Mapping[str, str] = {
    "source_providers": "offline_rag.source_providers",
    "loaders": "offline_rag.loaders",
    "parsers": "offline_rag.parsers",
    "chunkers": "offline_rag.chunkers",
    "processors": "offline_rag.processors",
    "embeddings": "offline_rag.embeddings",
    "vector_stores": "offline_rag.vector_stores",
    "retrieval_strategies": "offline_rag.retrieval_strategies",
    "organizers": "offline_rag.organizers",
}


class PluginEntryPoint(Protocol):
    name: str
    value: str

    def load(self) -> object: ...


EntryPointReader = Callable[[str], Sequence[PluginEntryPoint]]


def _installed_entry_points(group: str) -> Sequence[PluginEntryPoint]:
    return tuple(metadata.entry_points(group=group))


class PluginManager:
    """Load only explicitly enabled package entry points into component registries.

    Enabled identifiers use ``<kind>:<entry-point-name>``. For example,
    ``chunkers:company_manual`` resolves only entry points installed in the
    ``offline_rag.chunkers`` group. Python file paths and import strings are never
    accepted from configuration.
    """

    def __init__(
        self,
        registries: Mapping[str, ComponentRegistry[object]],
        *,
        entry_point_reader: EntryPointReader | None = None,
    ) -> None:
        unknown_kinds = set(registries).difference(PLUGIN_GROUPS)
        if unknown_kinds:
            raise ValueError(f"unknown registry kinds: {sorted(unknown_kinds)}")
        self._registries = dict(registries)
        self._entry_point_reader = entry_point_reader or _installed_entry_points

    def load_enabled(self, enabled: Sequence[str]) -> tuple[str, ...]:
        loaded: list[str] = []
        seen: set[str] = set()
        for identifier in enabled:
            kind, separator, name = identifier.strip().partition(":")
            normalized_name = name.strip().lower()
            normalized_identifier = f"{kind}:{normalized_name}"
            if not separator or not kind or not normalized_name:
                raise RegistryError(
                    f"invalid plugin identifier: {identifier!r}; expected kind:name"
                )
            if kind not in self._registries:
                raise RegistryError(
                    f"plugin kind is not enabled by this application: {kind}",
                    details={"kind": kind, "available_kinds": sorted(self._registries)},
                )
            if normalized_identifier in seen:
                raise RegistryError(f"plugin is enabled more than once: {normalized_identifier}")
            seen.add(normalized_identifier)

            group = PLUGIN_GROUPS[kind]
            matches = [
                entry_point
                for entry_point in self._entry_point_reader(group)
                if entry_point.name.strip().lower() == normalized_name
            ]
            if not matches:
                raise RegistryError(
                    f"enabled plugin is not installed: {normalized_identifier}",
                    details={"kind": kind, "name": normalized_name, "group": group},
                )
            if len(matches) > 1:
                values = sorted(entry_point.value for entry_point in matches)
                raise RegistryError(
                    f"multiple installed plugins use the same name: {normalized_identifier}",
                    details={"entry_points": values},
                )

            try:
                component = matches[0].load()
            except Exception as exc:
                raise RegistryError(
                    f"failed to load enabled plugin: {normalized_identifier}",
                    details={"entry_point": matches[0].value},
                ) from exc
            self._registries[kind].register(normalized_name, component)
            loaded.append(normalized_identifier)
        return tuple(loaded)

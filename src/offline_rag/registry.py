"""Thread-safe registries for named built-ins and trusted plugins."""

from __future__ import annotations

from collections.abc import Iterator
from threading import RLock
from typing import Generic, TypeVar

from offline_rag.exceptions import RegistryError

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    def __init__(self, component_kind: str) -> None:
        if not component_kind.strip():
            raise ValueError("component_kind must not be empty")
        self._component_kind = component_kind
        self._items: dict[str, T] = {}
        self._lock = RLock()

    @property
    def component_kind(self) -> str:
        return self._component_kind

    def register(self, name: str, component: T, *, replace: bool = False) -> None:
        normalized = self._normalize_name(name)
        with self._lock:
            if normalized in self._items and not replace:
                raise RegistryError(
                    f"{self._component_kind} component is already registered: {normalized}",
                    details={"component_kind": self._component_kind, "name": normalized},
                )
            self._items[normalized] = component

    def get(self, name: str) -> T:
        normalized = self._normalize_name(name)
        with self._lock:
            try:
                return self._items[normalized]
            except KeyError as exc:
                raise RegistryError(
                    f"unknown {self._component_kind} component: {normalized}",
                    details={
                        "component_kind": self._component_kind,
                        "name": normalized,
                        "available": sorted(self._items),
                    },
                ) from exc

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        with self._lock:
            return name.strip().lower() in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("component name must not be empty")
        return normalized

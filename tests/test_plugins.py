from __future__ import annotations

import unittest
from dataclasses import dataclass

from offline_rag.exceptions import RegistryError
from offline_rag.plugins import PLUGIN_GROUPS, PluginManager
from offline_rag.registry import ComponentRegistry


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    component: object | None = None
    error: Exception | None = None

    def load(self) -> object:
        if self.error:
            raise self.error
        return self.component


class PluginManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.component = object()
        self.entries = {
            PLUGIN_GROUPS["chunkers"]: (
                FakeEntryPoint("company_manual", "company_plugin:Chunker", self.component),
                FakeEntryPoint("not_enabled", "other_plugin:Chunker", object()),
            )
        }
        self.registry: ComponentRegistry[object] = ComponentRegistry("chunker")
        self.manager = PluginManager(
            {"chunkers": self.registry},
            entry_point_reader=lambda group: self.entries.get(group, ()),
        )

    def test_only_explicitly_enabled_plugin_is_loaded(self) -> None:
        loaded = self.manager.load_enabled(["chunkers:company_manual"])
        self.assertEqual(loaded, ("chunkers:company_manual",))
        self.assertIs(self.registry.get("company_manual"), self.component)
        self.assertNotIn("not_enabled", self.registry)

    def test_direct_import_or_file_syntax_is_rejected(self) -> None:
        with self.assertRaises(RegistryError):
            self.manager.load_enabled(["company_plugin:Chunker"])

    def test_missing_enabled_plugin_is_rejected(self) -> None:
        with self.assertRaisesRegex(RegistryError, "not installed"):
            self.manager.load_enabled(["chunkers:missing"])

    def test_duplicate_installed_name_is_rejected(self) -> None:
        entry = self.entries[PLUGIN_GROUPS["chunkers"]][0]
        self.entries[PLUGIN_GROUPS["chunkers"]] = (entry, entry)
        with self.assertRaisesRegex(RegistryError, "multiple installed plugins"):
            self.manager.load_enabled(["chunkers:company_manual"])

    def test_broken_plugin_is_wrapped_without_import_error_text(self) -> None:
        self.entries[PLUGIN_GROUPS["chunkers"]] = (
            FakeEntryPoint(
                "broken",
                "broken_plugin:Chunker",
                error=RuntimeError("document content must not leak"),
            ),
        )
        with self.assertRaises(RegistryError) as raised:
            self.manager.load_enabled(["chunkers:broken"])
        self.assertNotIn("document content", str(raised.exception))
        self.assertEqual(raised.exception.details["entry_point"], "broken_plugin:Chunker")

    def test_duplicate_enabled_identifier_is_rejected(self) -> None:
        with self.assertRaisesRegex(RegistryError, "more than once"):
            self.manager.load_enabled(["chunkers:company_manual", "chunkers:company_manual"])


if __name__ == "__main__":
    unittest.main()

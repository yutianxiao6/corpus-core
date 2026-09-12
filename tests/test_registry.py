from __future__ import annotations

import unittest

from corpuscore.exceptions import RegistryError
from corpuscore.registry import ComponentRegistry


class ComponentRegistryTests(unittest.TestCase):
    def test_register_and_lookup_use_normalized_names(self) -> None:
        registry = ComponentRegistry[object]("chunker")
        component = object()

        registry.register(" Recursive ", component)

        self.assertIs(registry.get("recursive"), component)
        self.assertEqual(registry.names(), ("recursive",))

    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ComponentRegistry[int]("loader")
        registry.register("text", 1)

        with self.assertRaises(RegistryError) as context:
            registry.register("TEXT", 2)

        self.assertEqual(context.exception.code, "registry_error")

    def test_unknown_component_lists_available_names(self) -> None:
        registry = ComponentRegistry[int]("organizer")
        registry.register("flat", 1)

        with self.assertRaises(RegistryError) as context:
            registry.get("context")

        self.assertEqual(context.exception.details["available"], ("flat",))

    def test_replace_is_explicit(self) -> None:
        registry = ComponentRegistry[int]("parser")
        registry.register("text", 1)
        registry.register("text", 2, replace=True)
        self.assertEqual(registry.get("text"), 2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from corpuscore.config import deep_merge, load_config, parse_cli_overrides
from corpuscore.exceptions import ConfigurationError


class ConfigurationTests(unittest.TestCase):
    def _write(self, directory: str, content: str) -> Path:
        path = Path(directory) / "corpus.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_deep_merge_replaces_sequences_and_preserves_nested_defaults(self) -> None:
        merged = deep_merge(
            {"runtime": {"offline": True, "log_level": "INFO"}, "items": [1]},
            {"runtime": {"log_level": "DEBUG"}, "items": [2]},
        )
        self.assertEqual(merged["runtime"], {"offline": True, "log_level": "DEBUG"})
        self.assertEqual(merged["items"], [2])

    def test_precedence_is_defaults_then_yaml_then_cli_then_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "runtime:\n  log_level: WARNING\n")
            config = load_config(
                path,
                cli_overrides={"runtime": {"log_level": "ERROR"}},
                api_overrides={"runtime": {"log_level": "DEBUG"}},
            )
        self.assertEqual(config.runtime.log_level, "DEBUG")
        self.assertTrue(config.runtime.offline)

    def test_filesystem_paths_are_relative_to_configuration_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write(
                directory,
                "runtime:\n  work_dir: ./state\n  cache_dir: ./cache\n"
                "embedding:\n  model_path: ./models/embedding\n"
                "vector_store:\n  path: ./vectors\n"
                "rerankers:\n  local:\n    model_path: ./models/reranker\n",
            )
            config = load_config(config_path)

        self.assertEqual(Path(config.runtime.work_dir), (root / "state").resolve())
        self.assertEqual(Path(config.runtime.cache_dir), (root / "cache").resolve())
        self.assertEqual(Path(config.embedding.model_path), (root / "models/embedding").resolve())
        self.assertEqual(Path(config.vector_store.path or ""), (root / "vectors").resolve())
        self.assertEqual(
            Path(config.rerankers["local"].model_path), (root / "models/reranker").resolve()
        )

    def test_embedding_checksum_must_be_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid = self._write(
                directory,
                f"embedding:\n  model_checksum: {'a' * 64}\n",
            )
            self.assertEqual(load_config(valid).embedding.model_checksum, "a" * 64)
            invalid = self._write(directory, "embedding:\n  model_checksum: short\n")
            with self.assertRaises(ConfigurationError):
                load_config(invalid)

    def test_default_chunk_profiles_include_structured_formats(self) -> None:
        config = load_config()
        self.assertEqual(config.chunk_profiles["table_rows"].type, "table_rows")
        self.assertEqual(config.chunk_profiles["table_rows"].max_rows_per_chunk, 1)
        self.assertEqual(config.chunk_profiles["page_aware"].type, "page_aware")

    def test_unknown_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "runtime:\n  log_lvel: INFO\n")
            with self.assertRaises(ConfigurationError) as raised:
                load_config(path)
        self.assertEqual(raised.exception.code, "configuration_error")
        self.assertIn("errors", raised.exception.details)

    def test_duplicate_yaml_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "version: 1\nversion: 1\n")
            with self.assertRaisesRegex(ConfigurationError, "duplicate YAML key"):
                load_config(path)

    def test_version_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "version: 2\n")
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_cli_overrides_parse_types_and_dotted_paths(self) -> None:
        overrides = parse_cli_overrides(
            ["embedding.batch_size=32", "runtime.log_level=DEBUG", "vector_store.on_disk=true"]
        )
        self.assertEqual(overrides["embedding"]["batch_size"], 32)
        self.assertEqual(overrides["runtime"]["log_level"], "DEBUG")
        self.assertIs(overrides["vector_store"]["on_disk"], True)

    def test_invalid_cross_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                "routing:\n  - match:\n      extensions: ['.md']\n    use: absent\n",
            )
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_hybrid_profile_requires_sparse_configuration_and_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_sparse = self._write(
                directory,
                "retrieval_profiles:\n  balanced:\n    strategy: hybrid\n    fusion:\n      type: rrf\n",
            )
            with self.assertRaises(ConfigurationError):
                load_config(missing_sparse)

            valid = self._write(
                directory,
                "sparse_embedding:\n  provider: hashed_lexical\n"
                "retrieval_profiles:\n  balanced:\n    strategy: hybrid\n"
                "    fusion:\n      type: rrf\n",
            )
            config = load_config(valid)
            self.assertEqual(config.retrieval_profiles["balanced"].strategy, "hybrid")

    def test_reranker_candidate_limits_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid = self._write(
                directory,
                "rerankers:\n  local:\n    model_path: ./model\n    maximum_candidates: 5\n"
                "retrieval_profiles:\n  precise:\n    strategy: dense\n"
                "    reranker: local\n    rerank_top_n: 4\n    final_k: 6\n",
            )
            with self.assertRaises(ConfigurationError):
                load_config(invalid)

    def test_qdrant_server_requires_explicit_http_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid = self._write(
                directory,
                "vector_store:\n  mode: server\n  url: http://qdrant.internal:6333\n"
                "  prefer_grpc: true\n  timeout_seconds: 15\n",
            )
            config = load_config(valid)
            self.assertEqual(config.vector_store.mode, "server")
            self.assertTrue(config.vector_store.prefer_grpc)

            invalid = self._write(directory, "vector_store:\n  mode: server\n  url: qdrant:6333\n")
            with self.assertRaises(ConfigurationError):
                load_config(invalid)

    def test_query_concurrency_limits_are_strictly_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid = self._write(
                directory,
                "query_concurrency:\n  queue_capacity: 32\n"
                "  embedding_microbatch_size: 8\n  embedding_wait_ms: 3\n"
                "  reranker_workers: 2\n",
            )
            config = load_config(valid)
            self.assertEqual(config.query_concurrency.queue_capacity, 32)
            self.assertEqual(config.query_concurrency.embedding_microbatch_size, 8)

            invalid = self._write(directory, "query_concurrency:\n  embedding_microbatch_size: 0\n")
            with self.assertRaises(ConfigurationError):
                load_config(invalid)


if __name__ == "__main__":
    unittest.main()

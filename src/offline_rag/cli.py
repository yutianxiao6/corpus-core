"""Command-line control plane for offline index operations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

from offline_rag import __version__
from offline_rag.config import load_config, parse_cli_overrides
from offline_rag.config.models import RagConfig
from offline_rag.contracts.common import JSONValue
from offline_rag.exceptions import OfflineRagError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-index",
        description="Build and inspect offline RAG retrieval indexes.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, help="versioned YAML configuration file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a dotted configuration key; may be repeated",
    )
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="create config and local directories")
    init_parser.add_argument("target", nargs="?", type=Path, default=Path.cwd())

    preview_parser = subparsers.add_parser("preview", help="parse and chunk without indexing")
    preview_parser.add_argument("inputs", nargs="+", type=Path)
    preview_parser.add_argument("--show-content", action="store_true")
    preview_parser.add_argument("--json", action="store_true")

    build_command = subparsers.add_parser("build", help="build the configured local index")
    build_command.add_argument("inputs", nargs="+", type=Path)
    build_command.add_argument("--json", action="store_true")

    sync_command = subparsers.add_parser("sync", help="incrementally synchronize an index")
    sync_command.add_argument("inputs", nargs="+", type=Path)
    sync_command.add_argument("--json", action="store_true")

    rebuild_command = subparsers.add_parser(
        "rebuild", help="build a staging collection and atomically activate it"
    )
    rebuild_command.add_argument("inputs", nargs="+", type=Path)
    rebuild_command.add_argument("--json", action="store_true")

    activate_command = subparsers.add_parser(
        "activate", help="activate a retained compatible collection"
    )
    activate_command.add_argument("collection_name")

    query_parser = subparsers.add_parser("query", help="run a retrieval-only debug query")
    query_parser.add_argument("query")
    query_parser.add_argument("--profile", default="fast")
    query_parser.add_argument("--organizer")
    query_parser.add_argument("--final-k", type=int)
    query_parser.add_argument("--score-threshold", type=float)
    query_parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="apply a payload filter; may be repeated",
    )
    query_parser.add_argument("--json", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate retrieval on a JSON fixture")
    evaluate_parser.add_argument("fixture", type=Path)
    evaluate_parser.add_argument("--profile", default="fast")
    evaluate_parser.add_argument("--k", type=int, default=5)
    evaluate_parser.add_argument("--min-recall", type=float)
    evaluate_parser.add_argument("--min-precision", type=float)
    evaluate_parser.add_argument("--min-mrr", type=float)
    evaluate_parser.add_argument("--min-ndcg", type=float)
    evaluate_parser.add_argument("--json", action="store_true")

    subparsers.add_parser("doctor", help="validate local configuration and resources")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "init":
            return _init(args.target)
        config = load_config(args.config, cli_overrides=parse_cli_overrides(args.set))
        if args.command == "preview":
            return _preview(config, args.inputs, show_content=args.show_content, as_json=args.json)
        if args.command == "build":
            return _index(config, args.inputs, operation="build", as_json=args.json)
        if args.command == "sync":
            return _index(config, args.inputs, operation="sync", as_json=args.json)
        if args.command == "rebuild":
            return _index(config, args.inputs, operation="rebuild", as_json=args.json)
        if args.command == "activate":
            return _activate(config, args.collection_name)
        if args.command == "query":
            return _query(
                config,
                args.query,
                profile=args.profile,
                organizer=args.organizer,
                final_k=args.final_k,
                score_threshold=args.score_threshold,
                filters=_parse_filters(args.filter),
                as_json=args.json,
            )
        if args.command == "evaluate":
            return _evaluate(
                config,
                args.fixture,
                profile=args.profile,
                k=args.k,
                minimums={
                    metric: value
                    for metric, value in {
                        "recall_at_k": args.min_recall,
                        "precision_at_k": args.min_precision,
                        "mrr_at_k": args.min_mrr,
                        "ndcg_at_k": args.min_ndcg,
                    }.items()
                    if value is not None
                },
                as_json=args.json,
            )
        if args.command == "doctor":
            return _doctor(config)
    except (OfflineRagError, OSError, TypeError, ValueError) as exc:
        code = exc.code if isinstance(exc, OfflineRagError) else "command_error"
        print(json.dumps({"error": code, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 2


def _init(target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    config_path = target / "rag.yaml"
    if config_path.exists():
        raise ValueError(f"configuration already exists: {config_path}")
    for directory in ("data", "documents", "models"):
        (target / directory).mkdir(exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(RagConfig().model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"created {config_path}")
    return 0


def _preview(
    config: RagConfig,
    inputs: list[Path],
    *,
    show_content: bool,
    as_json: bool,
) -> int:
    from offline_rag.ingestion import IngestionService

    report = IngestionService(config).preview(inputs)
    data: dict[str, object] = {
        "source_count": report.source_count,
        "successful_count": len(report.items),
        "failed_count": len(report.failures),
        "chunk_count": report.chunk_count,
        "items": [
            {
                "source": item.source.relative_path,
                "document_id": item.document.document_id,
                "chunk_count": len(item.chunks),
                **({"chunks": [chunk.content for chunk in item.chunks]} if show_content else {}),
            }
            for item in report.items
        ],
        "failures": [
            {"source": item.source_uri, "code": item.error_code, "message": item.message}
            for item in report.failures
        ],
    }
    _print_data(data, as_json=as_json)
    return 1 if report.failures else 0


def _index(
    config: RagConfig,
    inputs: list[Path],
    *,
    operation: str,
    as_json: bool,
) -> int:
    from offline_rag.engine import OfflineRagEngine

    with OfflineRagEngine.from_config(config) as engine:
        report = engine.sync(*inputs) if operation == "sync" else engine.rebuild(*inputs)
    data: dict[str, object] = {
        "job_id": report.job_id,
        "index_version": report.index_version,
        "indexed_count": report.indexed_count,
        "failed_count": report.failed_count,
        "chunk_count": report.chunk_count,
        "duration_ms": round(report.duration_ms, 3),
    }
    _print_data(data, as_json=as_json)
    return 1 if report.failed_count else 0


def _activate(config: RagConfig, collection_name: str) -> int:
    from offline_rag.engine import OfflineRagEngine

    with OfflineRagEngine.from_config(config) as engine:
        engine.activate(collection_name)
    print(f"activated: {collection_name}")
    return 0


def _query(
    config: RagConfig,
    query: str,
    *,
    profile: str,
    organizer: str | None,
    final_k: int | None,
    score_threshold: float | None,
    filters: Mapping[str, JSONValue],
    as_json: bool,
) -> int:
    from offline_rag.engine import OfflineRagEngine

    with OfflineRagEngine.from_config(config) as engine:
        result = engine.retrieve(
            query,
            profile=profile,
            organizer=organizer,
            final_k=final_k,
            score_threshold=score_threshold,
            filters=filters,
        )
    data: dict[str, object] = {
        "query": result.query,
        "index_version": result.index_version,
        "hits": [
            {
                "chunk_id": candidate.chunk.chunk_id,
                "score": candidate.final_score,
                "dense_score": candidate.dense_score,
                "sparse_score": candidate.sparse_score,
                "fusion_score": candidate.fusion_score,
                "rerank_score": candidate.rerank_score,
                "origins": candidate.origins,
                "source": candidate.chunk.source_uri,
                "content": candidate.chunk.content,
            }
            for candidate in result.hits
        ],
        "context": result.context,
        "citations": [
            {
                "id": citation.citation_id,
                "chunk_ids": list(citation.chunk_ids),
                "source": citation.source_uri,
                "page_start": citation.page_start,
                "page_end": citation.page_end,
            }
            for citation in result.citations
        ],
        "groups": [
            {
                "group_id": group.group_id,
                "chunk_ids": [candidate.chunk.chunk_id for candidate in group.hits],
            }
            for group in result.groups
        ],
        "debug": _plain_json(result.debug),
        "timings_ms": dict(result.timings_ms),
        "warnings": list(result.warnings),
    }
    _print_data(data, as_json=as_json)
    return 0


def _doctor(config: RagConfig) -> int:
    model_path = Path(config.embedding.model_path).expanduser()
    checks: list[dict[str, object]] = [
        {
            "name": "embedding_model",
            "ok": model_path.is_dir() and any(model_path.iterdir()),
            "detail": str(model_path),
        },
        {"name": "offline_mode", "ok": config.runtime.offline, "detail": "required"},
    ]
    if config.vector_store.mode == "local":
        vector_path = Path(config.vector_store.path or "").expanduser()
        parent = vector_path if vector_path.exists() else vector_path.parent
        checks.append(
            {
                "name": "vector_store_parent",
                "ok": parent.exists() and parent.is_dir(),
                "detail": str(parent),
            }
        )
    else:
        checks.append(
            {
                "name": "qdrant_server_url",
                "ok": bool(config.vector_store.url),
                "detail": config.vector_store.url or "missing",
            }
        )
        if config.vector_store.api_key_env:
            checks.append(
                {
                    "name": "qdrant_api_key_env",
                    "ok": bool(os.environ.get(config.vector_store.api_key_env)),
                    "detail": config.vector_store.api_key_env,
                }
            )
    checks.extend(
        {
            "name": f"reranker_model:{name}",
            "ok": (path := Path(reranker.model_path).expanduser()).is_dir() and any(path.iterdir()),
            "detail": str(path),
        }
        for name, reranker in config.rerankers.items()
    )
    ok = all(bool(item["ok"]) for item in checks)
    _print_data({"ok": ok, "checks": checks}, as_json=True)
    return 0 if ok else 1


def _evaluate(
    config: RagConfig,
    fixture: Path,
    *,
    profile: str,
    k: int,
    minimums: Mapping[str, float],
    as_json: bool,
) -> int:
    from offline_rag.engine import OfflineRagEngine
    from offline_rag.evaluation import RetrievalEvaluator, assert_thresholds, load_examples

    examples = load_examples(fixture)
    with OfflineRagEngine.from_config(config) as engine:
        report = RetrievalEvaluator(
            lambda query, limit: engine.retrieve(query, profile=profile, final_k=limit).hits
        ).evaluate(examples, k=k)
    assert_thresholds(report, minimums)
    _print_data(report.to_dict(), as_json=as_json)
    return 0


def _print_data(data: Mapping[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            print(f"{key}: {value}")


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _parse_filters(pairs: list[str]) -> dict[str, JSONValue]:
    filters: dict[str, JSONValue] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"invalid filter {pair!r}; expected FIELD=VALUE")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if not key or key in filters:
            raise ValueError(f"invalid or duplicate filter field: {key!r}")
        value = yaml.safe_load(raw)
        if isinstance(value, list):
            if not all(
                isinstance(item, (str, int)) and not isinstance(item, bool) for item in value
            ):
                raise ValueError(f"filter list for {key!r} must contain strings or integers")
            value = tuple(value)
        elif not isinstance(value, (str, int, bool)):
            raise TypeError(f"filter value for {key!r} must be a scalar")
        filters[key] = value
    return filters


if __name__ == "__main__":
    raise SystemExit(main())

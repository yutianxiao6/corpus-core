"""SQLite WAL journal for resumable and auditable ingestion jobs."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Self

from corpuscore.contracts.documents import SourceDescriptor
from corpuscore.contracts.indexing import IngestionStage


@dataclass(frozen=True, slots=True)
class IndexedSource:
    source_id: str
    source_uri: str
    relative_path: str | None
    content_hash: str
    document_id: str
    chunk_ids: tuple[str, ...]
    index_version: str


class IngestionJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    index_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    finished_at REAL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    source_id TEXT,
                    source_uri TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    error_code TEXT,
                    message TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    source_uri TEXT NOT NULL,
                    relative_path TEXT,
                    content_hash TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_ids_json TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS collection_sources (
                    collection_name TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    relative_path TEXT,
                    content_hash TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_ids_json TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(collection_name, source_id)
                );
                CREATE TABLE IF NOT EXISTS collection_snapshots (
                    collection_name TEXT PRIMARY KEY,
                    index_version TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_job_id_idx ON events(job_id);
                """
            )

    def start_job(self, index_version: str, *, job_id: str | None = None) -> str:
        identifier = job_id or f"job-{uuid.uuid4()}"
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO jobs(job_id, index_version, status, started_at) VALUES (?, ?, ?, ?)",
                (identifier, index_version, "running", time.time()),
            )
        return identifier

    def finish_job(self, job_id: str, *, status: str = "completed") -> None:
        if status not in {"completed", "completed_with_errors", "failed", "interrupted"}:
            raise ValueError("invalid terminal job status")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE jobs SET status = ?, finished_at = ? WHERE job_id = ? AND status = 'running'",
                (status, time.time(), job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"running journal job does not exist: {job_id}")

    def record_event(
        self,
        job_id: str,
        *,
        source_uri: str,
        stage: IngestionStage,
        source_id: str | None = None,
        error_code: str | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO events(
                    job_id, source_id, source_uri, stage, error_code, message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, source_id, source_uri, stage.value, error_code, message, time.time()),
            )

    def record_indexed(
        self,
        job_id: str,
        source: SourceDescriptor,
        *,
        document_id: str,
        chunk_ids: Sequence[str],
        index_version: str,
    ) -> None:
        encoded_ids = json.dumps(list(chunk_ids), separators=(",", ":"))
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO sources(
                    source_id, source_uri, relative_path, content_hash, document_id,
                    chunk_ids_json, index_version, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    source_uri = excluded.source_uri,
                    relative_path = excluded.relative_path,
                    content_hash = excluded.content_hash,
                    document_id = excluded.document_id,
                    chunk_ids_json = excluded.chunk_ids_json,
                    index_version = excluded.index_version,
                    updated_at = excluded.updated_at
                """,
                (
                    source.source_id,
                    source.uri,
                    source.relative_path,
                    source.content_hash,
                    document_id,
                    encoded_ids,
                    index_version,
                    time.time(),
                ),
            )
            self._connection.execute(
                """
                INSERT INTO events(job_id, source_id, source_uri, stage, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, source.source_id, source.uri, IngestionStage.INDEXED.value, time.time()),
            )

    def get_source(self, source_id: str) -> IndexedSource | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sources WHERE source_id = ?", (source_id,)
            ).fetchone()
        return self._decode_source(row) if row is not None else None

    def list_sources(self) -> tuple[IndexedSource, ...]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM sources ORDER BY source_id").fetchall()
        return tuple(self._decode_source(row) for row in rows)

    def remove_source(self, source_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))

    def replace_sources(self, sources: Sequence[IndexedSource]) -> None:
        """Atomically replace the active-index source snapshot after alias activation."""

        with self._lock, self._connection:
            self._connection.execute("DELETE FROM sources")
            self._connection.executemany(
                """
                INSERT INTO sources(
                    source_id, source_uri, relative_path, content_hash, document_id,
                    chunk_ids_json, index_version, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        source.source_id,
                        source.source_uri,
                        source.relative_path,
                        source.content_hash,
                        source.document_id,
                        json.dumps(list(source.chunk_ids), separators=(",", ":")),
                        source.index_version,
                        time.time(),
                    )
                    for source in sources
                ],
            )

    def save_collection_snapshot(
        self, collection_name: str, sources: Sequence[IndexedSource]
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        with self._lock, self._connection:
            index_version = sources[0].index_version if sources else ""
            self._connection.execute(
                """
                INSERT INTO collection_snapshots(collection_name, index_version, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(collection_name) DO UPDATE SET
                    index_version = excluded.index_version,
                    created_at = excluded.created_at
                """,
                (collection_name, index_version, time.time()),
            )
            self._connection.execute(
                "DELETE FROM collection_sources WHERE collection_name = ?",
                (collection_name,),
            )
            self._connection.executemany(
                """
                INSERT INTO collection_sources(
                    collection_name, source_id, source_uri, relative_path, content_hash,
                    document_id, chunk_ids_json, index_version, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        collection_name,
                        source.source_id,
                        source.source_uri,
                        source.relative_path,
                        source.content_hash,
                        source.document_id,
                        json.dumps(list(source.chunk_ids), separators=(",", ":")),
                        source.index_version,
                        time.time(),
                    )
                    for source in sources
                ],
            )

    def collection_snapshot(self, collection_name: str) -> tuple[IndexedSource, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT source_id, source_uri, relative_path, content_hash, document_id,
                       chunk_ids_json, index_version
                FROM collection_sources
                WHERE collection_name = ?
                ORDER BY source_id
                """,
                (collection_name,),
            ).fetchall()
        return tuple(self._decode_source(row) for row in rows)

    def has_collection_snapshot(self, collection_name: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM collection_snapshots WHERE collection_name = ?",
                (collection_name,),
            ).fetchone()
        return row is not None

    def recover_interrupted_jobs(self) -> tuple[str, ...]:
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT job_id FROM jobs WHERE status = 'running' ORDER BY started_at"
            ).fetchall()
            identifiers = tuple(str(row["job_id"]) for row in rows)
            if identifiers:
                self._connection.execute(
                    "UPDATE jobs SET status = 'interrupted', finished_at = ? WHERE status = 'running'",
                    (time.time(),),
                )
        return identifiers

    def job_status(self, job_id: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return str(row["status"]) if row else None

    @staticmethod
    def _decode_source(row: sqlite3.Row) -> IndexedSource:
        values = json.loads(str(row["chunk_ids_json"]))
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("journal contains invalid chunk IDs")
        return IndexedSource(
            source_id=str(row["source_id"]),
            source_uri=str(row["source_uri"]),
            relative_path=(str(row["relative_path"]) if row["relative_path"] is not None else None),
            content_hash=str(row["content_hash"]),
            document_id=str(row["document_id"]),
            chunk_ids=tuple(values),
            index_version=str(row["index_version"]),
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from corpuscore.contracts.documents import SourceDescriptor
from corpuscore.contracts.indexing import IngestionStage
from corpuscore.indexing import IngestionJournal, plan_sync


def source(identifier: str, content_hash: str) -> SourceDescriptor:
    return SourceDescriptor(
        source_id=identifier,
        uri=f"file:///{identifier}.txt",
        relative_path=f"{identifier}.txt",
        content_hash=content_hash,
    )


class JournalTests(unittest.TestCase):
    def test_indexed_source_persists_across_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.sqlite3"
            with IngestionJournal(path) as journal:
                job_id = journal.start_job("index-v1")
                journal.record_event(
                    job_id,
                    source_uri="file:///one.txt",
                    source_id="one",
                    stage=IngestionStage.PARSED,
                )
                journal.record_indexed(
                    job_id,
                    source("one", "hash-1"),
                    document_id="document-1",
                    chunk_ids=("chunk-1", "chunk-2"),
                    index_version="index-v1",
                )
                journal.finish_job(job_id)
                self.assertEqual(journal.job_status(job_id), "completed")

            with IngestionJournal(path) as reopened:
                indexed = reopened.get_source("one")

        assert indexed is not None
        self.assertEqual(indexed.chunk_ids, ("chunk-1", "chunk-2"))
        self.assertEqual(indexed.document_id, "document-1")

    def test_running_jobs_are_recovered_as_interrupted(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            IngestionJournal(Path(directory) / "journal.sqlite3") as journal,
        ):
            first = journal.start_job("v1", job_id="first")
            second = journal.start_job("v1", job_id="second")
            recovered = journal.recover_interrupted_jobs()

            self.assertEqual(recovered, (first, second))
            self.assertEqual(journal.job_status(first), "interrupted")
            self.assertEqual(journal.job_status(second), "interrupted")
            self.assertEqual(journal.recover_interrupted_jobs(), ())

    def test_invalid_job_transition_is_rejected(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            IngestionJournal(Path(directory) / "journal.sqlite3") as journal,
        ):
            job_id = journal.start_job("v1")
            journal.finish_job(job_id)
            with self.assertRaises(ValueError):
                journal.finish_job(job_id)

    def test_replace_sources_is_atomic_snapshot_update(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            IngestionJournal(Path(directory) / "journal.sqlite3") as journal,
        ):
            job_id = journal.start_job("v1")
            journal.record_indexed(
                job_id,
                source("old", "old-hash"),
                document_id="old-document",
                chunk_ids=("old-chunk",),
                index_version="v1",
            )
            old = journal.list_sources()[0]
            journal.replace_sources(
                (
                    type(old)(
                        source_id="new",
                        source_uri="file:///new.txt",
                        relative_path="new.txt",
                        content_hash="new-hash",
                        document_id="new-document",
                        chunk_ids=("new-chunk",),
                        index_version="v2",
                    ),
                )
            )
            current = journal.list_sources()
            journal.save_collection_snapshot("documents_v2", current)
            snapshot = journal.collection_snapshot("documents_v2")
            journal.save_collection_snapshot("empty", ())
            has_empty = journal.has_collection_snapshot("empty")
            empty_snapshot = journal.collection_snapshot("empty")

        self.assertEqual([item.source_id for item in current], ["new"])
        self.assertEqual(snapshot, current)
        self.assertTrue(has_empty)
        self.assertEqual(empty_snapshot, ())


class SyncPlannerTests(unittest.TestCase):
    def test_plan_classifies_new_modified_unchanged_and_missing(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            IngestionJournal(Path(directory) / "journal.sqlite3") as journal,
        ):
            job_id = journal.start_job("v1")
            for item in (
                source("modified", "old"),
                source("unchanged", "same"),
                source("missing", "gone"),
            ):
                journal.record_indexed(
                    job_id,
                    item,
                    document_id=f"doc-{item.source_id}",
                    chunk_ids=(f"chunk-{item.source_id}",),
                    index_version="v1",
                )
            journal.finish_job(job_id)
            plan = plan_sync(
                (
                    source("new", "new"),
                    source("modified", "changed"),
                    source("unchanged", "same"),
                ),
                journal.list_sources(),
            )

        self.assertEqual([item.source_id for item in plan.new], ["new"])
        self.assertEqual([item[0].source_id for item in plan.modified], ["modified"])
        self.assertEqual([item[0].source_id for item in plan.unchanged], ["unchanged"])
        self.assertEqual([item.source_id for item in plan.missing], ["missing"])


if __name__ == "__main__":
    unittest.main()

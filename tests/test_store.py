from __future__ import annotations

import pytest

from semantic_librarian import Document
from semantic_librarian.store import DuplicateDocumentError, LibraryStore


def _docs():
    return [
        Document(
            "a",
            {"title": "Attention", "abstract": "Selective attention to colour."},
            {"year": 2001},
            {"author": ["Smith, J.", "Jones, K."], "year": ["2001"]},
        ),
        Document(
            "b",
            {"title": "Memory", "abstract": "Episodic memory for words."},
            {"year": 2005},
            {"author": ["Jones, K."]},
        ),
    ]


def test_roundtrip(tmp_path):
    with LibraryStore(tmp_path / "db.sqlite") as store:
        assert store.add_documents(_docs()) == 2
        assert store.count_documents() == 2
        docs = list(store.documents())
        assert [d.id for d in docs] == ["a", "b"]
        assert docs[0].factors == {"author": ["Smith, J.", "Jones, K."], "year": ["2001"]}
        assert docs[0].meta == {"year": 2001}
        assert store.get_document("b").fields["title"] == "Memory"
        assert store.get_document("zzz") is None
        assert store.factor_names() == ["author", "year"]


def test_duplicates_roll_back(tmp_path):
    with LibraryStore(tmp_path / "db.sqlite") as store:
        store.add_documents(_docs()[:1])
        with pytest.raises(DuplicateDocumentError):
            store.add_documents([Document("c", {"t": "x"}), Document("a", {"t": "y"})])
        assert [d.id for d in store.documents()] == ["a"]


def test_replace_source_removes_previous_documents(tmp_path):
    with LibraryStore(tmp_path / "db.sqlite") as store:
        sid = store.replace_source("/x.json", "json", "sha1")
        store.add_documents(_docs(), sid)
        assert store.source_sha("/x.json") == "sha1"
        sid = store.replace_source("/x.json", "json", "sha2")
        store.add_documents(_docs()[1:], sid)
        assert [d.id for d in store.documents()] == ["b"]
        assert store.sources()[0]["documents"] == 1
        assert [h[0] for h in store.lexical_search("attention")] == []


def test_lexical_search_and_builds(tmp_path):
    with LibraryStore(tmp_path / "db.sqlite") as store:
        store.add_documents(_docs())
        hits = store.lexical_search('episodic "memory"')
        assert hits[0][0] == "b"
        assert store.lexical_search("   ") == []
        store.record_build("beagle", {"dim": 8}, 2, 10)
        assert store.builds()[0]["config"] == {"dim": 8}


def test_reopen_keeps_data(tmp_path):
    path = tmp_path / "db.sqlite"
    with LibraryStore(path) as store:
        store.add_documents(_docs())
    with LibraryStore(path) as store:
        assert store.count_documents() == 2
        assert store.get_meta("schema_version") == "2"

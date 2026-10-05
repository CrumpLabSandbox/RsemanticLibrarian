from __future__ import annotations

import json

import pytest

from semantic_librarian.ingest import FieldMapping, detect_format, read_documents


def test_json_list_with_auto_fields(tmp_path):
    path = tmp_path / "papers.json"
    path.write_text(
        json.dumps(
            [
                {
                    "doi": "10.1/a",
                    "title": "A",
                    "abstract": "Alpha text.",
                    "authors": ["X, Y", "Z, W"],
                    "year": 1999,
                    "journal": "J",
                },
                {
                    "title": "B",
                    "abstract": "Beta text.",
                    "authorlist": "P, Q; R, S",
                    "year": "2001",
                },
            ]
        )
    )
    docs = list(read_documents(path))
    assert [d.id for d in docs] == ["10.1/a", "papers-000002"]
    assert docs[0].fields == {"title": "A", "abstract": "Alpha text."}
    assert docs[0].factors == {"author": ["X, Y", "Z, W"], "year": ["1999"]}
    assert docs[1].factors["author"] == ["P, Q", "R, S"]
    assert docs[0].meta["journal"] == "J"


def test_json_wrapper_and_explicit_mapping(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            {
                "documents": [
                    {"key": 7, "body": "Some text", "speaker": "Alice", "tags": "a|b"},
                ]
            }
        )
    )
    mapping = FieldMapping.from_options(
        text=["body"], factors=["who=speaker", "tag=tags:|"], id="key"
    )
    (doc,) = read_documents(path, mapping=mapping)
    assert doc.id == "7"
    assert doc.factors == {"who": ["Alice"], "tag": ["a", "b"]}


def test_jsonl_and_csv(tmp_path):
    jl = tmp_path / "x.jsonl"
    jl.write_text('{"text": "one"}\n\n{"text": "two"}\n')
    assert [d.fields["text"] for d in read_documents(jl)] == ["one", "two"]

    csv_path = tmp_path / "x.csv"
    csv_path.write_text(
        "id,title,abstract,author,year\n"
        '1,T1,"First, abstract.",Doe J.; Roe K.,2010\n'
        "2,T2,,Doe J.,\n"
    )
    docs = list(read_documents(csv_path))
    assert docs[0].factors == {"author": ["Doe J.", "Roe K."], "year": ["2010"]}
    assert docs[1].fields == {"title": "T2"} and "year" not in docs[1].factors


def test_missing_text_fields_error(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps([{"foo": "bar"}]))
    with pytest.raises(ValueError, match="no text fields"):
        list(read_documents(path))


def test_bibtex(crump_bib):
    docs = list(read_documents(crump_bib))
    assert len(docs) == 39
    first = docs[0]
    assert first.id == "loganHierarchicalControlCognitive2011"
    assert first.fields["title"] == (
        "Hierarchical Control of Cognitive Processes: The Case for Skilled Typewriting"
    )
    assert first.factors["author"] == ["Logan, Gordon D.", "Crump, Matthew J. C."]
    assert first.factors["year"] == ["2011"]
    assert "{" not in "".join(first.fields.values())


def test_text_splits(tmp_path):
    path = tmp_path / "alice.txt"
    path.write_text(
        "Alice was beginning to get very tired. She had nothing to do!\n\n"
        "So she was considering in her own mind.\n"
    )
    assert len(list(read_documents(path))) == 1
    paras = list(read_documents(path, split="paragraph"))
    assert [d.id for d in paras] == ["alice-p000001", "alice-p000002"]
    sents = list(read_documents(path, split="sentence"))
    assert len(sents) == 3 and sents[0].factors == {"source": ["alice"]}


def test_detect_format(tmp_path):
    assert detect_format(tmp_path / "a.BIB") == "bibtex"
    assert detect_format(tmp_path / "a.ndjson") == "jsonl"
    assert detect_format(tmp_path / "a.PDF") == "pdf"
    with pytest.raises(ValueError):
        detect_format(tmp_path / "a.xyz")

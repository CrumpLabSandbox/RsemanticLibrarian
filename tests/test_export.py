from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from semantic_librarian import Chunk, Document, Library, LibraryError
from semantic_librarian.cli import main
from semantic_librarian.export import export_library, make_server
from semantic_librarian.text.clean import TextProcessor

RANK_SCRIPT = Path(__file__).parent.parent / "web" / "test" / "rank.mjs"
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    lib = Library.create(root / "lib", embedder_params={"dim": 256})
    lib.add_file(Path(__file__).parent / "fixtures" / "crump.bib")
    lib.add_documents(
        [
            Document(
                "notes",
                {"title": "Lab notes"},
                {"doi": "10.1/x"},
                {"author": ["Crump, Matthew J. C."]},
                [
                    Chunk(
                        "Typists learn keyboard letters by practice.", {"page": 1, "page_end": 1}
                    ),
                    Chunk("Attention and memory guide skilled typing.", {"page": 2, "page_end": 3}),
                ],
            )
        ]
    )
    lib.build()
    yield lib
    lib.close()


def load(out: Path, entry: dict) -> np.ndarray:
    raw = np.fromfile(out / "data" / entry["vectors"], dtype="int8" if entry["scale"] else "<f4")
    vectors = raw.reshape(entry["n"], entry["dim"]).astype(np.float64)
    if entry["scale"]:
        vectors *= np.fromfile(out / "data" / entry["scale"], dtype="<f4")[:, None]
    return vectors


def test_export_contents(built, tmp_path):
    out = tmp_path / "site"
    report = export_library(built, out)
    assert (out / "index.html").exists() and (out / "assets" / "app.js").exists()
    manifest = json.loads((out / "data" / "manifest.json").read_text())
    names = [s["name"] for s in manifest["spaces"]]
    assert names == list(built.spaces) and report.spaces == built.build_info["spaces"]
    assert manifest["n_documents"] == 40 and manifest["n_chunks"] == 2
    assert manifest["text_mode"] == "unicode" and "the" in manifest["stopwords"]
    by_name = {s["name"]: s for s in manifest["spaces"]}
    assert by_name["author"]["kind"] == "factor" and by_name["chunk"]["kind"] == "chunk"

    # 8-bit vectors keep directions (cosines) and, through the scales, lengths
    for name, space in built.spaces.items():
        stored = load(out, by_name[name])
        exact = np.asarray(space.vectors, dtype=np.float64)
        cosines = (stored * exact).sum(1) / (
            np.linalg.norm(stored, axis=1) * np.linalg.norm(exact, axis=1)
        )
        assert cosines.min() > 0.999
        np.testing.assert_allclose(
            np.linalg.norm(stored, axis=1), np.linalg.norm(exact, axis=1), rtol=0.01
        )
        labels = json.loads((out / "data" / by_name[name]["labels"]).read_text())
        assert labels == space.labels

    documents = json.loads((out / "data" / "documents.json").read_text())
    row = built.space("document").row("notes")
    assert documents["title"][row] == "Lab notes" and documents["year"][row] is None
    authors = built.space("author").labels
    assert [authors[i] for i in documents["factors"]["author"][row]] == ["Crump, Matthew J. C."]
    first = built.documents[0]
    assert documents["year"][0] == first.factors["year"][0]
    record = json.loads((out / "data" / "docs" / "00000.json").read_text())[row]
    assert record["id"] == "notes" and record["chunks"] == [0, 2]
    assert record["meta"] == {"doi": "10.1/x"}
    chunks = json.loads((out / "data" / "chunks.json").read_text())
    assert chunks == {"doc": [row, row], "pages": [[1, 1], [2, 3]]}
    texts = json.loads((out / "data" / "chunks" / "00000.json").read_text())
    assert texts[1] == "Attention and memory guide skilled typing."


def test_export_float32_and_overwrite_rules(built, tmp_path):
    out = tmp_path / "site"
    export_library(built, out, precision="float32")
    manifest = json.loads((out / "data" / "manifest.json").read_text())
    entry = manifest["spaces"][0]
    assert entry["dtype"] == "float32" and entry["scale"] is None
    np.testing.assert_array_equal(
        load(out, entry), np.asarray(built.space(entry["name"]).vectors, dtype=np.float64)
    )
    # an earlier export is replaced, including files the new one does not write
    stray = out / "data" / "old.bin"
    stray.write_bytes(b"x")
    export_library(built, out)
    assert not stray.exists()
    assert json.loads((out / "data" / "manifest.json").read_text())["spaces"][0]["scale"]
    # anything else is left alone
    other = tmp_path / "other"
    other.mkdir()
    (other / "thesis.txt").write_text("precious")
    with pytest.raises(LibraryError, match="not an earlier export"):
        export_library(built, other)
    assert (other / "thesis.txt").read_text() == "precious"
    with pytest.raises(LibraryError, match="is a file"):
        export_library(built, other / "thesis.txt")
    with pytest.raises(ValueError):
        export_library(built, tmp_path / "x", precision="float16")  # type: ignore[arg-type]


def test_export_needs_a_current_build(tmp_path, crump_bib):
    with Library.create(tmp_path / "lib", embedder_params={"dim": 64}) as lib:
        lib.add_file(crump_bib)
        with pytest.raises(LibraryError, match="not been built"):
            export_library(lib, tmp_path / "site")
        lib.build()
        lib.add_documents([Document("extra", {"text": "one more document"})])
        with pytest.raises(LibraryError, match="sl build"):
            export_library(lib, tmp_path / "site")


def test_cli_export_and_server(built, tmp_path, capsys):
    out = tmp_path / "site"
    assert main(["export", str(built.root), str(out), "-q"]) == 0
    assert "exported to" in capsys.readouterr().out
    server = make_server(out, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(f"{base}/data/manifest.json") as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert json.load(response)["name"] == "lib"
        with urllib.request.urlopen(f"{base}/") as response:
            assert b"<html" in response.read()
    finally:
        server.shutdown()
        server.server_close()


def test_cli_serve_exports_once(built, monkeypatch, capsys):
    served = []
    monkeypatch.setattr(
        "semantic_librarian.export.serve",
        lambda directory, port, open_browser: served.append((directory, port, open_browser)),
    )
    assert main(["serve", str(built.root), "--port", "8123"]) == 0
    assert "exported to" in capsys.readouterr().out
    assert served == [(built.root / "web", 8123, False)]
    assert (built.root / "web" / "data" / "manifest.json").exists()
    # nothing changed, so the second run serves the same export
    assert main(["serve", str(built.root), "--open"]) == 0
    assert "exported to" not in capsys.readouterr().out
    assert main(["serve", str(built.root), "--precision", "float32"]) == 0
    assert "exported to" in capsys.readouterr().out
    assert served[1][2] is True


def run_node(out: Path, requests: list[dict], tmp_path: Path) -> list:
    request_file = tmp_path / "requests.json"
    request_file.write_text(json.dumps(requests))
    done = subprocess.run(
        [NODE, str(RANK_SCRIPT), str(out / "data"), str(request_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


QUERIES = [
    ("attention and memory in skilled typing", "compound", "document"),
    ("keyboard typing practice", "and", "document"),
    ("perception attention memory", "or", "document"),
    ("typing", "compound", "word"),
    ("the stroop task and zzzunknown", "compound", "author"),
    ("skilled typing keyboard", "compound", "chunk"),
    ("of the and", "compound", "document"),
]
SIMILAR = [
    ("Crump, Matthew J. C.", "author", "author"),
    ("Crump, Matthew J. C.", "author", "document"),
    ("crumpWarningThisKeyboard2010", "document", "author"),
    ("notes#2", "chunk", "document"),
    ("typing", "word", "word"),
]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_search_matches_python(built, tmp_path):
    """The JavaScript that runs in the browser ranks exactly as ``Library`` does."""
    out = tmp_path / "site"
    export_library(built, out, precision="float32")
    requests = [{"query": q, "mode": m, "target": t, "k": 15} for q, m, t in QUERIES]
    requests += [{"item": i, "space": s, "target": t, "k": 15} for i, s, t in SIMILAR]
    results = run_node(out, requests, tmp_path)
    for (query, mode, target), got in zip(QUERIES, results, strict=False):
        want = built.search(query, mode=mode, space=target, k=15)  # type: ignore[arg-type]
        assert got["terms"] == want.terms and got["unknown"] == want.unknown
        assert got["labels"] == [h.label for h in want.hits], query
        np.testing.assert_allclose(got["scores"], [h.score for h in want.hits], atol=1e-9)
    for (item, space, target), got in zip(SIMILAR, results[len(QUERIES) :], strict=True):
        want_hits = built.similar(item, space=space, target=target, k=15)
        assert got["labels"] == [h.label for h in want_hits], item
        np.testing.assert_allclose(got["scores"], [h.score for h in want_hits], atol=1e-9)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_search_with_8_bit_vectors(built, tmp_path):
    out = tmp_path / "site"
    export_library(built, out)
    requests = [{"query": q, "mode": m, "target": t, "k": 10} for q, m, t in QUERIES[:3]]
    for (query, mode, target), got in zip(QUERIES, run_node(out, requests, tmp_path), strict=False):
        want = built.search(query, mode=mode, space=target, k=10)  # type: ignore[arg-type]
        scores = {h.label: h.score for h in built.search(query, mode=mode, k=None).hits}
        assert got["labels"][0] == want.hits[0].label
        assert len(set(got["labels"]) & {h.label for h in want.hits}) >= 9
        np.testing.assert_allclose(got["scores"], [scores[x] for x in got["labels"]], atol=0.01)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_tokenizer_and_map(built, tmp_path):
    out = tmp_path / "site"
    export_library(built, out, precision="float32")
    texts = [
        "Don't split naïve N400 effects; 3.14 isn't a word (e.g., Straße, ŒUVRE)!",
        "Über den Muskelsinn — P300/N400, well-known co-operation_x 2nd",
        "日本語 and Ελληνικά ǅ ﬁnger",
    ]
    requests = [{"tokenize": t, "mode": m} for t in texts for m in ("unicode", "legacy")]
    requests.append({"query": "attention memory typing", "target": "document", "k": 12, "map": 3})
    *tokens, mapped = run_node(out, requests, tmp_path)
    expected = [TextProcessor(m).tokens(t) for t in texts for m in ("unicode", "legacy")]  # type: ignore[arg-type]
    assert tokens == expected
    # the map reproduces Python's classical MDS up to mirroring
    want = built.project(mapped["labels"], n_clusters=3)
    got = np.array(mapped["coords"])
    signs = np.sign((got * want.coords).sum(axis=0))
    np.testing.assert_allclose(got * signs, want.coords, atol=1e-6)
    assert len(set(mapped["clusters"])) == 3 and len(mapped["clusters"]) == 12

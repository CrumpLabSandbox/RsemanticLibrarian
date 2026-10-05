from __future__ import annotations

import numpy as np
import pytest

from semantic_librarian import Document, Library, LibraryError
from semantic_librarian.cli import main
from semantic_librarian.ingest import read_documents
from semantic_librarian.ingest.pdf import (
    Block,
    dehyphenate,
    largest_text,
    paragraphs_from_blocks,
    plausible_title,
    split_authors,
    strip_headers_footers,
)
from semantic_librarian.store.sqlite import LibraryStore
from semantic_librarian.text.chunk import Paragraph, chunk_paragraphs

pymupdf = pytest.importorskip("pymupdf")

TYPING = (
    "Skilled typists move their fingers quickly across the keyboard. Typing speed depends "
    "on practice with letters, words and keystrokes. Expert typists rarely look at the "
    "keyboard while their fingers strike the keys in a fluent sequence."
)
MEMORY = (
    "Recognition memory improves when items are studied repeatedly. Participants recall "
    "more words from a studied list after rehearsal. Forgetting follows a curve in which "
    "recall declines rapidly and then slowly across retention intervals."
)
ATTENTION = (
    "Selective attention filters distracting information in the Stroop task. Conflict "
    "between a color and a word slows responses. Attention control adapts to the "
    "proportion of congruent trials experienced in a context."
)


def make_pdf(path, pages, title="Typing, Memory and Attention", header=True, metadata=None):
    """Write a PDF with a large title, body paragraphs, a running header and page numbers."""
    doc = pymupdf.open()
    for number, paragraphs in enumerate(pages, start=1):
        page = doc.new_page()
        if header:
            page.insert_text((72, 40), f"Journal of Testing, 2010, Vol. {number}", fontsize=8)
            page.insert_text((290, 815), str(number), fontsize=8)
        y = 100.0
        if number == 1 and title:
            page.insert_text((72, y), title, fontsize=20)
            y += 40
        for text in paragraphs:
            box = pymupdf.Rect(72, y, 520, y + 110)
            assert page.insert_textbox(box, text, fontsize=10) >= 0
            y += 130
    if metadata:
        doc.set_metadata(metadata)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture()
def paper(tmp_path):
    return make_pdf(
        tmp_path / "paper.pdf",
        [
            [TYPING, "The efficient finger move-\nment was studied and it continues on the"],
            ["next page without a break. " + MEMORY, MEMORY],
            [ATTENTION, ATTENTION],
            [TYPING, MEMORY],
        ],
        metadata={"title": "Microsoft Word - draft7.docx", "author": "Crump, M.; Logan, G. D."},
    )


# -- cleaning ----------------------------------------------------------------------------


def test_dehyphenate():
    assert dehyphenate("finger move-\nment was\nstudied") == "finger movement was studied"
    assert dehyphenate("the Stroop-\nSimon task") == "the Stroop-Simon task"
    assert dehyphenate("efﬁcient co­operation") == "efficient cooperation"
    assert dehyphenate("ages 5-\n7 years") == "ages 5- 7 years"


def test_strip_headers_footers():
    def page(n, body, top=0.03):
        return [
            Block(n, f"Smith & Jones {n}", top, top + 0.02),
            Block(n, body, 0.3, 0.6),
            Block(n, f"Page {n} of 4", 0.95, 0.97),
        ]

    pages = [page(n, f"Body text {n}.") for n in range(1, 5)]
    pages[0].append(Block(1, "Received May 2010", 0.92, 0.94))
    pages[1].append(Block(2, "Smith & Jones 7", 0.4, 0.5))
    kept = strip_headers_footers(pages)
    assert [b.text for b in kept[0]] == ["Body text 1.", "Received May 2010"]
    assert [b.text for b in kept[1]] == ["Body text 2.", "Smith & Jones 7"]
    # a lone page number goes even on a one-page file; other margin text stays
    single = strip_headers_footers([[Block(1, "A note", 0.02, 0.04), Block(1, "12", 0.95, 0.97)]])
    assert [b.text for b in single[0]] == ["A note"]


def test_paragraphs_merge_across_breaks():
    pages = [
        [Block(1, "Method", 0.2, 0.25), Block(1, "Typists were exam-", 0.3, 0.4)],
        [Block(2, "ined while typing a\nparagraph that ran", 0.1, 0.2), Block(2, " \n", 0.2, 0.3)],
        [Block(3, "onto a third page.", 0.1, 0.2), Block(3, "another one.", 0.3, 0.4)],
    ]
    paragraphs = paragraphs_from_blocks(pages)
    assert [(p.text, p.page, p.page_end) for p in paragraphs] == [
        ("Method", 1, 1),
        ("Typists were examined while typing a paragraph that ran onto a third page.", 1, 3),
        ("another one.", 3, 3),
    ]


def test_chunks_partition_the_text():
    paragraphs = [Paragraph(f"w{i} " * 30 + "end.", i // 3 + 1, i // 3 + 1) for i in range(10)]
    paragraphs.append(Paragraph("tail words", 9, 9))
    chunks = chunk_paragraphs(paragraphs, target_words=100)
    words = [w for c in chunks for w in c.text.split()]
    assert words == [w for p in paragraphs for w in p.text.split()]
    assert [len(c.text.split()) for c in chunks] == [124, 124, 64]
    assert chunks[0].meta == {"page": 1, "page_end": 2}
    assert chunks[-1].meta == {"page": 3, "page_end": 9}


def test_long_paragraphs_are_cut_at_sentences():
    long = " ".join(f"Sentence number {i} has exactly seven words." for i in range(40))
    chunks = chunk_paragraphs([Paragraph(long, 1, 1)], target_words=50)
    assert all(c.text.rstrip().endswith(".") for c in chunks)
    assert max(len(c.text.split()) for c in chunks) <= 75
    assert sum(len(c.text.split()) for c in chunks) == 280
    unbroken = chunk_paragraphs([Paragraph("word " * 500, 1, 1)], target_words=100)
    assert [len(c.text.split()) for c in unbroken] == [150, 150, 150, 50]
    assert chunk_paragraphs([]) == []
    with pytest.raises(ValueError):
        chunk_paragraphs([], target_words=0)


def test_title_and_author_helpers():
    assert plausible_title("  Skilled   typing ") == "Skilled typing"
    for junk in (None, "", "abc", "paper.pdf", "Microsoft Word - x.docx", "untitled"):
        assert plausible_title(junk) is None
    spans = [(8.0, "Journal"), (20.0, "A Big"), (19.8, "Title"), (10.0, "Body text"), (30.0, "1")]
    assert largest_text(spans) == "A Big Title"
    assert largest_text([(30.0, "Logo"), (10.0, "Body text")]) is None
    assert largest_text([(10.0, "Only body"), (10.0, "text here")]) is None
    assert largest_text([]) is None
    assert split_authors("Crump, M.; Logan, G. D.") == ["Crump, M.", "Logan, G. D."]
    assert split_authors("Ann Lee and Bo Chan & Cy Diaz") == ["Ann Lee", "Bo Chan", "Cy Diaz"]
    assert split_authors(None) == []


# -- reading -----------------------------------------------------------------------------


def test_read_pdf(paper):
    (doc,) = read_documents(paper, chunk_words=30)
    assert doc.id == "paper"
    assert doc.fields == {"title": "Typing, Memory and Attention"}
    assert doc.meta["title_source"] == "largest text on page 1"
    assert doc.meta["pages"] == 4
    assert doc.factors == {"author": ["Crump, M.", "Logan, G. D."]}
    text = " ".join(" ".join(c.text.split()) for c in doc.chunks)
    assert "Journal of Testing" not in text
    assert "finger movement was studied and it continues on the next page without" in text
    assert text.count("Selective attention filters") == 2
    assert len(doc.chunks) > 4
    assert doc.chunks[0].meta["page"] == 1 and doc.chunks[-1].meta["page_end"] == 4
    pages = [c.meta["page"] for c in doc.chunks]
    assert pages == sorted(pages)
    # the paragraph that crosses the page break is recorded on both pages
    crossing = next(c for c in doc.chunks if "continues on the next" in c.text)
    assert (crossing.meta["page"], crossing.meta["page_end"]) == (1, 2)


def test_metadata_title_and_fallbacks(tmp_path):
    good = make_pdf(tmp_path / "a.pdf", [[TYPING]], metadata={"title": "A Proper Title"})
    (doc,) = read_documents(good)
    assert (doc.title(), doc.meta["title_source"]) == ("A Proper Title", "metadata")
    assert "author" not in doc.factors
    bare = make_pdf(tmp_path / "plain_file.pdf", [[TYPING]], title=None, header=False)
    (doc,) = read_documents(bare)
    # every span has the same size, so the "largest text" is the whole page: not a title
    assert (doc.title(), doc.meta["title_source"]) == ("plain_file", "file name")


def test_unreadable_pdfs(tmp_path):
    empty = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(empty)
    with pytest.raises(ValueError, match="no extractable text"):
        list(read_documents(empty))
    locked = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret text")
    doc.save(locked, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    with pytest.raises(ValueError, match="password"):
        list(read_documents(locked))
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="cannot read broken"):
        list(read_documents(broken))


# -- library -----------------------------------------------------------------------------


@pytest.fixture()
def built(tmp_path, paper, crump_bib):
    lib = Library.create(tmp_path / "lib", embedder_params={"dim": 256})
    lib.add_file(crump_bib)
    lib.add_file(paper, chunk_words=30)
    lib.build()
    yield lib
    lib.close()


def test_chunk_space_and_document_vector(built):
    doc = built.document("paper")
    chunks = built.space("chunk")
    assert chunks.labels == [f"paper#{n}" for n in range(1, len(doc.chunks) + 1)]
    assert built.build_info["spaces"]["chunk"] == len(doc.chunks)
    # the document vector is the sum of its chunk vectors
    total = np.asarray(chunks.vectors, dtype=np.float64).sum(axis=0)
    np.testing.assert_allclose(built.space("document").vector("paper"), total, rtol=1e-4)
    assert built.chunk("paper#1") == (doc, doc.chunks[0])
    with pytest.raises(KeyError):
        built.chunk("paper#999")


def test_search_chunks(built):
    result = built.search("Stroop conflict attention congruent", space="chunk", k=3)
    top = result.hits[0]
    assert top.document.id == "paper"
    assert "Stroop" in top.chunk.text and top.chunk.meta["page"] == 3
    assert top.to_dict()["chunk"]["page"] == 3 and "Stroop" in top.to_dict()["text"]
    # the whole PDF is also a document, found alongside the BibTeX entries
    assert "paper" in [h.label for h in built.search("typing keyboard", k=None).hits]
    assert built.search("typing", method="lexical", k=None).hits
    # filters look at the chunk's document
    assert built.search("typing", space="chunk", where=lambda d: d.id != "paper").hits == []
    with pytest.raises(LibraryError):
        built.search("typing", space="author", where=lambda d: True)
    with pytest.raises(LibraryError):
        built.search("typing", space="chunk", method="hybrid")
    # chunks compare with every other space
    assert built.similar("paper#1", space="chunk", target="document", k=1)[0].document
    assert built.similar("paper", target="chunk", k=1)[0].chunk is not None


def test_chunks_are_stored_and_replaced(tmp_path, paper):
    with Library.create(tmp_path / "lib") as lib:
        lib.add_file(paper, chunk_words=60)
        n = lib.store.count_chunks()
        assert n == len(lib.document("paper").chunks) > 0
        assert lib.store.get_document("paper").chunks == lib.document("paper").chunks
        assert Document.from_dict(lib.document("paper").to_dict()) == lib.document("paper")
        assert lib.store.lexical_search("Stroop")[0][0] == "paper"
        assert lib.add_file(paper, chunk_words=60).skipped
        lib.add_file(paper, chunk_words=120, force=True)
        assert 0 < lib.store.count_chunks() < n
    # a library written before chunks existed gains the table when opened
    old = tmp_path / "old.sqlite"
    with LibraryStore(old) as store:
        store.conn.execute("DROP TABLE chunks")
        store.set_meta("schema_version", "1")
        store.conn.commit()
    with LibraryStore(old) as store:
        assert store.count_chunks() == 0
        assert store.get_meta("schema_version") == "2"


def test_cli_folder_of_pdfs(tmp_path, paper, capsys):
    folder = tmp_path / "papers" / "sub"
    folder.mkdir(parents=True)
    paper.rename(folder / "paper.pdf")
    make_pdf(folder / "second.pdf", [[MEMORY, ATTENTION]], title="Memory Notes")
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4")
    (folder / "ignored.xyz").write_text("not a document")
    lib = str(tmp_path / "lib")
    assert main(["init", lib]) == 0
    assert main(["add", lib, str(tmp_path / "papers"), "--chunk-words", "30"]) == 2
    out = capsys.readouterr()
    assert "broken.pdf: cannot read" in out.err and "1 file(s) could not be added" in out.err
    assert "library now holds 2 documents (" in out.out and "ignored" not in out.out
    assert main(["build", lib, "--dim", "128", "-q"]) == 0
    assert main(["info", lib]) == 0
    assert "chunks " in capsys.readouterr().out
    assert main(["search", lib, "Stroop conflict", "--space", "chunk", "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert "Typing, Memory and Attention" in out and "paper#" in out and "p. 3" in out

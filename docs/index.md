# Semantic Librarian

The Semantic Librarian builds a search engine from your own documents. It learns what
words mean from the documents themselves, then places words, documents, authors and
anything else you care about in one space, where things with similar meaning sit close
together. You can search that space by typing, or start from one item and look at its
neighbours.

It runs on your own computer. Nothing is uploaded, and no model is downloaded.

[Try the live demo](demo/): 9,932 research projects funded by the U.S. National Science
Foundation in 2024, searchable in your browser together with their investigators,
institutions and funding programs. The first search downloads about 30 MB.

The method is described in Aujla, Crump, Cook & Jamieson (2019), *The Semantic
Librarian: A search engine built from vector-space models of semantics*, Behavior
Research Methods, <https://doi.org/10.3758/s13428-019-01268-4>.

## Install

You need Python 3.11 or newer.

```bash
pip install "semantic-librarian[pdf] @ git+https://github.com/CrumpLab/RsemanticLibrarian"
```

Leave out `[pdf]` if you will not read PDF files. The command `sl` is then available.

## Five minutes

```bash
sl init mylib                    # make an empty library folder
sl add mylib references.bib      # or .csv, .json, .txt, .md, .pdf, or a folder of them
sl build mylib                   # learn the vectors
sl search mylib "attention and memory in skilled typing"
sl serve mylib --open            # the same, in your web browser
```

A library is an ordinary folder. Copy it, back it up, or delete it like any other.

## Where to go next

| If you want to | Read |
|---|---|
| see searching, maps and neighbourhoods on a small example | [Search a library of abstracts](tutorials/abstracts.md) |
| use your reference manager's export | [A library from a BibTeX file](tutorials/bibliography.md) |
| use a book, transcripts or notes, with your own grouping variables | [A library from any text](tutorials/any-text.md) |
| see words, projects, institutions and funding programs in one space, on public-domain data | [NSF awards](tutorials/nsf-awards.md) |
| understand what the numbers mean | [How it works](guide/concepts.md) |
| read PDFs | [Adding documents](guide/inputs.md#pdfs) |
| put a library on a web page | [The web app](guide/web-app.md) |
| move over from the R package | [Coming from the R package](guide/from-r.md) |

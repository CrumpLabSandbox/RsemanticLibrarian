# The web app

```bash
sl serve mylib --open
```

This opens the library in your web browser, at <http://127.0.0.1:8000>. It is served
from your own computer and nothing leaves it. Stop it with Ctrl+C. Use `--port` to
choose another port.

## Using it

**Search by text** ranks documents, words, or the levels of any factor against what you
type. "Matching" chooses how the words combine: the words together, every word, or any
word. A year range can be set when the results are documents.

**Find similar** starts from one item of the library. Choose what kind of thing to
start from, type part of its name, and choose what kind of thing to show. Documents
near a document, authors near an author, and authors near a document were separate tabs
in the original interface; here they are three settings of the same two menus, and
every other pairing works too.

Select a result to read it. From there, "Show the closest" jumps to its neighbours of
any kind, and the names of its authors (or other factor levels) are links.

**The map** places the current results so that similar items are close together, and
colours them in groups. It describes these results only, and is redrawn for each
search.

## Publishing a library

```bash
sl export mylib site/
```

`site/` is then a complete website made of plain files. Upload it to any web host that
serves files, such as GitHub Pages, Netlify or a university web space. No software runs
on the server, because the visitor's browser loads the vectors and does the searching.

Before you publish, remember that the export contains the full text of every document.
Only publish text you have the right to share.

`sl export` replaces an earlier export in the same folder, and refuses to write into a
folder that holds anything else.

## Size

Vectors are stored as 8-bit numbers, a quarter of their size in the library, which
changes similarities by less than 0.01. `--precision float32` keeps them exact at four
times the size.

For a sense of scale, a library of 27,560 abstracts with 43,589 words and 24,215
authors exports to 144 MB. A visitor downloads the document and word vectors, 73 MB,
at their first search, and other spaces when first used.

## Limits

- Search in the browser is by meaning only. Keyword and hybrid search need the command
  line.
- The query is not stored in the page address, so a search cannot be shared as a link.
- On large libraries the page pauses briefly while it scores a search.

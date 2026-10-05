# Adding documents

`sl add LIBRARY FILES...` reads files into a library. You can add as often as you like
and mix formats. A file that has not changed since it was last added is skipped, and a
file that has changed replaces its earlier documents. Run `sl build` afterwards.

A folder can be given instead of files. It is searched at every depth for the file
types below. If one file cannot be read, the others are still added and the problem is
reported.

| Format | Extensions | A document is | Factors |
|---|---|---|---|
| BibTeX | `.bib` | each entry (title, abstract, keywords) | author, year, venue |
| CSV / TSV | `.csv`, `.tsv` | each row | `author`, `authors` or `authorlist`, and `year`, or your own |
| JSON / JSON Lines | `.json`, `.jsonl` | each object | as for CSV |
| Text / Markdown | `.txt`, `.md` | the file, or each paragraph or sentence | the source file |
| PDF | `.pdf` | the file, and each passage of it | author, from the PDF's metadata |

The format is taken from the extension; override it with `--format`.

## Tables: CSV and JSON

By default the columns `title`, `abstract`, `keywords`, `text`, `body`, `content`,
`sentence`, `summary` and `description` are read as text when present, and `author`,
`authors`, `authorlist` and `year` become factors. To choose yourself:

```bash
sl add mylib articles.csv --text title,abstract --id doi \
    --factor "author=authors:;" --factor year --factor journal
```

- `--text` lists the columns to learn from, in order.
- `--id` names the column that identifies a document. Without it, rows are numbered.
- `--factor NAME` uses the column `NAME` as a factor.
- `--factor NAME=COLUMN` uses a differently named column.
- `--factor "NAME=COLUMN:SEP"` also splits each cell on `SEP`, for cells that hold
  several levels such as a list of authors.

Columns that are neither text nor factors are kept and shown with the document.

## Text files

```bash
sl add mylib notes/*.md --split paragraph
```

`--split` is `document` (default), `paragraph` or `sentence`.

## PDFs

Reading PDFs needs the `pdf` extra (`pip install "semantic-librarian[pdf]"`).

```bash
sl add mylib ~/papers
sl build mylib
sl search mylib "response conflict" --space chunk
```

A PDF is read in full. Running headers, footers and page numbers are removed, words
hyphenated across lines are rejoined, and the text is cut into passages of about 200
words (`--chunk-words`) that remember their pages.

The PDF is one document, whose vector is the sum of its passages, so it appears in
ordinary searches next to everything else. The passages form a space of their own,
called `chunk`. Searching it returns the best passages with their page numbers. A
passage is named by its document and position, for example `smith2010#12`.

Things to know:

- **Titles** come from the PDF's metadata when it holds something that looks like a
  title, otherwise from the largest text on the first page, otherwise from the file
  name. Expect some wrong titles.
- **No year** is assigned, because the date inside a PDF is usually when the file was
  made. To have reliable authors and years, also add a BibTeX file of the same papers.
- **Scanned PDFs** without a text layer are reported and skipped. There is no OCR.
- Tables, figure captions and reference lists are read as ordinary text.

!!! warning "Not yet tried on many real articles"
    PDF reading has been developed against generated test files. Check the first few
    results on your own PDFs, and report layouts that come out badly.

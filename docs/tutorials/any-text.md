# A library from any text

This tutorial follows the R package's "Semantic Librarian for any text (Alice in
Wonderland)". The idea is general: take any text, cut it into units, and attach
whatever grouping variables you like to each unit. Every grouping variable, or
*factor*, gets its own space. A level of a factor (one chapter, one speaker, one year)
is the sum of the vectors of the units that carry it.

Here the units are the sentences of a book and the factor is the chapter. You end up
with words, sentences and chapters in one space.

## Get a text

Any plain text file works. For *Alice's Adventures in Wonderland*, download the "Plain
Text UTF-8" file from Project Gutenberg, <https://www.gutenberg.org/ebooks/11>, and
save it as `alice.txt`. Delete the licence text at the top and bottom, and the table of
contents, so that they do not become part of the library.

## The quick way: one sentence per document

No code is needed if the file itself is the only grouping you want:

```bash
sl init libraries/alice
sl add libraries/alice alice.txt --split sentence
sl build libraries/alice
sl search libraries/alice "the queen and the game of croquet"
```

`--split` can be `document` (the whole file is one document), `paragraph` or
`sentence`. Each file becomes a level of a `source` factor, so with several files you
can compare the files themselves: `sl search libraries/alice "croquet" --space source`.

## Your own factors

To attach chapters, the script `examples/any_text.py` builds the documents in Python.
The part that matters is short:

```python
from semantic_librarian import Document, Library

docs = [
    Document(
        id="c08-s0012",
        fields={"text": "The Queen turned crimson with fury."},
        factors={"chapter": ["CHAPTER VIII. The Queen's Croquet-Ground"]},
    ),
    # ... one Document per sentence
]

lib = Library.create("libraries/alice-chapters")
lib.add_documents(docs, source="alice.txt")
lib.build()
```

`fields` holds the text to learn from. `factors` holds the grouping variables, each with
one or more levels; add as many as you like (speaker, scene, date). The script finds
chapters from lines that begin with "CHAPTER" and uses each heading as the level.

Run it on your file, with an optional query:

```bash
python examples/any_text.py alice.txt libraries/alice-chapters "the queen and the game of croquet"
```

It prints the sizes of the spaces, the sentences and the chapters closest to the query,
and the chapters closest to the first chapter. The same questions from the command
line:

```bash
sl search libraries/alice-chapters "the queen and the game of croquet"
sl search libraries/alice-chapters "the queen and the game of croquet" --space chapter
sl similar libraries/alice-chapters "CHAPTER I." --space chapter
sl similar libraries/alice-chapters "CHAPTER I." --space chapter --target word
```

Name a chapter exactly as the script printed it, which is the heading line as it
appears in your file. The last command crosses spaces: it lists the words closest to a
chapter.

!!! note
    This page shows no output for Alice because the book is not distributed with the
    project. The script itself is run by the test suite on a short stand-in text.

## Without writing Python

A spreadsheet does the same job. Put one unit of text per row, with a column for each
factor, save it as CSV, and say which columns are which:

```bash
sl add libraries/mylib sentences.csv --text sentence --factor chapter --factor speaker
```

A cell can hold several levels. `--factor "speaker=speakers:;"` reads the `speakers`
column as a `speaker` factor and splits each cell on semicolons.

## In the browser

The original vignette ended with a Shiny app in which you chose a source space, an item
and a target space. That is the "Find similar" tab of the web app:

```bash
sl serve libraries/alice-chapters --open
```

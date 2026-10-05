# A library from a BibTeX file

This tutorial follows the R package's "Semantic Librarian for bib files". A `.bib` file
is what reference managers such as Zotero export. When the entries include abstracts, the
file is a ready-made corpus about one person's or one topic's literature.

The example is `tests/fixtures/crump.bib`, a Zotero export of 39 publications with
their abstracts, the same file the R vignette used. The script version is
`examples/bibliography.py`.

!!! tip "Exporting from Zotero"
    Select a collection, choose *Export Collection*, and pick the BibTeX or BibLaTeX
    format. Abstracts are included by default.

## Build the library

```bash
sl init libraries/bibliography
sl add libraries/bibliography tests/fixtures/crump.bib
sl build libraries/bibliography
```

```
added     39 documents from tests/fixtures/crump.bib (bibtex)
built with beagle: document (39), word (1429), author (47), venue (24), year (13)
```

Each entry is a document, named by its citation key. The title, abstract and keywords
are the text. Authors, year and venue (the journal or book) are factors, so each gets
its own space. The R vignette needed about forty lines of cleaning code for the braces
and escapes of BibTeX; that is now done when the file is read.

## Look around

Words near a word:

```bash
sl search libraries/bibliography "typing" --space word -k 6
```

```
   1   1.0000  typing
   2   0.3482  typists
   3   0.3396  skilled
   4   0.3388  words
   5   0.3283  performance
   6   0.3143  knowledge
```

Co-authors and near neighbours of an author:

```bash
sl similar libraries/bibliography "Crump, Matthew J. C." --space author -k 4
```

```
   1   1.0000  Crump, Matthew J. C.
   2   0.9231  Brosowsky, Nicholaus P.
   3   0.8970  Logan, Gordon D.
   4   0.8615  Milliken, Bruce
```

Which journals a topic belongs to, and when it was most written about:

```bash
sl search libraries/bibliography "learning" --space venue -k 3
sl search libraries/bibliography "attention" --space year -k 3
```

```
   1   0.5916  Learning & Behavior
   2   0.4810  Canadian Journal of Experimental Psychology
   3   0.4624  Consciousness and Cognition

   1   0.5871  2018
   2   0.4893  2012
   3   0.4722  2016
```

## Map the whole bibliography

The R vignette ended with a map of all the abstracts. In the web app, search for a broad
word and raise "Results" to see many documents at once:

```bash
sl serve libraries/bibliography --open
```

Or compute the map in Python. With three groups, the papers fall into recognizable
lines of work:

```python
from semantic_librarian import Library

lib = Library.open("libraries/bibliography")
labels = lib.space("document").labels
projection = lib.project(labels, n_clusters=3)
```

```
cluster 0: 14 documents, for example
    Hierarchical Control of Cognitive Processes: The Case for Skilled Typewriti
    Warning: This Keyboard Will Deconstruct— The Role of the Keyboard in Skille
    Cognitive Illusions of Authorship Reveal Hierarchical Error Detection in Sk

cluster 1: 17 documents, for example
    Evaluating Amazon's Mechanical Turk as a Tool for Experimental Behavioral R
    Context-Specific Learning and Control: The Roles of Awareness, Task Relevan
    In Support of a Distinction between Voluntary and Stimulus-Driven Control: 

cluster 2: 8 documents, for example
    False Recognition of Instruction-Set Lures
    An Instance Theory of Associative Learning
    The Psychophysics of Contingency Assessment.
```

## Small libraries

Word meanings are learned only from the documents in the library. Thirty-nine abstracts
are enough to group papers sensibly, but word neighbours are rough, as the similarities
around 0.3 above show. Results sharpen as the library grows. To add more, run `sl add`
with another file and `sl build` again; a file that has not changed is skipped.

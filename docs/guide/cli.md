# Command reference

Every command is `sl COMMAND LIBRARY ...`, where `LIBRARY` is the library's folder.
`sl COMMAND --help` lists all options.

## `sl init LIBRARY`

Create an empty library.

| Option | Meaning |
|---|---|
| `--name NAME` | display name (default: the folder name) |
| `--embedder NAME` | `beagle` (default), `beagle-legacy`, `beagle-rp`, `random` |
| `--text-mode MODE` | `unicode` (default) or `legacy`, the R package's exact cleaning |

## `sl add LIBRARY FILES...`

Read files or folders into the library. See [Adding documents](inputs.md).

| Option | Meaning |
|---|---|
| `--format FORMAT` | `json`, `jsonl`, `csv`, `bibtex`, `text`, `pdf` (default: by extension) |
| `--text A,B` | columns to learn from (CSV, JSON) |
| `--factor SPEC` | `NAME`, `NAME=COLUMN` or `NAME=COLUMN:SEP`; repeatable (CSV, JSON) |
| `--id COLUMN` | column holding the document id (CSV, JSON) |
| `--split UNIT` | `document`, `paragraph` or `sentence` (text files) |
| `--chunk-words N` | approximate words per passage (PDFs; default 200) |
| `--force` | read files again even if unchanged |

## `sl build LIBRARY`

Learn the vectors and write every space. Run it after adding documents.

| Option | Meaning |
|---|---|
| `--embedder NAME` | change the embedder |
| `--dim N` | vector length (default 1024) |
| `--seed N` | random seed (default 0) |
| `--param KEY=VALUE` | any embedder parameter, e.g. `max_ngram=5`, `workers=4`; repeatable |
| `--text-mode MODE` | change the text cleaning |
| `-q` | no progress messages |

Settings are saved in the library's `config.toml` and used by later builds.

## `sl search LIBRARY "QUERY"`

Rank a space against free text.

| Option | Meaning |
|---|---|
| `--space NAME` | what to rank: `document` (default), `word`, `chunk`, or a factor |
| `--mode MODE` | `compound` (default), `and`, `or` |
| `--method METHOD` | `semantic` (default), `lexical`, `hybrid`; the last two rank documents |
| `--lexical-weight X` | share of keywords in a hybrid search, 0 to 1 (default 0.5) |
| `-k N` | number of results (default 10) |
| `--years A-B` | only documents in a year range; `2000-` and `-1990` also work |
| `--stopwords` | include very common words when ranking words |
| `--json` | machine-readable output |

## `sl similar LIBRARY ITEM`

Rank a space against an item of the library: a document id, a word, a passage label, or
a factor level such as an author's name.

| Option | Meaning |
|---|---|
| `--space NAME` | the space `ITEM` belongs to (default `document`) |
| `--target NAME` | the space to rank (default: the same space) |
| `-k N`, `--years A-B`, `--stopwords`, `--json` | as for `sl search` |

## `sl serve LIBRARY`

Open the library as a web app on this computer. See [The web app](web-app.md).

| Option | Meaning |
|---|---|
| `--port N` | port (default 8000) |
| `--open` | open the browser |
| `--precision P` | `int8` (default) or `float32` |
| `--fresh` | export again even if nothing changed |

## `sl export LIBRARY OUT`

Write the web app and the library's data to the folder `OUT`, for a static web host.
Options: `--precision`, `-q`.

## `sl info LIBRARY`

Show the documents, sources, settings and spaces of a library, and whether documents
were added since the last build.

## `sl evaluate LIBRARY`

Compare retrieval methods with the paper's three simulations. See
[Evaluation on the APA corpus](../apa-evaluation.md) for what they measure.

| Option | Meaning |
|---|---|
| `--methods A,B` | embedders and/or `wordmatch`, `bm25`, `hybrid` |
| `-n N` | number of target documents (default 200) |
| `--percent A,B` | query sizes as a percentage of a document (default 5,10,25,50,100) |
| `--simulations A,B` | which of 1, 2, 3 to run |
| `--associates MODE` | simulation 2: `nearest` (paper) or `outside-document` |
| `--reference NAME` | model that supplies associates (default `beagle`) |
| `--dim N`, `--param METHOD.KEY=VALUE`, `--seed N` | model settings |
| `--out FILE` | also write all results as JSON |

## `sl embedders`

List the available embedders.

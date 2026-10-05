# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this project is

The Semantic Librarian: a search engine built from vector-space models of semantics,
described in Aujla, Crump, Cook & Jamieson (2019), *Behavior Research Methods*,
<https://doi.org/10.3758/s13428-019-01268-4>. Word meanings are learned with BEAGLE
(Jones & Mewhort, 2007). Documents are sums of their word vectors (paper Eq. 8). Authors
and any other grouping variable ("factor") are sums of their documents' vectors (Eq. 11).
Everything lives in one space, so words, documents and authors can be compared.

The repository is being rebuilt from an R package into a general, local-first Python tool.
Users point it at their own documents, build a library, and search it. The full plan,
decisions, and results are in `plan.md`; read it before starting new work.

- Owner: Matthew Crump (CrumpLab). The repository is `CrumpLabSandbox/RsemanticLibrarian`.
- The Python rebuild was developed on a branch and squash-merged into `master` as one
  commit, so that the APA abstracts (tracked for a while on that branch) are not in
  `master`'s history.
- The original R package is frozen, unchanged, in `legacy/R/`. Do not develop it further.

## Setup

Requires Python 3.11 or newer.

```bash
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[dev]"
# or: python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

## Checks (run all before every commit)

```bash
pytest              # 111 tests, about 20 s, about 97% line coverage
mkdocs build --strict   # only when docs/ or docstrings changed; needs the [docs] extra
ruff check .        # lint (line length 100)
ruff format .       # formatter; CI runs `ruff format --check .`
mypy                # strict-ish: disallow_untyped_defs
```

Three of the export tests run the browser's JavaScript under Node and are skipped when
`node` is not installed.

CI is `.github/workflows/python.yml`. It runs Python 3.11, 3.12 and 3.13 with lint,
format check, mypy and pytest, plus a job that builds the documentation.

## Using the tool

```bash
sl init demo
sl add demo tests/fixtures/crump.bib                # also .json .jsonl .csv .tsv .txt .md .pdf
sl add demo ~/papers --chunk-words 200              # a folder of PDFs (needs pymupdf, in [dev])
sl build demo                                       # default embedder: beagle
sl search demo "attention and memory in skilled typing"
sl search demo "keyboard typing" --method hybrid    # semantic | lexical | hybrid
sl search demo "perception attention" --mode or     # compound | and | or
sl search demo "typing" --space word                # nearest words
sl search demo "response conflict" --space chunk    # passages of PDFs, with page numbers
sl similar demo "Crump, Matthew J. C." --space author --target document
sl evaluate demo                                    # the paper's simulations 1-3
sl serve demo --open                                # the web app, on localhost:8000
sl export demo site/                                # the web app as static files
sl info demo
sl embedders
```

Python API: `from semantic_librarian import Library`, then call `Library.create`,
`.open`, `.add_file`, `.add_documents`, `.build`, `.search`, `.similar` and `.project`.

### The NSF awards example (public domain)

```bash
python examples/nsf_awards.py          # builds libraries/nsf from nsf-data/nsf-awards-2024.jsonl
sl serve libraries/nsf --open
```

`nsf-data/nsf-awards-2024.jsonl` (32 MB, 9,932 projects) is committed so adopters have
something to try. It was prepared by the script from NSF's raw file and keeps only
titles, abstracts, names of people and organizations, amount, type and start date. NSF's
raw yearly zips also hold investigators' and program officers' e-mail addresses and
phone numbers; they are git-ignored (`nsf-data/*.zip`) and must not be committed.

9,932 projects, 47,603 words, and seven factor spaces (investigator, institution,
state, program, division, directorate, year); about 3 minutes to build, 420 MB. NSF's
file list is at `https://api.nsf.gov/services/v2/s3/list-files`. Unlike the APA corpus
this data can be published.

### Working on the web app

```bash
cd web && npm install
npm run dev        # hot reload; reads /data from `sl serve <library>` on port 8000
npm run build      # writes src/semantic_librarian/web/, which is committed
```

`.claude/launch.json` defines preview servers `apa` (port 8000), `demo` (port 8001),
`docs` (the documentation site, port 8002) and `nsf` (port 8003).

### The full APA corpus

`apa-data/all_journals_rev.csv` holds 27,560 articles from 1894 to 2016. It is the
paper's corpus, and its columns are journal, year, title, abstract, keywords,
authorlist, pages, issue and volume. **It is not in the repository:** `apa-data/` is
git-ignored because the abstracts are not ours to redistribute. The owner has a copy;
never add it back.

```bash
sl init libraries/apa
sl add libraries/apa apa-data/all_journals_rev.csv \
    --factor "author=authorlist:;" --factor year --factor journal
sl build libraries/apa                      # about 9 min on 4 cores, 491 MB library
python scripts/apa_paper_examples.py libraries/apa   # reproduces the paper's Figs. 5-8
sl evaluate libraries/apa -n 1000           # about 9 min
sl evaluate libraries/apa -n 1000 --simulations 2 --associates outside-document
```

Results and their interpretation are in `docs/apa-evaluation.md`. The `libraries/`
folder is git-ignored; never commit built libraries.

## Repository layout

```
src/semantic_librarian/
  document.py        Document(id, fields, meta, factors, chunks) and Chunk(text, meta)
  library.py         Library: create/open, add_file, build, search, similar, project
  config.py          LibraryConfig <-> config.toml (hand-written TOML writer; tomllib reads)
  cli.py             the `sl` command (argparse)
  text/clean.py      TextProcessor: "legacy" (exact R cleaning) and "unicode" (default) modes
  text/chunk.py      pack paragraphs into non-overlapping chunks of about 200 words
  text/stopwords.py  English stop word list
  embed/base.py      Embedder ABC, WordSumEmbedder (documents/queries = sums of word vectors), Corpus
  embed/beagle.py    BEAGLE: composite (context + order, default) and legacy (R parity); parallel order
  embed/beagle_rp.py BEAGLE-RP (random permutations)
  embed/random.py    the paper's non-semantic control
  embed/cooccurrence.py  sparse-matrix context sums and BEAGLE-RP order sums
  embed/registry.py  make_embedder(name, **params); names: beagle, beagle-legacy, beagle-rp, random
  search/similarity.py  cosine, compound/AND/OR combination, rank
  search/lexical.py  doc-term matrix, BM25, word-match control
  search/fusion.py   weighted reciprocal rank fusion (hybrid search)
  space/             Vocabulary, VectorSpace (labels + matrix, .npy on disk), factor_space
  store/sqlite.py    LibraryStore: documents, chunks, factors, sources, builds, FTS5 table
  project/maps.py    classical MDS (= R cmdscale) + k-means
  eval/simulations.py  the paper's Simulations 1-3 for any methods
  ingest/            json/jsonl/csv (FieldMapping), bibtex, plain text, pdf (pymupdf)
  export/bundle.py   `sl export`: web app + data folder (8-bit vectors, sharded records)
  export/server.py   `sl serve`: static file server on localhost (no live API yet)
  web/               the BUILT web app (generated by `npm run build`; do not edit by hand)
web/                 web app source: Vite + Svelte 5, plain JavaScript
  src/lib/           search.js, text.js, mds.js, bundle.js: browser ports of the Python
  src/*.svelte       App (state and layout), Picker (item autocomplete), Map (SVG scatter)
  test/rank.mjs      runs the browser modules under Node for tests/test_export.py
tests/               pytest; fixtures in tests/fixtures (R references, APA 100-article sample, crump.bib)
scripts/             legacy_reference.R, convert_legacy_fixtures.py, apa_paper_examples.py
docs/                documentation site sources (mkdocs-material; config in mkdocs.yml)
  tutorials/         three tutorials mirroring the R vignettes
  guide/             concepts, inputs, web app, CLI and Python reference, coming from R
  apa-evaluation.md  full-corpus results
examples/            the tutorials as runnable scripts (run by tests/test_examples.py)
legacy/R/            the frozen R package (installable with subdir = "legacy/R")
plan.md              plan, decisions, phase reports
```

A library on disk:

```
mylib/
  config.toml       text mode, fields, factors, embedder + params
  library.sqlite    documents, chunks, factors, sources (sha256 for incremental adds), builds
  vectors/          <space>.npy + <space>.labels.json, vocab.json, doc_terms.npz, build.json
```

Builds write to `vectors.tmp/` and then swap in, so a failed build never leaves a
half-written `vectors/`.

## Decisions already made (don't relitigate)

- Local-first Python package. SQLite plus numpy files; no server database.
- The web UI must work from a static export first; that is built. `sl serve` is a
  static file server for now and the FastAPI live API comes later. The frontend is
  Vite + Svelte, not Astro, because it is a single interactive page (plan Section 12).
- **BEAGLE only for now.** Transformer embedders are deferred by the owner. When
  revisited, add an ONNX Runtime extra as the default and sentence-transformers as an
  optional second extra. The `Embedder` interface already allows non-word-sum models.
- Default embedder `beagle` (composite). Default search method `semantic`; hybrid is
  opt-in.
- License: GPL-2.0-or-later, matching the R package.

## Conventions and gotchas

- **R parity is tested, not assumed.** `beagle-legacy` plus the `legacy` text mode
  reproduce `sl_beagle_vectors`, `LSAfun::breakdown`, `sl_clean`, the search functions
  and `cmdscale` exactly (tests in `tests/test_text.py`, `test_beagle.py`,
  `test_search.py`). To regenerate the references you need R with dplyr, tidyr,
  stringr, magrittr and jsonlite. Run
  `LANG=C.UTF-8 Rscript scripts/legacy_reference.R && python scripts/convert_legacy_fixtures.py`.
  LSAfun, qdapRegex and lsa are not needed because the script contains exact copies of
  the three functions used.
- **Legacy behaviors deliberately not carried into the defaults:** regular-expression
  author matching (now exact), the `abs(max(x))` normalization, context-only BEAGLE,
  sentence splitting on every period, and empty documents getting the first word's
  vector. Keep them only in `beagle-legacy` and the `legacy` text mode.
- **Parallel BEAGLE must stay deterministic.** Order information is summed over chunks
  of about 20,000 tokens (`CHUNK_TOKENS`). Chunk boundaries depend only on the corpus
  and partial sums are added in chunk order. Vectors must be bit-identical for any
  worker count, and a test enforces this. Workers are limited to one BLAS thread;
  without that they oversubscribe the CPU and run slower than serial. Process pools
  use the platform default start method; this has been run on Linux and macOS.
- **FFTs use `scipy.fft`**, which is faster than `numpy.fft` with identical results.
- **Word match** (evaluation control) counts distinct non-stop query words in a
  document, the paper's "largest word overlap". Counting all occurrences including stop
  words gave median ranks of 200-370 instead of about 1 on the APA corpus.
- **Evaluation:** the same targets are used at every query size, and queries keep word
  multiplicity (`embed_query(..., unique=False)`). Tied ranks share the average.
  `sl evaluate` reuses a built library's model when its settings match, ignoring
  `workers`.
- Word-space results hide stop words unless `include_stopwords=True` / `--stopwords`.
- **PDFs and chunks.** A PDF is one document with `chunks`; it is embedded from its
  chunks and its fields (the title) are display only. Chunks never overlap, so the
  document vector equals the sum of its chunk vectors; a test enforces this. Chunk
  labels are `<document id>#<n>`, and `chunk` is a reserved space name. pymupdf is
  imported lazily; it is an optional `pdf` extra and AGPL-licensed. PDF metadata is
  unreliable, so no `year` factor is assigned from it.
- **The browser must rank like Python.** `web/src/lib/search.js` and `text.js` mirror
  `search/similarity.py`, `Library.search` and `text/clean.py`. Change both sides
  together; `tests/test_export.py` compares them. The web app has semantic search only.
- **The built web app is committed.** After editing `web/src`, run `npm run build` and
  commit `src/semantic_librarian/web/`. File names are fixed (no hashes) on purpose.
- **Export format** is versioned (`EXPORT_FORMAT` and `manifest.format`); the app
  rejects other versions. `sl export` only overwrites a folder that already holds an
  export.
- **Documentation shows real output.** Every output block in `docs/tutorials/` was
  pasted from running the command above it. When behaviour or defaults change, rerun
  and repaste; do not edit numbers by hand. `docs/guide/cli.md` lists options by hand,
  so update it with `cli.py`. The Python reference is generated from docstrings.
- **Sums of many documents look alike.** Factor levels with many documents (large
  institutions, programs, prolific authors) have cosines near 0.99 with each other, and
  their nearest words are generic. Ranks are still meaningful. This is a property of
  summed vectors, seen in both the APA and NSF libraries; removing the shared component
  (for example by centering) is an untried idea, not a decision.
- Lexical and hybrid search rank documents only. They need `vectors/doc_terms.npz`;
  libraries built before Phase 2 must be rebuilt.
- Commit messages: imperative subject, explanatory body. Never put model identifiers
  in commits, PRs or code. Do not open pull requests unless asked.

## Status and next steps

Done: Phase 0 (restructure, scaffold), Phase 1 (core library, BEAGLE parity, CLI),
Phase 2 for BEAGLE (hybrid search, parallel builds, evaluation harness), the full
APA evaluation, Phase 3 (PDF ingestion with a chunk space), and the static half of
Phase 4 (`sl export`, `sl serve`, the web app), and Phase 5 (documentation site sources,
tutorials, R migration notes). Details and numbers are in `plan.md` Sections 8-13.

Next, in order (see `plan.md` Section 4):

1. **Try Phase 3 on real articles.** PDF ingestion has only been run on generated PDFs.
   Run it on a folder of journal PDFs and check titles, header stripping and two-column
   reading order. GROBID or BibTeX pairing for reliable metadata is deferred.
2. **Phase 4, live server.** Replace the static `sl serve` with FastAPI behind a
   `SearchProvider` interface in the app, adding keyword and hybrid search. Other gaps
   are listed in `plan.md` Section 12 (whole-corpus map, shareable links, Web Worker).
3. **Publish the documentation** once the owner decides about making the repository
   public (plan Section 13 lists what to check first).

Open follow-ups: performance of BEAGLE's convolution order term on very large corpora
(consider smaller `max_ngram`, measured with `sl evaluate`), and confirming the parallel
build on Windows. On macOS (12 cores, 8 workers) the full APA build ran in 187 s.

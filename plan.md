# Semantic Librarian 2.0: Refactor Plan

Status: Phases 0, 1 and 2 are implemented (Sections 8 and 9), and the paper's
simulations have been re-run on the full APA corpus (Section 10,
[docs/apa-evaluation.md](docs/apa-evaluation.md)). Transformer embedders
are deferred by decision; the project is BEAGLE-only for now. Phase 3 (PDF ingestion)
is implemented (Section 11), and so is the static half of Phase 4, the web app
(Section 12), and Phase 5, the documentation (Section 13). The live server API is
the main piece still to build.

## 0. Decisions so far

| Decision | Answer |
|---|---|
| Primary use case | A researcher runs it **locally** on their own material. They may need a database, and they will have Python (or can install it). |
| Deployment of search | **Both, static first**: the web UI must work from a static export with no server; a local Python server adds live features on top. |
| Embeddings | **Both behind one interface**: BEAGLE parity with the R code is milestone one, transformer models follow. |
| Code location | **Restructure this repo** into a monorepo: Python package, web app, and the R package moved under `legacy/`. |
| Database | **SQLite + numpy files**. FTS5 for BM25; `sqlite-vec` as an optional extra later. |
| PDF ingestion | **Full text, chunked** with pymupdf. Chunks are searchable individually and summed into a document vector. GROBID deferred. |
| R package | **Frozen under `legacy/R/`**, moved with `git mv`, still installable via `subdir`. No further R development. |
| Default embedder | **BEAGLE**. No download, seed-reproducible, works in a static export. |
| Transformers | **Deferred.** BEAGLE only for now. When revisited: an ONNX Runtime extra (about 200-250 MB installed with a small model) as the default, sentence-transformers as a second extra. |

## 1. Goal

Turn the Semantic Librarian from an R package tied to one APA abstract corpus into a
general, local-first utility: a researcher installs one Python package, points it at a
folder of PDFs, a BibTeX file, a JSON/CSV export, or plain text, and gets a searchable
vector-space library with a modern web interface running on their own machine. The
same library can be exported as a static site to share.

The intended workflow is:

```
pip install semantic-librarian          # or pipx / uv tool install
sl init mylib                           # creates mylib/ with a config and SQLite db
sl add mylib ~/papers/*.pdf             # ingest (also: .bib, .json, .csv, .txt)
sl build mylib --embedder beagle        # or --embedder minilm, bge-small, openai, ...
sl serve mylib                          # opens http://localhost:8000 with the full UI
sl export mylib ./site                  # static bundle, deployable to GitHub Pages
```

The scientific identity of the project stays intact. BEAGLE and the "sum word vectors
to get document and factor vectors" idea remain first-class, but they become one
embedding backend among several, and the paper's three simulations become a benchmark
harness for comparing backends.

## 2. What exists today (summary)

- R package `RsemanticLibrarian` with 19 exported functions: text cleaning, BEAGLE
  word vectors (context information only, no order/convolution term), article and
  author vectors by summation, cosine search, MDS + k-means for 2-D plots.
- A separate Shiny app (CrumpLab/SemanticLibrarian, not in this repo) that provides
  the UI described in the paper.
- Pure R loops; building the 27k-document space is slow. One Rcpp helper.
- Pipeline hard-codes the APA schema (`title`, `abstract`, `authorlist`, `year`).
- Known defects: regex author matching, `abs(max(x))` normalization, single-term
  AND/OR queries dropped, `sl_clean` returns NULL on vector input, factor-bloated
  sample data (31 MB in memory for 100 rows), empty examples, vignette headers
  commented out.

## 3. Target architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  semantic_librarian  (Python package, pip-installable)                │
│                                                                      │
│  ingest/    pdf · bibtex · json/jsonl/csv · txt · (zotero rdf?)      │
│      │      -> Document records: id, text fields, metadata/factors    │
│  text/      cleaning, sentence + token segmentation (spaCy or regex)  │
│  embed/     Embedder protocol                                        │
│             ├─ BEAGLE (numpy; context + order via FFT convolution)    │
│             ├─ BEAGLE-RP (sparse ternary, permutation)                │
│             ├─ SentenceTransformer (local, e.g. bge / e5 / MiniLM)    │
│             ├─ API embedders (OpenAI, Voyage, ...), optional extras   │
│             └─ (later) SPLADE / BM25 hybrid for lexical fallback      │
│  space/     Library object: word, document, factor (author, year,     │
│             chapter, ...) vector spaces in ONE coordinate system      │
│  search/    cosine, compound / AND / OR queries, cross-space search   │
│             (word->doc, doc->doc, doc->author, author->author)        │
│  project/   MDS, UMAP, t-SNE; k-means / HDBSCAN clusters              │
│  eval/      Simulations 1-3 from the paper as a benchmark harness     │
│  export/    static bundle (JSON + binary float16 arrays) for the web  │
│  cli        `sl ingest`, `sl build`, `sl search`, `sl export`, `sl serve` │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  web/  (Astro site)                                                   │
│  - Static pages: about, corpus summary, per-document pages            │
│  - Interactive islands (React or Svelte): search box, ranked results, │
│    2-D semantic map (deck.gl or Plotly.js / Observable Plot),         │
│    cluster controls, author/factor neighborhoods                      │
│  - Search modes:                                                      │
│    (a) fully static: BEAGLE word vectors shipped to the browser,      │
│        query = sum of word vectors, cosine against doc matrix in WASM │
│    (b) in-browser transformer (transformers.js / ONNX) for modern     │
│        embeddings without a server                                    │
│    (c) optional FastAPI backend for large corpora or API embeddings   │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.1 Core data model

```python
@dataclass
class Document:
    id: str
    fields: dict[str, str]  # e.g. title, abstract, body, keywords
    meta: dict[str, Any]  # year, journal, doi, source path, ...
    factors: dict[str, list[str]]  # author -> ["Crump, M.", "Logan, G."], chapter -> ["3"]


class Library:
    documents: list[Document]
    vocab: Vocabulary  # token -> row index, frequencies, stoplist
    spaces: dict[str, VectorSpace]  # "word", "document", "author", "year", ...
    embedder: EmbedderSpec  # name + params, so the build is reproducible
```

`VectorSpace` is a labelled float matrix (labels, matrix, optional metadata). Factor
spaces are derived from the document space by summation (or mean) over the documents
carrying each level, exactly the paper's Equation 11, but for any factor column.

### 3.2 Embedder protocol

```python
class Embedder(Protocol):
    def fit(self, sentences: Iterable[list[str]], vocab: Vocabulary) -> None: ...
    def embed_words(self, vocab) -> np.ndarray           # may be None for transformers
    def embed_documents(self, docs) -> np.ndarray
    def embed_query(self, text: str) -> np.ndarray
```

- **BEAGLE**: faithful to the paper. Context term (Eq. 1) plus order term (Eqs. 2-5)
  using circular convolution via FFT, with the placeholder vector Φ and n-gram window
  up to a configurable limit. Vectorized with numpy; optional numba for the sentence
  loop. Document vectors are word sums; queries are word sums. Fully reproducible
  from a seed.
- **BEAGLE-RP**: the Recchia et al. variant with sparse ternary vectors and index
  permutation for order. Cheap to add once BEAGLE exists.
- **Transformer embedders**: `sentence-transformers` locally (default a small open
  model so the whole thing runs on a laptop), with API embedders as optional extras.
  For these, "word vectors" do not exist in the same sense, so word-level neighborhoods
  come from a separate pass (embed vocab tokens) or are disabled.
- **Hybrid**: BM25 (rank_bm25 or sqlite FTS5) blended with dense scores, to recover
  exact-term retrieval that pure semantic search can miss.

### 3.3 Ingestion

| Source | Library | Notes |
|---|---|---|
| PDF | `pymupdf` (fitz) | Text, page spans, basic metadata. Optional GROBID client for title/authors/abstract/references on scholarly PDFs. |
| BibTeX / Zotero export | `bibtexparser` | Replaces the `RefManageR` vignette. Abstract + title + authors + year. |
| JSON / JSONL / CSV | stdlib / pandas | User declares which columns are text, which are factors, via a small mapping config. |
| Plain text / Markdown | stdlib | Split into sentences or paragraphs; user-supplied factors (the "Alice" case). |
| Existing APA `.RData` | `pyreadr` | One-time migration of the 27,560-document corpus and author list. |

Every ingester emits the same `Document` records, so everything downstream is
source-agnostic.

### 3.4 Storage and index (local database)

A library is a directory containing one SQLite file plus the config. SQLite is chosen
because it is zero-install (ships with Python), single-file, copyable, and inspectable
with ordinary tools.

```
mylib/
├── config.toml          # sources, field mapping, embedder spec, seed
├── library.sqlite       # documents, chunks, vocab, factors, source provenance
├── vectors/             # one .npy (float32) per space: word, document, chunk, author...
└── cache/               # extracted PDF text, model downloads, projections
```

- Tables: `documents`, `chunks` (for PDFs and long texts), `fields`, `factors`,
  `factor_levels`, `vocab`, `sources` (file hash + mtime so `sl add` is incremental),
  `builds` (embedder spec + seed + timestamp so results are reproducible).
- Vectors live in `.npy` files next to the database rather than as BLOBs, so numpy can
  memory-map them. `sqlite-vec` is an optional extra for in-database ANN search when a
  corpus outgrows brute force.
- Search: brute-force cosine with numpy up to roughly 10^5 documents at 1,024 dims is
  fine on a laptop. Add `hnswlib` or FAISS behind the same interface only if a corpus
  needs it.
- Full-text search: SQLite FTS5 gives BM25 for free, which is the lexical half of the
  hybrid ranker.

### 3.5 Web interface

Static-first, so one frontend serves two modes:

1. **Static export** (`sl export`): the frontend plus a data bundle. Search runs
   entirely in the browser. Works for BEAGLE out of the box (query = sum of shipped
   word vectors). For transformer backends the export can either embed the query with
   transformers.js in the browser or fall back to lexical-only search.
2. **Local server** (`sl serve`): the same frontend served by a small FastAPI app
   that adds what static cannot do: embedding queries with any backend, searching
   corpora too large to ship, adding documents from the UI, and recomputing
   projections on demand.

The frontend is built once and shipped inside the Python wheel, so a user never needs
Node installed; Node is a developer dependency only.

- Astro for the shell, routing, and static corpus pages; one framework for islands
  (recommend Svelte or React; see questions). If the UI turns out to be a single
  interactive app with little static content, a plain Vite + Svelte/React app is the
  simpler choice and the plan should not force Astro.
- The build step exports: document metadata (JSON), document matrix (float16 binary),
  word vectors if BEAGLE (float16 binary, can be large: 40k × 1024 × 2 bytes ≈ 80 MB,
  so ship a vocab-pruned or dimension-reduced version, or lazy-load), precomputed 2-D
  projection and clusters at a few cluster counts.
- Search in the browser: cosine over the document matrix in a Web Worker (or a small
  Rust/WASM kernel if needed). For transformer backends, either transformers.js in the
  browser or the local FastAPI endpoint. The frontend talks to one `SearchProvider`
  interface with a static implementation and a server implementation.
- Semantic map: WebGL scatter (deck.gl ScatterplotLayer or regl-scatterplot) with
  hover, click-to-detail, lasso, color by cluster/year/factor. Optionally recompute
  the local MDS for the top-N results client-side (as the Shiny app did) versus
  showing a precomputed global UMAP with the results highlighted.
- Reproduces all four tabs of the original: free search (compound/AND/OR), document
  neighborhood, factor (author) neighborhood, document-to-factor.

### 3.6 Evaluation harness

Port Simulations 1-3 so any `Embedder` can be scored on the same corpus:

1. Target recovery: median rank of a document when queried with 5/10/25/50/100% of its
   own words.
2. Associate substitution: same, with words replaced by nearest semantic neighbors.
3. Rank agreement between backends (Spearman).

This gives a principled, paper-continuous way to decide which modern embedding earns a
place in the default configuration, and material for a follow-up paper.

## 4. Phases

### Phase 0: Decisions and scaffolding — done
- Settle the questions in Section 6.
- Create the repo layout, `pyproject.toml`, CI (ruff, mypy, pytest), docs skeleton.

### Phase 1: Core Python library with BEAGLE parity — done
- `Document` / `Library` / `VectorSpace` types.
- Text cleaning and sentence segmentation that reproduces `LSAfun::breakdown`
  behavior closely enough to compare with the R output.
- BEAGLE context-only embedder that matches `sl_beagle_vectors` numerically given the
  same random vectors (parity test against the R package on the 100-doc sample).
- Then add the order term and BEAGLE-RP.
- Cosine search with compound / AND / OR; factor spaces; MDS + k-means.
- JSON/CSV/JSONL and BibTeX ingesters. CLI.
- Migrate the sample data from this repo into test fixtures.

### Phase 2: Search quality, speed and evaluation — done (BEAGLE only)
- Hybrid search: BM25 over the library's own tokens fused with semantic rankings.
- Parallel BEAGLE order computation.
- Evaluation harness reproducing Simulations 1-3; run on the available corpora.
- Deferred: transformer embedders (ONNX or sentence-transformers extras) and API
  embedders. The `Embedder` interface already accommodates them.

### Phase 3: PDF ingestion — done (GROBID still deferred)
- pymupdf text extraction, page-aware chunking, de-hyphenation, header/footer
  stripping.
- Optional GROBID metadata.
- Chunk-level versus document-level vectors (a PDF yields many chunks; document vector
  = sum or mean of chunk vectors, searchable at both levels).

### Phase 4: Web app — static export and browser app done; live API not yet
- Export format from Python (`sl export`).
- Frontend with search island, results table, semantic map, detail panel, driven by
  the `SearchProvider` interface (static bundle first).
- `sl serve`: FastAPI app serving the built frontend plus the live search API; the
  built frontend is packaged into the wheel.
- Static deployment target (GitHub Pages / Netlify) for shared libraries.
- Flagship demo: the APA corpus (if licensing allows), plus the Alice and Crump.bib
  examples.

### Phase 5: R package disposition and docs — done, except publishing the site
- Move `RsemanticLibrarian` under `legacy/R/` and either freeze it with a README
  pointer to the Python package, or reduce it to a thin `reticulate` wrapper.
  (Question below.) The pkgdown `docs/` site is either rebuilt for the new project or
  retired.
- Documentation site (mkdocs-material or the web app itself), tutorials mirroring the
  three existing vignettes.

## 5. Repository layout (this repo, restructured)

```
RsemanticLibrarian/   (consider renaming the repo to semantic-librarian later)
├── pyproject.toml
├── src/semantic_librarian/
│   ├── ingest/   text/   embed/   space/   store/   search/   project/   eval/   export/
│   ├── server/       (FastAPI app for `sl serve`)
│   ├── web/          (built frontend assets, generated, shipped in the wheel)
│   └── cli.py
├── tests/            (fixtures = the 100-doc sample, Alice, Crump.bib)
├── web/              (frontend source: Astro or Vite app; Node is a dev dependency only)
├── examples/         (notebooks / scripts for each tutorial)
├── legacy/R/         (the R package, moved verbatim, including its pkgdown docs/ output)
├── scripts/          (R reference generator and fixture converter for the parity tests)
├── docs/             (new docs site, later)
└── plan.md
```

The move to `legacy/R/` is a pure `git mv` so history stays attached. The R package
remains installable with `devtools::install_github("CrumpLab/RsemanticLibrarian",
subdir = "legacy/R")`.

## 6. Remaining questions and working defaults

The material decisions are settled (Section 0). These remain open but none blocks
Phase 1; the default in bold is what I will use unless told otherwise.

1. **Corpus scale**: hundreds of PDFs, the 27k APA abstracts, or 10^6 chunks?
   Default: **design for up to ~10^5 chunks with brute-force cosine**; ANN is an
   extra, not a requirement.
2. **APA corpus**: do you still have the full 27,560-document dataset, and can it be
   redistributed as a demo? Default: **assume it is available privately for the
   evaluation harness but the public demo uses the Alice text and Crump.bib**.
3. **Island framework**: Svelte or React? Default: **Svelte**, inside Astro, unless
   the UI turns out to need deck.gl's React bindings.
4. **Python version floor**: default **3.11+**.
5. **Licensing for new code**: default **keep GPL (>= 2)** to match the R package;
   say so if you prefer MIT or Apache-2.0.
6. **Repository name**: `RsemanticLibrarian` no longer describes the project. Default:
   **leave as is for now**; renaming is a one-line GitHub change later.

## 7. Phase 0 and Phase 1 checklist (all done)

(Phase 2 is summarized in Section 9.)

1. `git mv` the R package (`R/`, `man/`, `src/`, `data/`, `vignettes/`, `DESCRIPTION`,
   `NAMESPACE`, `.Rbuildignore`, `_pkgdown.yml`, `*.Rproj`) into `legacy/R/`; remove
   the committed pkgdown `docs/` output; update README.
2. Scaffold `pyproject.toml` (hatchling or uv), `src/semantic_librarian/`, `tests/`,
   ruff + pytest + mypy config, GitHub Actions CI.
3. Convert the 100-document sample data (`.rda`) to Parquet/JSON test fixtures with
   `pyreadr`, including the R-generated word, article and author vectors so the BEAGLE
   parity test has a reference.
4. Implement `Document`, `Vocabulary`, `VectorSpace`, the SQLite store, the text
   cleaner (matching `LSAfun::breakdown` behavior), and the context-only BEAGLE
   embedder; pass the parity test.
5. Add the BEAGLE order term and BEAGLE-RP; cosine search with compound / AND / OR;
   factor spaces; MDS + k-means; JSON/CSV/BibTeX ingesters; the `sl` CLI.

## 8. Phase 0 and Phase 1: what was built and what was learned

### Built

- **Restructure.** The R package moved verbatim to `legacy/R/` with `git mv`, including
  its pkgdown `docs/` output (kept rather than deleted). Its README now says it is
  frozen and how to install it from the subdirectory.
- **Python package** `semantic_librarian` (`pyproject.toml`, hatchling, Python 3.11+),
  with the `sl` CLI: `init`, `add`, `build`, `search`, `similar`, `info`, `embedders`.
- **Text**: `legacy` mode reproduces `LSAfun::breakdown()` + `rm_white_lead_trail()`
  exactly; `unicode` mode (the default) keeps any script, folds accents and keeps
  alphanumeric tokens such as "n400". A sentence splitter protects abbreviations.
- **Embedders** behind one interface: `beagle` (context + order, Eqs. 1-5),
  `beagle-legacy` (bit-for-bit the R package), `beagle-rp`, and the paper's `random`
  control. Documents are word sums; factor spaces are document sums with exact level
  matching.
- **Search**: compound / AND / OR, cross-space similarity (e.g. author to documents),
  document filters, classical MDS + k-means, and BM25 keyword search via SQLite FTS5.
- **Storage**: `config.toml` + `library.sqlite` + `vectors/*.npy`; unchanged files are
  skipped on re-add; each build is recorded and written atomically.
- **Ingesters**: JSON, JSONL, CSV/TSV (with field mapping and factor separators),
  BibTeX/BibLaTeX, plain text/Markdown split by document, paragraph or sentence.
- **Tests**: 70 tests, 95% line coverage, ruff + mypy clean, GitHub Actions workflow
  for Python 3.11-3.13.

### Parity with the R package

`scripts/legacy_reference.R` runs the original R functions (with exact copies of the
three functions used from LSAfun 0.6.2, qdapRegex and lsa, which are not installable
from this environment) and saves reference outputs that the tests compare against:

| R function | Python | Agreement |
|---|---|---|
| `LSAfun::breakdown`, `sl_clean`, `sl_corpus_dictionary`, `sl_clean_vector` | `text.clean` legacy mode | identical strings |
| `sl_beagle_vectors` (same random vectors) | `Beagle(variant="legacy")` | identical (max diff 0.0) |
| `sl_article_vectors`, `sl_author_vectors` | `compose_documents`, `factor_space` | 1e-13 |
| `get_search_article_similarities` (compound / AND / OR) | `cosine_scores`, `combine_term_scores` | 1e-12 (AND/OR: R rounds to 4 dp) |
| `get_article_article_similarities`, `get_author_similarities` | `cosine_scores` | R's 4 dp rounding |
| `get_mds_article_fits`, `get_mds_author_fits` (`cmdscale`) | `classical_mds` | 1e-9, up to axis sign |

### Legacy behaviors deliberately not carried into the default path

- The R BEAGLE has no order information, initializes memory to the environment vectors
  and rescales by `abs(max(x))` after every update; the default `beagle` follows the
  paper instead. `beagle-legacy` keeps the old behavior for comparison.
- `sl_author_vectors` matched authors with a regular expression (substring matches);
  factors now match exactly.
- Single-word AND/OR queries returned nothing in R; they now return that word's ranking.
- Empty documents received the first word's vector in R; they now get a zero vector.
- Sentence splitting on every period (so "e.g." and "3.14" split sentences) is kept only
  in `legacy` text mode.
- The R pipeline read all titles before all abstracts; the Python pipeline reads
  document by document. This only matters for `beagle-legacy`, whose per-update
  normalization makes it order dependent.

### Performance (1,000 APA abstracts, 95,000 tokens, 4 cores)

| Model | Build time | Per token |
|---|---|---|
| `beagle` (context + order, dim 1024, n-grams up to 7) | 35 s | ~370 µs |
| `beagle` context only | 0.14 s | 1.5 µs |
| `beagle-rp` (dim 3000) | 0.7 s | 7 µs |

Context information and BEAGLE-RP order are linear, so they are computed as sparse
matrix products. BEAGLE's convolution-based order term is the bottleneck: the full
27,560-document APA corpus (about 2.6 M tokens) would take roughly 15-20 minutes on one
core. It is already about twice as fast as a direct implementation (shared chain
prefixes, batched FFTs). Follow-ups: spread sentences over worker processes, and report
how much smaller `max_ngram` values (e.g. 3-4) cost in retrieval quality using the
evaluation harness.

### Follow-ups found along the way

- Stop words appear among nearest neighbors in the word space; consider filtering them
  from word-space results by default.
- The bundled sample `article_df` keeps 27,560 factor levels for 100 rows; the Python
  fixtures store plain values (98 KB instead of 31 MB in memory).
- If GitHub Pages currently serves the pkgdown site from `docs/` on the default branch,
  that site will stop being served once this restructure is merged; it now lives in
  `legacy/R/docs/`.

## 9. Phase 2: what was built and what was learned

### Built

- **Hybrid search.** At build time the library stores a document-term matrix
  (`vectors/doc_terms.npz`). `search(method=...)` ranks by `semantic` (vector space),
  `lexical` (Okapi BM25 over the same tokens as the semantic model) or `hybrid`
  (weighted reciprocal rank fusion, `lexical_weight` 0-1; documents with no query word
  get no keyword contribution). CLI: `sl search --method hybrid --lexical-weight 0.5`.
  Hits from a hybrid search carry both component scores. The SQLite FTS5 index stays in
  the store for future phrase search but is no longer used for ranking.
- **Stop words hidden from word-space results** by default (`--stopwords` shows them).
- **Parallel BEAGLE.** Order information is computed over chunks of about 20,000 tokens
  in worker processes (automatic above 50,000 tokens; `workers` parameter). Chunk
  boundaries depend only on the corpus and partial sums are added in a fixed order, so
  vectors are bit-identical for any worker count (tested). Each worker is limited to one
  BLAS thread; without that, workers oversubscribed the CPU and ran slower than serial.
  FFTs moved from `numpy.fft` to `scipy.fft` (identical results, faster).
- **Evaluation harness** (`semantic_librarian.eval`, `sl evaluate`): Simulations 1-3
  for any embedders plus the paper's word-match control, BM25 and hybrid. Same targets at
  every percentage, word multiplicity kept in queries (paper Eq. 9), ties share their
  average rank, associates for Simulation 2 come from one reference model (BEAGLE) so
  every method sees identical queries. Outputs median rank, mean reciprocal rank, top-1
  and top-10 rates, and mean/SD Spearman rho per method pair; `--out` saves every query.
- 84 tests, 96% line coverage.

### BEAGLE build time (1,000 APA abstracts, 95,000 tokens; container with about 3
effective cores)

| Workers | Time |
|---|---|
| 1 | 43 s |
| 3 | 19 s |
| 4 | 17 s |

### Evaluation results (default settings, seed 0, up to 200 targets)

The full 27,560-document APA corpus is not available here (the CrumpLab/SemanticLibrarian
repository is not publicly readable), so the harness was run on the three corpora at
hand. Median rank of the target; top-10 rate in parentheses.

Simulation 2 (associates), 100-article APA sample:

| Method | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| beagle | 2.0 (0.76) | 1.0 (0.82) | 1.0 (0.95) | 1.0 (0.95) | 1.0 (0.99) |
| beagle-rp | 2.0 (0.78) | 1.0 (0.83) | 1.0 (0.94) | 1.0 (0.98) | 1.0 (0.99) |
| random | 11.0 (0.49) | 12.0 (0.46) | 7.0 (0.57) | 7.0 (0.58) | 5.5 (0.62) |
| wordmatch | 8.0 (0.52) | 8.8 (0.54) | 8.8 (0.56) | 7.5 (0.58) | 9.0 (0.55) |
| bm25 | 1.0 (0.83) | 1.0 (0.88) | 1.0 (0.99) | 1.0 (0.99) | 1.0 (1.00) |
| hybrid | 2.0 (0.83) | 1.0 (0.88) | 1.0 (0.99) | 1.0 (0.99) | 1.0 (1.00) |

Simulation 2 (associates), Alice in Wonderland, 1,631 sentences as documents:

| Method | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| beagle | 318 (0.13) | 276 (0.13) | 262 (0.13) | 158 (0.10) | 160 (0.10) |
| beagle-rp | 320 (0.13) | 270 (0.12) | 220 (0.15) | 156 (0.12) | 133 (0.10) |
| random | 598 (0.04) | 608 (0.03) | 480 (0.04) | 469 (0.02) | 408 (0.01) |
| wordmatch | 841 (0.07) | 840 (0.07) | 452 (0.10) | 370 (0.06) | 294 (0.06) |
| bm25 | 841 (0.09) | 840 (0.14) | 312 (0.24) | 118 (0.34) | 18.5 (0.46) |
| hybrid | 292 (0.12) | 284 (0.14) | 182 (0.19) | 80 (0.18) | 38 (0.25) |

Findings:

1. **The paper's central result replicates.** With associate queries, both BEAGLE
   variants beat the random-vector and word-match controls by a wide margin on every
   corpus; with the documents' own words (Simulation 1) all vector methods recover the
   target, as in the paper.
2. **BEAGLE and BEAGLE-RP agree** (mean Spearman rho 0.84-0.97 across corpora and query
   sizes), close to the paper's Fig. 4; agreement with the lexical methods is much lower.
3. **BM25 is a stronger keyword baseline than the paper's word match**, and on abstracts
   it does well even in Simulation 2. The likely reason is a property of the simulation:
   in a small corpus a word's nearest BEAGLE neighbor is often a word from the same
   abstract, so "associate" queries still share words with the target. On Alice, where
   documents are single sentences, BEAGLE beats BM25 for short associate queries and
   BM25 wins only when most of the sentence is used. Whether this holds at the scale of
   the full APA corpus is an open question.
4. **Hybrid ranking is a reasonable default for users** but not a free win: it tracks
   the better of its two parts on abstracts and lands between them on Alice. It stays
   opt-in (`--method hybrid`); semantic remains the default.

### Next

- Done: full APA evaluation and the outside-document Simulation 2 variant (Section 10).
- Phase 3, PDF ingestion.

## 10. Full APA corpus

The corpus (27,560 documents, `apa-data/all_journals_rev.csv`) was added to the
repository, and later removed from tracking and git-ignored ahead of making the
repository public. The development branch was then squash-merged into `master`, so the
file is not in `master`'s history. Full tables, commands and the comparison with the paper's figures are in
[docs/apa-evaluation.md](docs/apa-evaluation.md).

- **Scale.** 4.3 million tokens and 43,589 word types. The default BEAGLE build took
  8.6 minutes on 4 workers and produces a 491 MB library. Searches take about a second.
- **Replication.** Simulations 1 and 3 match the paper. In Simulation 2, BEAGLE beats
  the non-semantic control 5-6 times over.
- **The paper's examples reappear.** Reber's implicit-learning article retrieves
  nearly the same author list as Fig. 8.
- **Stricter Simulation 2** (`--associates outside-document`). When no substituted word
  occurs in the target, keyword methods fall below chance. BEAGLE keeps a median rank
  near 2,000 of 27,560, about ten times better than any non-semantic method. The
  paper's original procedure flatters keyword search at large query sizes, because
  associates often come from the same document.
- **Fix found by the full run.** The word-match control now counts distinct non-stop
  query words, the paper's "largest word overlap". Counting all occurrences including
  stop words gave median ranks of 200-370.
- **Recommendation.** Keep semantic search as the default, since it is the method that
  survives vocabulary mismatch. Offer hybrid as the safest single choice for users who
  mostly type the words they expect to find.

## 11. Phase 3: PDF ingestion

### Built

- **`ingest/pdf.py`.** One PDF becomes one `Document` with `chunks`. Text is read with
  pymupdf block by block in the file's own order, which follows the columns of most
  articles. pymupdf is an optional extra (`pip install "semantic-librarian[pdf]"`) and
  is also in the `dev` extra so CI runs the PDF tests.
- **Cleaning.** Margin text (top and bottom 10% of the page) that repeats on at least
  30% of the pages, numbers aside, is a running header or footer and is dropped, as are
  lone page numbers. Words hyphenated at a line end are rejoined when the next line
  starts in lower case. A block that starts in lower case continues the previous
  paragraph when that one ends in a hyphen or without sentence-final punctuation, so
  paragraphs survive column and page breaks. Ligatures are normalized (NFKC).
- **Chunking (`text/chunk.py`).** Paragraphs are packed into chunks of about 200 words
  (`--chunk-words`), never more than 1.5 times that; overlong paragraphs are cut at
  sentence ends. Chunks do not overlap and each records `page` and `page_end`.
- **Data model.** `Document.chunks` is a list of `Chunk(text, meta)`. A document with
  chunks is embedded from its chunks; its fields (the title) are for display only.
  Chunks are stored in a new `chunks` table (schema version 2; older libraries gain the
  table when opened).
- **Chunk space.** `sl build` writes a `chunk` space (labels `<document id>#<n>`) when
  the library holds chunked documents. Because chunks partition the text, the document
  vector of a word-sum model is exactly the sum of its chunk vectors (tested).
  `search(space="chunk")` returns hits carrying the chunk and its parent document;
  `where` filters apply to the parent; `similar` works between chunks and every other
  space.
- **CLI.** `sl add` accepts folders (searched at any depth for known extensions) and
  keeps going when one file cannot be read, reporting it and exiting with status 2.
  `sl info` shows the chunk count.
- **Metadata.** Title: PDF metadata if plausible, else the largest text on page 1, else
  the file name (`meta["title_source"]` records which). Metadata authors, split on ";",
  "and" and "&", become the `author` factor. The file's creation year is kept as
  `meta["file_year"]` but is not used as the `year` factor, since it is often not the
  publication year.

### Limits and follow-ups

- Only tested on generated PDFs and one single-page figure from the repository. It has
  not yet been run on a folder of real journal articles; that is the first thing to do.
- No OCR: PDFs without a text layer are rejected with a message.
- Tables, figure captions and reference lists are kept as ordinary text.
- Titles, authors and years from PDF metadata are unreliable. GROBID, or pairing a PDF
  with a BibTeX entry (Zotero exports link them), would fix this and is not built.
- Lexical and hybrid search rank documents only; chunk-level BM25 is not built.
- pymupdf is AGPL-3.0. It is an optional dependency and this project is
  GPL-2.0-or-later; whether that combination suits a public release is the owner's
  call. pypdf (BSD) is the usual alternative, with weaker layout handling.

## 12. Phase 4: web app (static first)

### Built

- **`sl export LIBRARY OUT`** (`export/bundle.py`) writes the web app plus a `data/`
  folder: a manifest, one vector file per space, labels, a light catalogue
  (`documents.json`: titles, years, factor levels as row numbers), and full records
  and chunk text in files of 500 so the browser fetches only what it shows. It refuses
  to write into a non-empty folder that is not an earlier export.
- **8-bit vectors.** Each row is divided by its largest absolute value and stored as
  -127..127, with the divisor in a scale file so that rows can still be summed into a
  query. Cosines against the exact vectors stay above 0.999 (tested).
  `--precision float32` exports exactly.
- **`sl serve LIBRARY`** exports into `LIBRARY/web/` when the library, the precision or
  the app has changed, then serves that folder on localhost with the standard
  library's HTTP server. There is no FastAPI and no live API yet.
- **Web app** (`web/`, Vite + Svelte 5, plain JavaScript). One page with two modes.
  "Search by text" covers the original free search (compound / AND / OR, year range).
  "Find similar" takes any item of any space and ranks any space against it, which
  covers the article neighborhood, author neighborhood and article-to-author tabs and
  every other pairing. Results list, MDS map with k-means groups (computed in the
  browser for the results shown, as the Shiny app did), and a detail panel with the
  record, clickable factor levels, an author's own documents, and "show the closest"
  buttons for moving between spaces. Light and dark themes; works at phone width.
- **Browser search is tested against Python.** `web/test/rank.mjs` runs the browser's
  own modules under Node on an exported bundle; `tests/test_export.py` checks that
  rankings and scores equal `Library.search` and `Library.similar` (float32 export),
  that the tokenizers agree on awkward text in both text modes, and that the map
  equals `classical_mds` up to mirroring. These tests are skipped without Node.
- **The built app is committed** in `src/semantic_librarian/web/` so that a user needs
  only `pip install`. File names carry no content hash, to keep the diffs small. After
  changing anything in `web/src`, run `npm run build` in `web/` and commit the result.

### Decisions taken here

- **Vite + Svelte instead of Astro.** The interface is one interactive page with no
  static content pages, the case in which Section 3.5 says not to force Astro. The
  Svelte components would move into Astro islands unchanged if corpus pages are wanted.
- **`sl serve` is a static file server for now**, so the command exists and the app has
  one code path. The FastAPI server can replace it behind the same command.

### Measured on the full APA library (27,560 documents)

- Export: 144 MB in about 2 s (95 MB vectors, 39 MB records, 3 MB catalogue).
- The browser loads the document and word vectors (73 MB) on the first search and
  other spaces when first used. Rankings and scores match the command line.

### Not built yet

- The live server API (`SearchProvider` with a server implementation): keyword and
  hybrid search in the browser, corpora too large to ship, adding documents from the
  UI.
- A whole-corpus map (WebGL scatter of every document); the map shows results only.
- Vocabulary pruning or dimension reduction for smaller exports.
- Shareable links (the query is not in the URL), and a Web Worker for scoring: a
  search blocks the page for a moment on large libraries.
- CI does not rebuild the app or check that the committed build matches `web/src`.
- Passage (chunk) results are implemented in the app but have only been exercised by
  tests, not on a real library of PDFs.

## 13. Phase 5: documentation and the R package

### Built

- **Documentation site** in `docs/`, built with mkdocs-material (`mkdocs.yml`, `docs`
  extra, `mkdocs build --strict` in CI). Pages: home; three tutorials; a guide (how it
  works, adding documents including PDFs, the web app, command reference, Python
  reference generated from docstrings, coming from the R package); and the existing
  APA evaluation.
- **Tutorials mirror the three R vignettes**, each with a runnable script in
  `examples/` that `tests/test_examples.py` runs:
  - `abstracts.py` / "Search a library of abstracts" follows *Semantic Librarian
    Tutorial* on the same 100 abstracts: text search, map, article and author
    neighbourhoods.
  - `bibliography.py` / "A library from a BibTeX file" follows *SL_bib* on the same
    `Crump.bib`.
  - `any_text.py` / "A library from any text" follows *Alice*: sentences as documents
    with a chapter factor, through the Python API, and the CSV route without code.
- Every command output printed in the tutorials was produced by running that command.
  The Alice tutorial prints none, because the book is not in the repository; its
  script is tested on a short stand-in text.
- **R package.** Already frozen under `legacy/R/` with a pointer in its README; that
  README now also points to `docs/guide/from-r.md`, which maps each R function to its
  replacement, says how to reproduce R's numbers (`legacy` text mode with
  `beagle-legacy`), and lists the defaults that changed.

### Left for the owner

- **Publishing the site.** Nothing deploys it. A GitHub Pages workflow is a few lines,
  but it makes the site public, so it waits on the decision about the repository.
- **The old pkgdown site** is still in `legacy/R/docs/`, untouched. If GitHub Pages
  served the old site from `/docs` on the default branch, that path now holds the new
  Markdown sources instead; check the Pages settings before merging.
- **Repository address.** The docs and README use `CrumpLab/RsemanticLibrarian`, as
  `pyproject.toml` does, while this work lives in `CrumpLabSandbox`.
- No screenshots or figures; maps are shown through the web app.

### A second, publishable example: NSF awards

`examples/nsf_awards.py` and the tutorial "One space, many kinds of things" build a
library from NSF's public-domain award data: fiscal year 2024, 11,687 awards merged
into 9,932 projects (collaborative awards share an abstract), 3.8 million tokens,
47,603 word types, with investigator, institution, state, program, division,
directorate and year as factors. The build took 174 s on 8 workers; the library is
420 MB and the web export 114 MB.

What it showed:

- Text to documents, programs, divisions and words gives sensible results, and so does
  factor to factor (institutions near Woods Hole, programs near Linguistics).
- Levels that sum many documents have cosines near 0.99 with each other, and their
  nearest words are generic grant language. Levels with one document are noisy as
  search targets. The tutorial says so. Whether to remove the component all documents
  share (centering, or dropping the first principal direction) is an open question
  worth measuring with `sl evaluate`.

Other public sources considered: arXiv abstracts (metadata is CC0; the closest
replacement for the APA corpus), Shakespeare's works with characters as factors, the
Federalist Papers, and Discogs (CC0, but 12 GB and little descriptive text). Wayback
Machine pages are not public domain.

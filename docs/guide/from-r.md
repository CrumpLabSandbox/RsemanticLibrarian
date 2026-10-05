# Coming from the R package

The R package `RsemanticLibrarian` is frozen. It stays installable, unchanged, from the
`legacy/R` folder of the repository:

```r
devtools::install_github("CrumpLab/RsemanticLibrarian", subdir = "legacy/R")
```

New work happens in the Python package described on this site. The three R vignettes
have Python counterparts under Tutorials.

## What replaces what

In R you called a function for each step and kept the pieces (dictionary, word
vectors, article vectors, author vectors) in your workspace. Here `sl build` does every
step and stores the results in the library folder.

| R function | Now |
|---|---|
| `sl_clean`, `sl_clean_vector`, `sl_corpus_dictionary`, `sl_word_ids_sentences` | done by `sl build`; the `legacy` text mode cleans exactly as R did |
| `sl_create_riv`, `sl_beagle_vectors` | the `beagle-legacy` embedder (identical numbers) or `beagle` (default) |
| `sl_article_vectors` | the `document` space |
| `sl_author_vectors` | the `author` space, and one space for every other factor |
| `sl_semantic_space` | `sl build`, or `Library.build()` |
| `get_search_terms` | `result.terms` and `result.unknown` from `Library.search()` |
| `get_search_article_similarities(query_type = 1, 2, 3)` | `sl search --mode compound`, `and`, `or` |
| `get_article_article_similarities` | `sl similar LIBRARY DOCUMENT_ID` |
| `get_author_similarities` | `sl similar LIBRARY "NAME" --space author` |
| `get_mds_article_fits`, `get_mds_author_fits` | `Library.project()`, or the map in the web app; `year_range` is `--years` |
| `cosine_x_to_m` | `semantic_librarian.search.similarity.cosine_scores` |
| the Shiny app | `sl serve`, and `sl export` for a hosted copy |

## Reproducing R results exactly

To get the same numbers as the R package, build with its cleaning and its model:

```bash
sl init mylib --text-mode legacy --embedder beagle-legacy
```

The test suite checks this combination against reference output from R: the cleaning,
the word vectors (given the same random vectors), the search functions and the MDS
coordinates.

## What changed in the defaults

These behaviours of the R package are kept only in the legacy mode and embedder.

- **BEAGLE now uses word order** as well as context. The R package used context only.
- **Authors are matched exactly.** R matched author names as regular expressions, so
  one author could pick up another's articles when one name contained the other.
- **Sentences** are split on sentence-ending punctuation, with abbreviations such as
  "e.g." and initials protected. R split on every period.
- **Letters of every script** are kept, with accents folded ("naïve" becomes "naive").
  R reduced all text to ASCII.
- **Very common words** are left out of document and query vectors.
- **An empty document** gets a zero vector. In R it received the first word's vector.
- **Single-word `and` / `or` queries** return that word's results. R returned nothing.

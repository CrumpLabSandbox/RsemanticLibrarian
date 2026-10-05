# How it works

## Words

The Semantic Librarian learns a vector for every word from the documents in the
library, using BEAGLE (Jones & Mewhort, 2007). Each word starts with a random vector.
A word's meaning vector is then built from two kinds of information:

- **Context**: the other words that occur in the same sentences.
- **Order**: the sequences the word occurs in, encoded with circular convolution.

Words used in similar ways end up with similar vectors. Nothing is downloaded and no
outside knowledge is used: a library knows only what its own documents say. This is why
results improve as a library grows, and why a word that never occurs in the library
cannot be searched for.

## Documents

A document's vector is the sum of the vectors of its words (Equation 8 of the paper).
Very common words such as "the" and "of" are left out of the sum.

## Factors

A factor is any grouping variable attached to documents: author, year, journal,
chapter, speaker. Each level of a factor gets a vector, the sum of the vectors of the
documents that carry it (Equation 11). An author is the sum of that author's documents.

## One space

Words, documents and factor levels all live in the same coordinate system. That is what
makes it possible to compare anything with anything: a query with documents, a document
with authors, a chapter with words. Each kind of item is called a *space*, and commands
take `--space` (what you start from or rank) and `--target` (what you rank instead).

Similarity is the cosine of the angle between two vectors: 1 for the same direction, 0
for unrelated, negative for opposed. Only compare scores within one search. Scores
between long documents, or between prolific authors, run high for everyone, because
sums of many words share a great deal.

## Queries

A query is treated like a small document. Its words are looked up, and:

- `compound` adds the word vectors and ranks by similarity to the sum;
- `and` multiplies each word's similarities, so a result must be close to every word;
- `or` takes the best similarity over the words, so a result may be close to any one.

## Keyword and hybrid search

Meaning-based search finds documents that never use your words. Sometimes you want the
opposite. `--method lexical` ranks documents by the words they actually contain (BM25),
and `--method hybrid` merges the two rankings. These rank documents only and are
available from the command line and Python, not yet in the web app.

## Embedders

| Name | Model |
|---|---|
| `beagle` (default) | BEAGLE with context and order information |
| `beagle-legacy` | context only, numerically identical to the R package |
| `beagle-rp` | order encoded with random permutations instead of convolution |
| `random` | the paper's control: random word vectors with no learning |

Choose with `sl build mylib --embedder beagle-rp`, and set parameters with `--param`,
for example `--param max_ngram=5` or `--dim 512`. Builds are reproducible: the same
documents, settings and seed give the same vectors, on any number of processor cores.

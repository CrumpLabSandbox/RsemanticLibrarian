# Evaluation on the full APA corpus

The paper's three simulations (Aujla, Crump, Cook & Jamieson, 2019), re-run with the
Python package on the same corpus: 27,560 titles, abstracts and keywords from eight
experimental psychology journals, 1894-2016.

The corpus is not distributed with this project, because the abstracts belong to their
publisher. The commands below assume you have your own copy at
`apa-data/all_journals_rev.csv`, a folder that git ignores.

## Reproduce

```bash
sl init libraries/apa
sl add libraries/apa apa-data/all_journals_rev.csv \
    --factor "author=authorlist:;" --factor year --factor journal
sl build libraries/apa                       # 8.6 min on 4 cores (about 3 effective)
sl evaluate libraries/apa -n 1000 --out apa_nearest.json                  # 8.7 min
sl evaluate libraries/apa -n 1000 --simulations 2 \
    --associates outside-document --out apa_outside.json                  # 4.8 min
python scripts/apa_paper_examples.py libraries/apa   # the examples in Figs. 5-8
```

Settings: default `beagle` (1,024 dimensions, context and order, n-grams up to 7, English
stop words excluded from context and composition), `beagle-rp` (3,000 dimensions), the
`random` control (1,024 dimensions), seed 0. The corpus tokenizes into 225,289 sentences,
4.3 million tokens and 43,589 word types. The same 1,000 target documents are used at
every query size. The 100% condition uses every token of the document. Ranks run from 1
(best) to 27,560; ties share their average rank.

Methods:

| Method | What it is |
|---|---|
| beagle, beagle-rp | the paper's two semantic models |
| random | the paper's non-semantic control: random vectors for words |
| wordmatch | the paper's word-match control: documents ranked by how many distinct non-stop query words they contain |
| bm25 | standard keyword ranking (Okapi BM25), not in the paper |
| hybrid | reciprocal rank fusion of beagle and bm25, equal weights, not in the paper |

## Simulation 1: recovering a document from a sample of its words

Median rank of the target (top-10 rate in parentheses).

| Method | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| beagle | 6.0 (0.56) | 1.0 (0.87) | 1.0 (0.99) | 1.0 (1.00) | 1.0 (1.00) |
| beagle-rp | 8.0 (0.53) | 1.0 (0.85) | 1.0 (0.99) | 1.0 (1.00) | 1.0 (1.00) |
| random | 4.5 (0.63) | 1.0 (0.92) | 1.0 (1.00) | 1.0 (1.00) | 1.0 (1.00) |
| wordmatch | 1.0 (0.89) | 1.0 (0.99) | 1.0 (1.00) | 1.0 (1.00) | 1.0 (1.00) |
| bm25 | 1.0 (0.96) | 1.0 (1.00) | 1.0 (1.00) | 1.0 (1.00) | 1.0 (1.00) |
| hybrid | 1.0 (0.81) | 1.0 (0.98) | 1.0 (1.00) | 1.0 (1.00) | 1.0 (1.00) |

As in the paper (Fig. 2), every method finds the target from 10% of its words on, and
all are within a median rank of about 10 at 5%. When the query uses the document's own
words, semantic vectors add nothing.

## Simulation 2: queries made of semantic associates

Each sampled word is replaced by its nearest neighbor in the BEAGLE word space. All
methods receive the same substituted queries.

**Nearest neighbor (the paper's procedure).**

| Method | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| beagle | 1,488 | 1,076 | 699 | 650 | 558 |
| beagle-rp | 1,675 | 1,105 | 896 | 865 | 777 |
| random | 7,717 | 6,191 | 5,003 | 3,671 | 3,028 |
| wordmatch | 5,808 | 4,449 | 1,776 | 868 | 475 |
| bm25 | 4,716 | 2,117 | 473 | 101 | 22 |
| hybrid | 1,410 | 617 | 220 | 61 | 22 |

**Nearest neighbor that does not occur in the target document.** This removes any
literal overlap between the substituted words and the target.

| Method | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| beagle | 2,584 | 2,010 | 1,681 | 1,628 | 1,635 |
| beagle-rp | 2,483 | 2,021 | 1,974 | 1,915 | 1,967 |
| random | 16,304 | 17,801 | 18,521 | 19,202 | 19,165 |
| wordmatch | 17,671 | 20,043 | 23,619 | 25,550 | 26,689 |
| bm25 | 17,671 | 20,043 | 23,619 | 25,550 | 26,689 |
| hybrid | 5,480 | 4,601 | 3,924 | 3,788 | 3,800 |

A random ranking would put the target at about 13,780.

## Simulation 3: agreement between methods

Mean Spearman correlation between complete document rankings for the Simulation 1
queries.

| Pair | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| beagle / beagle-rp | 0.954 | 0.967 | 0.978 | 0.982 | 0.984 |
| beagle / random | 0.196 | 0.239 | 0.314 | 0.358 | 0.404 |
| beagle / wordmatch | 0.318 | 0.361 | 0.443 | 0.495 | 0.541 |
| beagle / bm25 | 0.289 | 0.351 | 0.436 | 0.487 | 0.530 |
| beagle / hybrid | 0.763 | 0.783 | 0.815 | 0.833 | 0.849 |
| random / wordmatch | 0.456 | 0.489 | 0.525 | 0.525 | 0.519 |
| wordmatch / bm25 | 0.660 | 0.746 | 0.830 | 0.855 | 0.871 |

## Findings

1. **The paper's results replicate on its own corpus.** Simulation 1 matches Fig. 2.
   In Simulation 2, BEAGLE beats the non-semantic control by a factor of 5 to 6 with
   the paper's procedure. BEAGLE and BEAGLE-RP rankings agree at rho 0.95-0.98, higher
   than the paper's Fig. 4 (about 0.85-0.95), while agreement with the other methods
   is low (0.2-0.5).
2. **Semantic search is what works when the vocabulary differs.** With associates
   taken from outside the target document, keyword methods fall to worse than chance.
   Words that do match point to other documents. BEAGLE still ranks the target around
   1,600-2,600, about ten times better than any non-semantic method. This is the
   situation the Semantic Librarian was built for.
3. **The paper's Simulation 2 procedure favors keywords at large query sizes.** With
   plain nearest neighbors, BM25 beats BEAGLE from 50% of the document on (median 22
   against 558 at 100%). The stricter variant shows why: the associates of a document's
   words are often other words from the same document.
4. **Hybrid search is the safest single choice.** It is best or near best in the
   paper's Simulation 2 at every size. In the stricter variant it stays well ahead of
   the keyword methods (3,800-5,480) though behind pure BEAGLE. In Simulation 1 it finds
   the target in the top 10 for 81-100% of queries.
5. **The paper's word-match definition matters.** Counting every occurrence of every
   query word, stop words included, gives median ranks of 200-370 in Simulation 1
   instead of 1. The implementation counts distinct non-stop words, matching the
   paper's "largest word overlap".

## Same examples as the paper's figures

From `scripts/apa_paper_examples.py`; vectors are freshly learned, not the paper's.

- **Fig. 8**, authors closest to Reber (1989), "Implicit Learning And Tacit Knowledge":
  Reber, then Buss, Stanley, Blanchard-Fields, Cho, Druhan, Crossley, Willingham. The
  paper's list is Reber, Blanchard-Fields, Buss, Cho, Druhan, Stanley, Crossley,
  Redington, Willingham.
- **Fig. 7**, authors closest to Vokey: Jacoby, Brooks, Whittlesea, Humphreys, Hockley,
  Koriat, Hintzman. The paper shows Brooks, Jacoby, Glenberg, Bower, Hampton, Koriat.
- **Fig. 6**, articles closest to Aborn & Rubenstein (1952): Rubenstein & Aborn (1954),
  the paper's second hit, ranks fifth.
- **Fig. 5**, three-cluster map of the top 500 for "perception attention memory": an
  attention and perception cluster, a memory cluster, and a mixed attention-memory
  cluster between them, consistent with the paper's reading that attention bridges
  perception and memory.

# Search a library of abstracts

This tutorial follows the R package's "Semantic Librarian Tutorial" using the same data:
100 abstracts from the *Canadian Journal of Experimental Psychology*, 1947 to 1949. It
covers the four things the original interface did: search by text, map the results,
find similar articles, and find similar authors.

The sample is in the repository at `tests/fixtures/apa_sample/articles.json`, so run
the commands from a checkout. The whole tutorial is also a script,
`examples/abstracts.py`.

## Build the library

Each record in the file has a title, an abstract, keywords, an author list and a year.

```bash
sl init libraries/abstracts
sl add libraries/abstracts tests/fixtures/apa_sample/articles.json
sl build libraries/abstracts
```

```
built with beagle: document (100), word (2315), author (64), year (3)
```

Title, abstract and keywords were read as text. The author list and the year were
recognized as *factors*, so the build made four spaces: one point for each of the 100
documents, 2,315 words, 64 authors and 3 years.

## Search by text

```bash
sl search libraries/abstracts "president of psychology" -k 5
```

```
compound search in document space for: president psychology

   1   0.6300  Research Planning In The Canadian Psychological Association. I. Report On E
               articles-000040
   2   0.6091  Research Planning In Clinical Psychology. (1948)
               articles-000044
   3   0.6065  Canadian Psychology—Past, Present And Future. (1947)
               articles-000011
   4   0.5922  The Teaching Of Psychology In Canadian Universities. (1948)
               articles-000062
   5   0.5829  Canadian Theses In Psychology, 1948. (1949)
               articles-000087
```

The first line shows the words that were used. "of" is a stop word and was dropped. A
word that never occurs in the library cannot be used either, and is listed as ignored.
The number beside each result is its cosine similarity to the query, from -1 to 1,
and the line under each title is the document's id.

There are three ways to combine the words of a query:

| `--mode` | Ranks highly | R package `query_type` |
|---|---|---|
| `compound` (default) | documents close to the words taken together | 1 |
| `and` | documents close to every word | 2 |
| `or` | documents close to any one word | 3 |

Restrict results to a range of years with `--years 1947-1948`.

In Python:

```python
from semantic_librarian import Library

lib = Library.open("libraries/abstracts")
result = lib.search("president of psychology", mode="compound", k=25)
print(result.terms, result.unknown)  # ['president', 'psychology'] []
for hit in result.hits[:5]:
    print(hit.rank, round(hit.score, 3), hit.document.title())
```

## Map the results

The original interface drew the top results as a map: multidimensional scaling places
similar documents close together, and k-means colours them in groups. The web app does
this for every search:

```bash
sl serve libraries/abstracts --open
```

To get the coordinates yourself, for your own plot:

```python
projection = lib.project([hit.label for hit in result.hits], n_clusters=3)
projection.coords  # one (x, y) row per result
projection.clusters  # the group of each result, numbered from 0
```

```
  0.035   0.004  cluster 0  Research Planning In The Canadian Psychological As
  0.171   0.006  cluster 2  Research Planning In Clinical Psychology.
  0.068  -0.071  cluster 0  Canadian Psychology—Past, Present And Future.
```

The coordinates equal those of R's `cmdscale`, except that the map may be mirrored.

## Articles similar to an article

`sl similar` starts from an item that is already in the library. Documents are named by
their id.

```bash
sl similar libraries/abstracts articles-000001 -k 5
```

```
   1   1.0000  Foreword By The President. (1947)
               articles-000001
   2   0.6840  Constitution Of The Canadian Psychological Association. (1947)
               articles-000029
   3   0.6583  Membership List, September 1947. (1947)
               articles-000030
   4   0.6515  Proceedings Of The Annual Meeting Of The Canadian Psychological Association
               articles-000018
   5   0.6217  Proceedings Of The Annual Meeting Of The Psychological Association Of The P
               articles-000028
```

An item is always its own nearest neighbour.

## Authors similar to an author

An author's vector is the sum of the vectors of that author's documents, so authors who
write about similar things end up close together.

```bash
sl similar libraries/abstracts "Bernhardt, Karl S." --space author -k 5
```

```
   1   1.0000  Bernhardt, Karl S.
   2   0.7764  Mailloux, N.
   3   0.7563  Ketchum, J. D.
   4   0.7548  Nolast, Nofirst
   5   0.7183  Milner, Esther
```

("Nolast, Nofirst" is how this data set records items without an author.)

Because every space shares the same coordinates, you can also cross from one kind of
thing to another. `--target` chooses what to rank:

```bash
sl similar libraries/abstracts "Bernhardt, Karl S." --space author --target document -k 3
```

```
   1   0.8139  Canadian Psychology—Past, Present And Future. (1947)
               articles-000011
   2   0.7899  Review Of Psychology, Normal And Abnormal: An Introduction To The Study Of 
               articles-000026
   3   0.7764  Research Planning In Clinical Psychology. (1948)
               articles-000044
```

In Python these are `lib.similar("Bernhardt, Karl S.", space="author")` and
`lib.similar("Bernhardt, Karl S.", space="author", target="document")`.

## The full corpus

The paper's corpus, 27,560 abstracts from 1894 to 2016, builds the same way and takes a
few minutes. The results of the paper's simulations on it are in
[Evaluation on the APA corpus](../apa-evaluation.md).

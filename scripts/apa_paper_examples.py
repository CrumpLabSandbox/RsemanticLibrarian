"""Re-run the examples shown in the paper's figures (Figs. 5-8) on an APA library.

Build the library first (about 10 minutes on 4 cores)::

    sl init libraries/apa
    sl add libraries/apa apa-data/all_journals_rev.csv \
        --factor "author=authorlist:;" --factor year --factor journal
    sl build libraries/apa

then run::

    python scripts/apa_paper_examples.py libraries/apa
"""

import collections
import re
import sys
import time

from semantic_librarian import Library
from semantic_librarian.text import ENGLISH_STOPWORDS

lib = Library.open(sys.argv[1])
docs = lib.documents
by_title = collections.defaultdict(list)
for d in docs:
    by_title[d.fields.get("title", "").strip().lower()].append(d.id)


def doc_line(h):
    d = h.document
    authors = "; ".join(d.factors.get("author", [])[:2])
    year = d.factors.get("year", ["?"])[0]
    return f"{h.rank:>3} {h.score:.4f} {d.title()[:70]} | {authors} ({year})"


def find(title):
    ids = by_title[title.lower()]
    if not ids:
        raise SystemExit(f"title not found: {title}")
    return ids[0]


t = time.perf_counter()
print("== Fig 5: 'perception attention memory', OR search, top 10")
res = lib.search("perception attention memory", mode="or", k=500)
for h in res.hits[:10]:
    print(doc_line(h))
print(f"(search took {time.perf_counter() - t:.2f}s)")

print("\n== Fig 5: 3-cluster MDS map of the top 500; most frequent title words per cluster")
proj = lib.project([h.label for h in res.hits], n_clusters=3, seed=1)
for c in range(3):
    words = collections.Counter()
    members = [lab for lab, k in zip(proj.labels, proj.clusters, strict=True) if k == c]
    for lab in members:
        title = lib.document(lab).title().lower()
        words.update(
            w for w in re.findall(r"[a-z]+", title) if w not in ENGLISH_STOPWORDS and len(w) > 2
        )
    print(f"cluster {c} ({len(members)} articles):", ", ".join(w for w, _ in words.most_common(8)))

print("\n== Fig 6: articles similar to 'Information Theory And Immediate Recall.'")
for h in lib.similar(find("Information Theory And Immediate Recall."), k=6):
    print(doc_line(h))

print("\n== Fig 7: authors similar to 'Vokey, John R.'")
for h in lib.similar("Vokey, John R.", space="author", k=8):
    print(f"{h.rank:>3} {h.score:.4f} {h.label}")

print("\n== Fig 8: authors similar to 'Implicit Learning And Tacit Knowledge.'")
for h in lib.similar(
    find("Implicit Learning And Tacit Knowledge."), space="document", target="author", k=8
):
    print(f"{h.rank:>3} {h.score:.4f} {h.label}")

print("\n== words nearest 'memory', 'attention', 'perception'")
for w in ("memory", "attention", "perception"):
    print(w, "->", ", ".join(h.label for h in lib.similar(w, space="word", k=9)[1:]))

print("\n== 'implicit learning': semantic vs lexical vs hybrid, top 5")
for method in ("semantic", "lexical", "hybrid"):
    print(f"-- {method}")
    for h in lib.search("implicit learning", method=method, k=5).hits:
        print(doc_line(h))

print("\n== journals nearest 'Journal Of Experimental Psychology: Applied'")
for h in lib.similar("Journal Of Experimental Psychology: Applied", space="journal", k=9):
    print(f"{h.rank:>3} {h.score:.4f} {h.label}")

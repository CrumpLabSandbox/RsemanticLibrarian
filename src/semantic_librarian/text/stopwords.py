"""Stop word lists.

BEAGLE excludes very frequent function words from the context computation because they
co-occur with everything and would make all words similar (Jones & Mewhort, 2007).
"""

from __future__ import annotations

from collections.abc import Iterable

# The classic English list distributed with NLTK, plus a few corpus-generic additions.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren arent as at be because
    been before being below between both but by can cannot could couldn couldnt d did
    didn didnt do does doesn doesnt doing don dont down during each few for from further
    had hadn hadnt has hasn hasnt have haven havent having he her here hers herself him
    himself his how i if in into is isn isnt it its itself just ll m ma me might mightn
    more most must mustn my myself needn no nor not now o of off on once only or other
    our ours ourselves out over own re s same shan she should shouldn so some such t than
    that the their theirs them themselves then there these they this those through to too
    under until up ve very was wasn wasnt we were weren what when where which while who
    whom why will with won wont would wouldn y you your yours yourself yourselves also
    however thus may within without upon whether
    """.split()
)


def resolve_stopwords(spec: str | Iterable[str] | None) -> frozenset[str]:
    """Turn a stop word specification into a set.

    ``None`` or ``"none"`` gives an empty set, ``"english"`` the built-in list, and any
    iterable of strings is used as given.
    """
    if spec is None:
        return frozenset()
    if isinstance(spec, str):
        key = spec.lower()
        if key == "none":
            return frozenset()
        if key == "english":
            return ENGLISH_STOPWORDS
        raise ValueError(f"unknown stop word list {spec!r}; use 'english', 'none' or a list")
    return frozenset(spec)

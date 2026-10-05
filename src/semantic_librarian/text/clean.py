"""Text cleaning.

Two modes are provided:

``legacy``
    Reproduces the R package exactly: ``LSAfun::breakdown()`` (LSAfun <= 0.6.2) followed
    by ``qdapRegex::rm_white_lead_trail()``, as used by ``sl_clean()``,
    ``sl_corpus_dictionary()`` and ``sl_clean_vector()``. Text is lower-cased, folded to
    ASCII, and every punctuation character and digit becomes a word boundary. Sentences
    are split on every period, so "e.g." and "3.14" create extra sentence breaks.

``unicode``
    The default for new libraries. Letters from any script are kept, accents are folded
    ("naïve" -> "naive"), apostrophes inside words are dropped ("don't" -> "dont"), and
    tokens may mix letters and digits ("n400", "p300") but pure numbers are dropped.
    Sentences are split on terminal punctuation followed by whitespace, with common
    abbreviations and initials protected.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

TextMode = Literal["legacy", "unicode"]

# Replacements applied by LSAfun::breakdown() before transliteration.
_BREAKDOWN_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("ä", "ae"),
    ("ö", "oe"),
    ("ü", "ue"),
    ("à", "a"),
    ("á", "a"),
    ("â", "a"),
    ("è", "e"),
    ("é", "e"),
    ("ê", "e"),
    ("ì", "i"),
    ("í", "i"),
    ("î", "i"),
    ("ò", "o"),
    ("ó", "o"),
    ("ô", "o"),
    ("ù", "u"),
    ("ú", "u"),
    ("û", "u"),
    ("ß", "ss"),
)

# Letters that glibc's ASCII//TRANSLIT maps but Unicode decomposition does not.
_TRANSLIT: dict[str, str] = {
    "æ": "ae",
    "œ": "oe",
    "ø": "o",
    "ł": "l",
    "đ": "d",
    "ð": "d",
    "þ": "th",
    "ı": "i",
    "ħ": "h",
    "ŋ": "ng",
    "ĸ": "q",
    "ŀ": "l",
    "µ": "u",
    "ſ": "s",
    "—": "--",
    "–": "-",
    "‐": "-",
    "‑": "-",
    "−": "-",
    "“": '"',
    "”": '"',
    "„": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "«": "<<",
    "»": ">>",
}

_ASCII_PUNCT = re.compile(r"[!-/:-@\[-`{-~]")
_DIGIT = re.compile(r"[0-9]")
_R_WHITESPACE = re.compile(r"[ \t\n\r\f\v]+")


@lru_cache(maxsize=4096)
def _ascii_char(ch: str) -> str:
    if ch in _TRANSLIT:
        return _TRANSLIT[ch]
    decomposed = unicodedata.normalize("NFKD", ch)
    ascii_part = "".join(c for c in decomposed if ord(c) < 128)
    return ascii_part if ascii_part else "?"


def _to_ascii(text: str) -> str:
    if text.isascii():
        return text
    return "".join(c if ord(c) < 128 else _ascii_char(c) for c in text)


def breakdown(text: str) -> str:
    """Port of ``LSAfun::breakdown()`` from LSAfun 0.6.2.

    Lower-cases, folds to ASCII, and replaces punctuation, digits and newlines with spaces.
    Runs of whitespace are *not* collapsed, matching the original.
    """
    x = text.lower()
    for src, dst in _BREAKDOWN_REPLACEMENTS:
        if src in x:
            x = x.replace(src, dst)
    x = _to_ascii(x)
    x = _ASCII_PUNCT.sub(" ", x)
    x = _DIGIT.sub(" ", x)
    return x.replace("\n", " ")


def legacy_tokens(text: str) -> list[str]:
    """Tokens exactly as ``sl_clean()`` / ``sl_corpus_dictionary()`` produce them."""
    return [t for t in _R_WHITESPACE.split(breakdown(text)) if t]


def legacy_sentences(text: str) -> list[list[str]]:
    """Sentences exactly as ``sl_clean_vector()`` produces them.

    Mirrors R's ``strsplit(text, "[.]")``: every period is a boundary, and a single trailing
    empty piece is dropped. Pieces that clean to nothing are kept as empty lists, as in R.
    """
    if text == "":
        return []
    pieces = text.split(".")
    if text.endswith("."):
        pieces = pieces[:-1]
    return [legacy_tokens(p) for p in pieces]


# --- unicode mode -------------------------------------------------------------------

_APOSTROPHES = re.compile(r"(?<=\w)['’](?=\w)")
_UNICODE_TOKEN = re.compile(r"[^\W_]*[^\W\d_][^\W_]*")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def unicode_tokens(text: str) -> list[str]:
    """Tokenize text keeping letters from any script, folding case and accents."""
    folded = _fold(_APOSTROPHES.sub("", text))
    return [t for t in _UNICODE_TOKEN.findall(folded) if t]


_ABBREVIATIONS = frozenset(
    """
    e.g i.e etc al vs cf fig figs eq eqs no nos vol vols pp p ed eds dr mr mrs ms prof
    jr sr st approx ca resp viz inc ltd co corp dept univ jan feb mar apr jun jul aug sep
    sept oct nov dec exp expt ref refs ch sec tab
    """.split()
)
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+|\n\s*\n")
_LAST_WORD = re.compile(r"(\S+)\.$")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation or blank lines.

    Abbreviations such as "e.g.", "et al." and "Fig.", single-letter initials ("J. Smith")
    and decimals do not end a sentence.
    """
    pieces: list[str] = []
    buffer = ""
    pos = 0
    for match in _SENTENCE_END.finditer(text):
        chunk = text[pos : match.end()]
        pos = match.end()
        candidate = (buffer + chunk) if buffer else chunk
        stripped = candidate.rstrip()
        last = _LAST_WORD.search(stripped)
        if last and not match.group(0).startswith("\n"):
            word = last.group(1).lower().lstrip("(\"'“‘[")
            if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                buffer = candidate
                continue
        pieces.append(stripped)
        buffer = ""
    tail = (buffer + text[pos:]).strip()
    if tail:
        pieces.append(tail)
    return [p.strip() for p in pieces if p.strip()]


@dataclass(frozen=True)
class TextProcessor:
    """Turns raw text into tokens and sentences according to a cleaning mode."""

    mode: TextMode = "unicode"

    def __post_init__(self) -> None:
        if self.mode not in ("legacy", "unicode"):
            raise ValueError(f"unknown text mode {self.mode!r}; use 'legacy' or 'unicode'")

    def tokens(self, text: str) -> list[str]:
        if self.mode == "legacy":
            return legacy_tokens(text)
        return unicode_tokens(text)

    def sentences(self, text: str) -> list[list[str]]:
        if self.mode == "legacy":
            return legacy_sentences(text)
        return [unicode_tokens(s) for s in split_sentences(text)]

    def sentences_of(self, texts: Iterable[str]) -> list[list[str]]:
        out: list[list[str]] = []
        for t in texts:
            out.extend(self.sentences(t))
        return out

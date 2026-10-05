// Tokenizers matching semantic_librarian/text/clean.py, so that a query typed in the
// browser is cut into the same words the library was built from.

const APOSTROPHE = /(?<=[\p{L}\p{N}_])['’](?=[\p{L}\p{N}_])/gu;
const TOKEN = /[\p{L}\p{N}]*\p{L}[\p{L}\p{N}]*/gu;
const MARKS = /\p{M}/gu;

/** "unicode" mode: letters of any script, case and accents folded, pure numbers dropped. */
export function unicodeTokens(text) {
  const folded = text
    .replace(APOSTROPHE, "")
    .toLowerCase()
    .replace(/ß/g, "ss")
    .normalize("NFKD")
    .replace(MARKS, "");
  return folded.match(TOKEN) ?? [];
}

const BREAKDOWN = { ä: "ae", ö: "oe", ü: "ue", ß: "ss" };
const TRANSLIT = {
  æ: "ae", œ: "oe", ø: "o", ł: "l", đ: "d", ð: "d", þ: "th", ı: "i", ħ: "h", ŋ: "ng",
  ĸ: "q", ŀ: "l", µ: "u", ſ: "s", "—": "--", "–": "-", "‐": "-", "‑": "-", "−": "-",
  "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": "'", "«": "<<", "»": ">>",
};

function asciiChar(ch) {
  if (ch in TRANSLIT) return TRANSLIT[ch];
  const ascii = ch.normalize("NFKD").replace(/[^\x00-\x7f]/g, "");
  return ascii || "?";
}

/** "legacy" mode: the R package's cleaning. ASCII only; punctuation and digits split words. */
export function legacyTokens(text) {
  const ascii = text
    .toLowerCase()
    .replace(/[äöüß]/g, (c) => BREAKDOWN[c])
    .replace(/[^\x00-\x7f]/gu, asciiChar);
  return ascii
    .replace(/[!-/:-@[-`{-~0-9]/g, " ")
    .split(/[ \t\n\r\f\v]+/)
    .filter(Boolean);
}

export function tokenize(text, mode = "unicode") {
  return mode === "legacy" ? legacyTokens(text) : unicodeTokens(text);
}

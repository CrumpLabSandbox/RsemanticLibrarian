// Vector search over an exported space, mirroring semantic_librarian/search/similarity.py
// and Library.search. A space is { n, dim, data, scale, labels }: `data` holds the rows
// end to end (Int8Array or Float32Array) and `scale`, when present, the factor that turns
// a stored row back into the original vector.

/** Lengths of the stored rows. A row's scale cancels out of every cosine. */
export function norms(space) {
  if (!space.norms) {
    const { n, dim, data } = space;
    const out = new Float64Array(n);
    for (let r = 0, off = 0; r < n; r++, off += dim) {
      let s = 0;
      for (let d = 0; d < dim; d++) s += data[off + d] * data[off + d];
      out[r] = Math.sqrt(s);
    }
    space.norms = out;
  }
  return space.norms;
}

/** The original vector of one row. */
export function rowVector(space, row) {
  const { dim, data, scale } = space;
  const k = scale ? scale[row] : 1;
  const out = new Float64Array(dim);
  for (let d = 0, off = row * dim; d < dim; d++) out[d] = data[off + d] * k;
  return out;
}

/** The sum of several rows: how a query is built from its words (paper Eq. 9). */
export function sumRows(space, rows) {
  const { dim, data, scale } = space;
  const out = new Float64Array(dim);
  for (const row of rows) {
    const k = scale ? scale[row] : 1;
    for (let d = 0, off = row * dim; d < dim; d++) out[d] += data[off + d] * k;
  }
  return out;
}

/** Cosine between `q` and every row of the space; zero vectors score 0. */
export function cosineAll(q, space) {
  const { n, dim, data } = space;
  const rowNorms = norms(space);
  let qn = 0;
  for (let d = 0; d < dim; d++) qn += q[d] * q[d];
  qn = Math.sqrt(qn);
  const out = new Float64Array(n);
  if (qn === 0) return out;
  for (let r = 0, off = 0; r < n; r++, off += dim) {
    let s = 0;
    for (let d = 0; d < dim; d++) s += q[d] * data[off + d];
    const denom = qn * rowNorms[r];
    out[r] = denom > 0 ? s / denom : 0;
  }
  return out;
}

/** Combine per-word cosines: "and" multiplies them, "or" takes the best after rescaling. */
export function combineTermScores(termScores, mode) {
  if (termScores.length === 1) return termScores[0].slice();
  const n = termScores[0].length;
  const out = new Float64Array(n);
  if (mode === "and") {
    out.fill(1);
    for (const scores of termScores) for (let i = 0; i < n; i++) out[i] *= scores[i];
    return out;
  }
  out.fill(-Infinity);
  for (const scores of termScores) {
    let top = -Infinity;
    for (let i = 0; i < n; i++) if (scores[i] > top) top = scores[i];
    const scale = Math.abs(top) || 1;
    for (let i = 0; i < n; i++) if (scores[i] / scale > out[i]) out[i] = scores[i] / scale;
  }
  return out;
}

/** Row numbers of the highest scores, best first; ties keep row order. */
export function rank(scores, k = null, keep = null) {
  const rows = [];
  for (let i = 0; i < scores.length; i++) if (!keep || keep(i)) rows.push(i);
  rows.sort((a, b) => scores[b] - scores[a] || a - b);
  return k == null ? rows : rows.slice(0, k);
}

/**
 * The query words used for search. Words outside the vocabulary are set aside, and stop
 * words are dropped when the model composes without them (unless nothing else is left).
 */
export function queryTerms(tokens, wordIndex, stopwords, composeWithoutStopwords) {
  const distinct = [...new Set(tokens)];
  const known = distinct.filter((t) => wordIndex.has(t));
  const unknown = distinct.filter((t) => !wordIndex.has(t));
  let terms = known;
  if (composeWithoutStopwords) {
    const content = known.filter((t) => !stopwords.has(t));
    if (content.length) terms = content;
  }
  return { terms, unknown };
}

/** Scores of every row of `target` for a free-text query's terms. */
export function searchScores(terms, words, wordIndex, target, mode = "compound") {
  const rows = terms.map((t) => wordIndex.get(t));
  if (mode === "compound") return cosineAll(sumRows(words, rows), target);
  return combineTermScores(
    rows.map((r) => cosineAll(rowVector(words, r), target)),
    mode,
  );
}

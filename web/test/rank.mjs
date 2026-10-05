// Ranks queries against an exported bundle with the same code the browser runs, and
// prints the results as JSON. Used by tests/test_export.py to compare with Python.
//
//   node web/test/rank.mjs <bundle>/data <requests.json>
import { readFile } from "node:fs/promises";
import path from "node:path";
import { openBundle } from "../src/lib/bundle.js";
import { classicalMDS, kmeans } from "../src/lib/mds.js";
import { cosineAll, queryTerms, rank, rowVector, searchScores } from "../src/lib/search.js";
import { tokenize } from "../src/lib/text.js";

const [dataDir, requestFile] = process.argv.slice(2);
const read = async (file, kind) => {
  const bytes = await readFile(path.join(dataDir, file));
  if (kind === "json") return JSON.parse(bytes.toString("utf8"));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const bundle = await openBundle(read);
const requests = JSON.parse(await readFile(requestFile, "utf8"));
const out = [];
for (const request of requests) {
  if (request.tokenize !== undefined) {
    out.push(tokenize(request.tokenize, request.mode));
    continue;
  }
  const target = await bundle.space(request.target);
  let scores;
  let extra = {};
  if (request.query !== undefined) {
    const words = await bundle.space("word");
    const tokens = tokenize(request.query, bundle.manifest.text_mode);
    const { terms, unknown } = queryTerms(
      tokens,
      bundle.index("word"),
      bundle.stopwords,
      bundle.manifest.compose_without_stopwords,
    );
    scores = searchScores(terms, words, bundle.index("word"), target, request.mode);
    extra = { terms, unknown };
  } else {
    const source = await bundle.space(request.space);
    scores = cosineAll(rowVector(source, bundle.index(request.space).get(request.item)), target);
  }
  const keep =
    request.target === "word" ? (i) => !bundle.stopwords.has(target.labels[i]) : null;
  const rows = rank(scores, request.k, keep);
  const result = {
    ...extra,
    labels: rows.map((r) => target.labels[r]),
    scores: rows.map((r) => scores[r]),
  };
  if (request.map) {
    const coords = classicalMDS(rows.map((r) => rowVector(target, r)));
    result.coords = coords;
    result.clusters = kmeans(coords, request.map);
  }
  out.push(result);
}
console.log(JSON.stringify(out));

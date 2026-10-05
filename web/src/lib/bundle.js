// Reads an exported library (see semantic_librarian/export/bundle.py). `read(path, kind)`
// fetches one file of the data folder as "json" or as an ArrayBuffer ("buffer"), so the
// same code runs in the browser (fetch) and in Node tests (the file system).

export function browserReader(base = "data/") {
  return async (path, kind) => {
    const response = await fetch(base + path);
    if (!response.ok) throw new Error(`could not load ${path} (${response.status})`);
    return kind === "json" ? response.json() : response.arrayBuffer();
  };
}

export async function openBundle(read, onStatus = () => {}) {
  const manifest = await read("manifest.json", "json");
  if (manifest.format !== 1) throw new Error(`unsupported export format ${manifest.format}`);
  const meta = new Map(manifest.spaces.map((s) => [s.name, s]));
  onStatus("Loading the catalogue");
  const [documents, chunks, labelLists] = await Promise.all([
    read("documents.json", "json"),
    manifest.n_chunks ? read("chunks.json", "json") : null,
    Promise.all(manifest.spaces.map((s) => read(s.labels, "json"))),
  ]);
  const labels = new Map(manifest.spaces.map((s, i) => [s.name, labelLists[i]]));
  const spaces = new Map();
  const shards = new Map();
  const indexes = new Map();

  function shard(folder, row) {
    const number = Math.floor(row / manifest.shard);
    const path = `${folder}/${String(number).padStart(5, "0")}.json`;
    if (!shards.has(path)) shards.set(path, read(path, "json"));
    return shards.get(path).then((records) => records[row % manifest.shard]);
  }

  return {
    manifest,
    documents,
    chunks,
    stopwords: new Set(manifest.stopwords),
    spaceNames: manifest.spaces.map((s) => s.name),
    meta: (name) => meta.get(name),
    labels: (name) => labels.get(name),
    megabytes: (name) => {
      const m = meta.get(name);
      return (m.n * m.dim * (m.dtype === "int8" ? 1 : 4)) / 1e6;
    },
    isLoaded: (name) => spaces.has(name),

    /** Label -> row number for a space. */
    index(name) {
      if (!indexes.has(name)) indexes.set(name, new Map(labels.get(name).map((l, i) => [l, i])));
      return indexes.get(name);
    },

    /** The vectors of a space, fetched on first use. */
    space(name) {
      if (!spaces.has(name)) {
        const m = meta.get(name);
        const load = async () => {
          const [vectors, scale] = await Promise.all([
            read(m.vectors, "buffer"),
            m.scale ? read(m.scale, "buffer") : null,
          ]);
          return {
            name,
            n: m.n,
            dim: m.dim,
            data: m.dtype === "int8" ? new Int8Array(vectors) : new Float32Array(vectors),
            scale: scale ? new Float32Array(scale) : null,
            labels: labels.get(name),
          };
        };
        spaces.set(name, load());
      }
      return spaces.get(name);
    },

    /** The full record of a document: { id, fields, meta, factors, chunks? }. */
    document: (row) => shard("docs", row),
    chunkText: (row) => shard("chunks", row),
  };
}

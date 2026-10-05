// A 2-D map of a handful of items: classical multidimensional scaling of cosine distances
// (R's cmdscale, as in the original interface) followed by k-means on the coordinates.

function cosineMatrix(vectors) {
  const n = vectors.length;
  const lengths = vectors.map((v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)));
  const sims = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      const a = vectors[i];
      const b = vectors[j];
      for (let d = 0; d < a.length; d++) s += a[d] * b[d];
      const denom = lengths[i] * lengths[j];
      sims[i][j] = sims[j][i] = denom > 0 ? s / denom : 0;
    }
  }
  return sims;
}

/** Eigenvalues and eigenvectors of a symmetric matrix by cyclic Jacobi rotations. */
function symmetricEigen(a) {
  const n = a.length;
  const v = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    row[i] = 1;
    return row;
  });
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: a.map((row, i) => row[i]), vectors: v };
}

/** Coordinates [[x, y], ...] whose distances approximate 1 - cosine between the vectors. */
export function classicalMDS(vectors) {
  const n = vectors.length;
  if (n < 2) return vectors.map(() => [0, 0]);
  const sims = cosineMatrix(vectors);
  const d2 = sims.map((row) => Array.from(row, (s) => (1 - s) ** 2));
  const rowMean = d2.map((row) => row.reduce((s, x) => s + x, 0) / n);
  const mean = rowMean.reduce((s, x) => s + x, 0) / n;
  const b = d2.map((row, i) =>
    Float64Array.from(row, (x, j) => -0.5 * (x - rowMean[i] - rowMean[j] + mean)),
  );
  const { values, vectors: eig } = symmetricEigen(b);
  const top = values
    .map((value, i) => [value, i])
    .sort((x, y) => y[0] - x[0])
    .slice(0, 2);
  return Array.from({ length: n }, (_, i) =>
    top.map(([value, col]) => eig[i][col] * Math.sqrt(Math.max(value, 0))),
  );
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

/** Cluster number (0-based) for each point; the best of several seeded k-means runs. */
export function kmeans(points, k, seed = 0) {
  const n = points.length;
  k = Math.max(1, Math.min(k, n));
  if (k === 1) return new Array(n).fill(0);
  const random = mulberry32(seed + 1);
  let best = null;
  for (let attempt = 0; attempt < 8; attempt++) {
    // k-means++ starting centres
    const centres = [points[Math.floor(random() * n)]];
    while (centres.length < k) {
      const d = points.map((p) => Math.min(...centres.map((c) => dist2(p, c))));
      let pick = random() * d.reduce((s, x) => s + x, 0);
      let i = 0;
      while (i < n - 1 && (pick -= d[i]) > 0) i++;
      centres.push(points[i]);
    }
    let assign = new Array(n).fill(0);
    for (let iter = 0; iter < 100; iter++) {
      const next = points.map((p) => {
        let arg = 0;
        for (let c = 1; c < k; c++) if (dist2(p, centres[c]) < dist2(p, centres[arg])) arg = c;
        return arg;
      });
      const same = next.every((c, i) => c === assign[i]);
      assign = next;
      if (same && iter > 0) break;
      for (let c = 0; c < k; c++) {
        const members = points.filter((_, i) => assign[i] === c);
        if (members.length) {
          centres[c] = [
            members.reduce((s, p) => s + p[0], 0) / members.length,
            members.reduce((s, p) => s + p[1], 0) / members.length,
          ];
        }
      }
    }
    const inertia = points.reduce((s, p, i) => s + dist2(p, centres[assign[i]]), 0);
    if (!best || inertia < best.inertia) best = { inertia, assign };
  }
  // number clusters in order of first appearance, so colours are stable
  const order = new Map();
  return best.assign.map((c) => {
    if (!order.has(c)) order.set(c, order.size);
    return order.get(c);
  });
}

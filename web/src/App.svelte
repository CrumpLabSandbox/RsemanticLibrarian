<script>
  import { onMount } from "svelte";
  import { browserReader, openBundle } from "./lib/bundle.js";
  import { classicalMDS, kmeans } from "./lib/mds.js";
  import { capital, plural, singular } from "./lib/names.js";
  import { cosineAll, queryTerms, rank, rowVector, searchScores } from "./lib/search.js";
  import { tokenize } from "./lib/text.js";
  import Map from "./Map.svelte";
  import Picker from "./Picker.svelte";

  const STYLES = [
    ["compound", "the words together"],
    ["and", "every word"],
    ["or", "any word"],
  ];

  let bundle = $state.raw(null);
  let status = $state("Opening the library");
  let error = $state(null);
  let busy = $state(false);

  // what is being asked
  let mode = $state("search");
  let query = $state("");
  let style = $state("compound");
  let source = $state("document");
  let sourceRow = $state(null);
  let target = $state("document");
  let k = $state(25);
  let yearFrom = $state("");
  let yearTo = $state("");
  let nClusters = $state(3);

  // what came back
  let result = $state.raw(null);
  let coords = $state.raw([]);
  let snippets = $state.raw({});
  let selected = $state.raw(null);
  let record = $state.raw(null);
  let side = $state(null);

  let docs = $derived(bundle?.documents);
  let hasYears = $derived(docs?.year.some((y) => y != null) ?? false);
  let yearFilter = $derived(hasYears && (target === "document" || target === "chunk"));
  let clusters = $derived(coords.length ? kmeans(coords, nClusters) : []);
  let points = $derived(
    result && coords.length === result.hits.length
      ? result.hits.map((hit, i) => ({
          row: hit.row,
          rank: i + 1,
          label: labelOf(result.target, hit.row),
          x: coords[i][0],
          y: coords[i][1],
          cluster: clusters[i] ?? 0,
        }))
      : [],
  );
  let pickerItems = $derived(bundle ? itemNames(source) : []);

  onMount(async () => {
    try {
      bundle = await openBundle(browserReader(), (message) => (status = message));
      document.title = `${bundle.manifest.name} · Semantic Librarian`;
      status = null;
    } catch (problem) {
      error = `${problem.message}. This page needs the data folder written by "sl export".`;
      status = null;
    }
  });

  // -- names ---------------------------------------------------------------------------

  function itemNames(space) {
    return space === "document" ? docs.title : bundle.labels(space);
  }

  function pages(row) {
    const [first, last] = bundle.chunks.pages[row];
    if (first == null) return "";
    return first === last ? `p. ${first}` : `pp. ${first}–${last}`;
  }

  function labelOf(space, row) {
    if (space === "document") return docs.title[row];
    if (space === "chunk") return `${docs.title[bundle.chunks.doc[row]]}, ${pages(row)}`;
    return bundle.labels(space)[row];
  }

  function levels(space, docRow) {
    const rows = docs.factors[space]?.[docRow] ?? [];
    return rows.map((row) => ({ row, label: bundle.labels(space)[row] }));
  }

  function byline(docRow) {
    const authors = levels("author", docRow).map((a) => a.label);
    const shown = authors.length > 4 ? [...authors.slice(0, 3), "et al."] : authors;
    return [shown.join("; "), docs.year[docRow]].filter(Boolean).join(" · ");
  }

  const counts = {};
  function documentCount(space, row) {
    if (!docs.factors[space]) return null;
    if (!counts[space]) {
      counts[space] = new Uint32Array(bundle.meta(space).n);
      for (const rows of docs.factors[space]) for (const r of rows) counts[space][r]++;
    }
    return counts[space][row];
  }

  function documentsOf(space, row) {
    const out = [];
    const lists = docs.factors[space] ?? [];
    for (let d = 0; d < lists.length && out.length < 200; d++) {
      if (lists[d].includes(row)) out.push(d);
    }
    return out;
  }

  // -- searching -----------------------------------------------------------------------

  const pause = () => new Promise((resolve) => setTimeout(resolve, 0));

  async function load(space) {
    if (!bundle.isLoaded(space)) {
      status = `Loading ${plural(space)} (${Math.round(bundle.megabytes(space))} MB)`;
    }
    const loaded = await bundle.space(space);
    status = null;
    return loaded;
  }

  function keepFor(space) {
    if (space === "word") {
      const words = bundle.labels("word");
      return (i) => !bundle.stopwords.has(words[i]);
    }
    const from = parseInt(yearFrom, 10);
    const to = parseInt(yearTo, 10);
    if (!yearFilter || (Number.isNaN(from) && Number.isNaN(to))) return null;
    const lo = Number.isNaN(from) ? -Infinity : from;
    const hi = Number.isNaN(to) ? Infinity : to;
    const inRange = (docRow) => {
      const year = parseInt(String(docs.year[docRow] ?? "").slice(0, 4), 10);
      return year >= lo && year <= hi;
    };
    return space === "chunk" ? (i) => inRange(bundle.chunks.doc[i]) : inRange;
  }

  let runs = 0;
  async function run() {
    if (!bundle) return;
    const id = ++runs;
    const asking = mode === "search" ? query.trim() !== "" : sourceRow != null;
    if (!asking) {
      result = null;
      coords = [];
      return;
    }
    busy = true;
    error = null;
    try {
      const space = await load(target);
      let scores = null;
      let terms = [];
      let unknown = [];
      if (mode === "search") {
        const words = await load("word");
        const index = bundle.index("word");
        const tokens = tokenize(query, bundle.manifest.text_mode);
        ({ terms, unknown } = queryTerms(
          tokens,
          index,
          bundle.stopwords,
          bundle.manifest.compose_without_stopwords,
        ));
        await pause();
        if (terms.length) scores = searchScores(terms, words, index, space, style);
      } else {
        const from = await load(source);
        await pause();
        scores = cosineAll(rowVector(from, sourceRow), space);
      }
      if (id !== runs) return;
      const rows = scores ? rank(scores, k, keepFor(target)) : [];
      result = {
        mode,
        target,
        terms,
        unknown,
        from: mode === "similar" ? { space: source, row: sourceRow } : null,
        hits: rows.map((row) => ({ row, score: scores[row] })),
      };
      coords = classicalMDS(rows.map((row) => rowVector(space, row)));
      if (target === "chunk") {
        const texts = await Promise.all(rows.map((row) => bundle.chunkText(row)));
        if (id === runs) snippets = Object.fromEntries(rows.map((row, i) => [row, texts[i]]));
      }
    } catch (problem) {
      error = problem.message;
    } finally {
      if (id === runs) {
        busy = false;
        status = null;
      }
    }
  }

  function findSimilar(space, row, to) {
    mode = "similar";
    source = space;
    sourceRow = row;
    target = to;
    run();
    window.scrollTo({ top: 0, behavior: "smooth" });
    side?.scrollTo({ top: 0 });
  }

  let selections = 0;
  async function select(space, row) {
    const id = ++selections;
    selected = { space, row };
    record = null;
    // the side panel scrolls on its own: bring the newly selected item into view
    side?.querySelector(".detail")?.scrollIntoView({ block: "nearest" });
    if (space === "document") {
      const loaded = await bundle.document(row);
      if (id === selections) record = loaded;
    } else if (space === "chunk") {
      const text = await bundle.chunkText(row);
      if (id === selections) record = { text };
    }
  }

  function set(change) {
    change();
    run();
  }

  const isLink = (value) => typeof value === "string" && /^https?:\/\/\S+$/.test(value);
  const show = (value) => (typeof value === "object" ? JSON.stringify(value) : String(value));
  const snippet = (text) => (text.length > 260 ? `${text.slice(0, 260).trimEnd()}…` : text);
</script>

<header>
  <div>
    <h1>{bundle?.manifest.name ?? "Semantic Librarian"}</h1>
    {#if bundle}
      <p class="muted">
        {bundle.manifest.n_documents.toLocaleString()} documents{#if bundle.manifest.n_chunks},
          {bundle.manifest.n_chunks.toLocaleString()} passages{/if},
        {bundle.meta("word").n.toLocaleString()} words. Search runs in this browser, by meaning
        rather than exact wording.
      </p>
    {/if}
  </div>
  <span class="brand">Semantic Librarian</span>
</header>

{#if bundle}
  <form
    class="controls"
    onsubmit={(event) => {
      event.preventDefault();
      run();
    }}
  >
    <div class="modes" role="tablist" aria-label="What to do">
      <button
        type="button"
        role="tab"
        aria-selected={mode === "search"}
        onclick={() => set(() => (mode = "search"))}
      >
        Search by text
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === "similar"}
        onclick={() => set(() => (mode = "similar"))}
      >
        Find similar
      </button>
    </div>

    <div class="row">
      {#if mode === "search"}
        <input
          class="query"
          type="search"
          placeholder="Describe a topic, e.g. attention and memory in skilled typing"
          aria-label="Search text"
          bind:value={query}
        />
        <button class="go" type="submit">Search</button>
      {:else}
        <label>
          Start from a
          <select
            value={source}
            onchange={(event) => {
              source = event.target.value;
              sourceRow = null;
              result = null;
              coords = [];
            }}
          >
            {#each bundle.spaceNames as name (name)}
              <option value={name}>{singular(name)}</option>
            {/each}
          </select>
        </label>
        {#key source}
          <Picker
            items={pickerItems}
            value={sourceRow}
            placeholder="Type part of the {singular(source)}'s name"
            onpick={(row) => set(() => (sourceRow = row))}
          />
        {/key}
      {/if}
    </div>

    <div class="row options">
      <label>
        Show the closest
        <select value={target} onchange={(event) => set(() => (target = event.target.value))}>
          {#each bundle.spaceNames as name (name)}
            <option value={name}>{plural(name)}</option>
          {/each}
        </select>
      </label>
      {#if mode === "search"}
        <label>
          matching
          <select value={style} onchange={(event) => set(() => (style = event.target.value))}>
            {#each STYLES as [value, text] (value)}
              <option {value}>{text}</option>
            {/each}
          </select>
        </label>
      {/if}
      <label>
        Results
        <select value={k} onchange={(event) => set(() => (k = Number(event.target.value)))}>
          {#each [10, 25, 50, 100] as n (n)}
            <option value={n}>{n}</option>
          {/each}
        </select>
      </label>
      {#if yearFilter}
        <label>
          Years
          <input
            class="year"
            type="number"
            placeholder="from"
            aria-label="From year"
            bind:value={yearFrom}
            onchange={run}
          />
          <span aria-hidden="true">–</span>
          <input
            class="year"
            type="number"
            placeholder="to"
            aria-label="To year"
            bind:value={yearTo}
            onchange={run}
          />
        </label>
      {/if}
    </div>
  </form>
{/if}

{#if status}
  <p class="status" role="status"><span class="spinner"></span>{status}…</p>
{/if}
{#if error}
  <p class="error" role="alert">{error}</p>
{/if}

{#if bundle && result}
  <main class:busy>
    <section class="results" aria-label="Results">
      <h2>
        {#if result.from}
          {capital(plural(result.target))} closest to
          <em>{labelOf(result.from.space, result.from.row)}</em>
        {:else if result.terms.length}
          {capital(plural(result.target))} closest to <em>{result.terms.join(" ")}</em>
        {:else}
          No results
        {/if}
      </h2>
      {#if result.unknown.length}
        <p class="note">
          Not in this library's vocabulary, so ignored: {result.unknown.join(", ")}
        </p>
      {/if}
      {#if !result.from && !result.terms.length}
        <p class="note">None of those words occur in this library. Try different wording.</p>
      {:else if !result.hits.length}
        <p class="note">Nothing matches these settings.</p>
      {/if}
      <ol>
        {#each result.hits as hit, i (hit.row)}
          {@const isSelected = selected?.space === result.target && selected.row === hit.row}
          <li>
            <button
              type="button"
              class="hit"
              class:selected={isSelected}
              onclick={() => select(result.target, hit.row)}
            >
              <span class="rank">{i + 1}</span>
              <span class="dot" style:background="var(--c{(clusters[i] ?? 0) % 6})"></span>
              <span class="what">
                {#if result.target === "document"}
                  <span class="title">{docs.title[hit.row]}</span>
                  <span class="sub">{byline(hit.row)}</span>
                {:else if result.target === "chunk"}
                  <span class="title">{docs.title[bundle.chunks.doc[hit.row]]}</span>
                  <span class="sub">{pages(hit.row)}</span>
                  {#if snippets[hit.row]}
                    <span class="snippet">{snippet(snippets[hit.row])}</span>
                  {/if}
                {:else}
                  <span class="title plain">{bundle.labels(result.target)[hit.row]}</span>
                  {#if documentCount(result.target, hit.row) != null}
                    {@const n = documentCount(result.target, hit.row)}
                    <span class="sub">{n} document{n === 1 ? "" : "s"}</span>
                  {/if}
                {/if}
              </span>
              <span class="score" title="Cosine similarity">{hit.score.toFixed(3)}</span>
            </button>
          </li>
        {/each}
      </ol>
    </section>

    <aside bind:this={side}>
      {#if points.length > 1}
        <section class="map" aria-label="Map">
          <div class="maphead">
            <h3>Map of these results</h3>
            <label>
              Groups
              <select value={nClusters} onchange={(event) => (nClusters = Number(event.target.value))}>
                {#each [1, 2, 3, 4, 5, 6] as n (n)}
                  <option value={n}>{n}</option>
                {/each}
              </select>
            </label>
          </div>
          <Map
            {points}
            selected={selected?.space === result.target ? selected.row : null}
            showLabels={result.target !== "document" &&
              result.target !== "chunk" &&
              points.length <= 30}
            onselect={(row) => select(result.target, row)}
          />
          <p class="muted small">
            Items with similar meaning sit close together. Colours are groups found among
            these results only.
          </p>
        </section>
      {/if}

      <section class="detail" aria-label="Details" aria-live="polite">
        {#if !selected}
          <p class="muted">Select a result to see it here and to explore from it.</p>
        {:else}
          {@const { space, row } = selected}
          <p class="kind">{capital(singular(space))}</p>
          {#if space === "document"}
            <h3>{docs.title[row]}</h3>
            {#each Object.keys(docs.factors) as factor (factor)}
              {#if levels(factor, row).length}
                <p class="chips">
                  <span class="muted">{capital(singular(factor))}</span>
                  {#each levels(factor, row) as level (level.row)}
                    <button type="button" class="chip" onclick={() => select(factor, level.row)}>
                      {level.label}
                    </button>
                  {/each}
                </p>
              {/if}
            {/each}
            {#if record}
              {#each Object.entries(record.fields) as [name, text] (name)}
                {#if name !== "title"}
                  <h4>{capital(name)}</h4>
                  <p class="text">{text}</p>
                {/if}
              {/each}
              {#if record.chunks}
                <p class="muted">
                  Full text in {record.chunks[1]} passages. Show the closest passages to search
                  inside it.
                </p>
              {/if}
              {#if Object.keys(record.meta).length}
                <dl>
                  {#each Object.entries(record.meta) as [name, value] (name)}
                    {#if value != null && value !== ""}
                      <dt>{name}</dt>
                      <dd>
                        {#if isLink(value)}
                          <a href={value} target="_blank" rel="noopener noreferrer">{value}</a>
                        {:else}
                          {show(value)}
                        {/if}
                      </dd>
                    {/if}
                  {/each}
                </dl>
              {/if}
            {/if}
          {:else if space === "chunk"}
            <h3>
              <button
                type="button"
                class="linklike"
                onclick={() => select("document", bundle.chunks.doc[row])}
              >
                {docs.title[bundle.chunks.doc[row]]}
              </button>
            </h3>
            <p class="muted">{pages(row)}</p>
            {#if record}<p class="text">{record.text}</p>{/if}
          {:else}
            <h3 class="plain">{bundle.labels(space)[row]}</h3>
            {#if docs.factors[space]}
              {@const own = documentsOf(space, row)}
              <h4>{documentCount(space, row)} document{own.length === 1 ? "" : "s"}</h4>
              <ul class="own">
                {#each own as d (d)}
                  <li>
                    <button type="button" class="linklike" onclick={() => select("document", d)}>
                      {docs.title[d]}
                    </button>
                    {#if docs.year[d]}<span class="muted"> ({docs.year[d]})</span>{/if}
                  </li>
                {/each}
              </ul>
            {/if}
          {/if}
          <div class="actions">
            <span class="muted">Show the closest</span>
            {#each bundle.spaceNames as name (name)}
              <button type="button" onclick={() => findSimilar(space, row, name)}>
                {plural(name)}
              </button>
            {/each}
          </div>
        {/if}
      </section>
    </aside>
  </main>
{:else if bundle && !status}
  <section class="intro">
    <h2>Two ways in</h2>
    <p>
      <strong>Search by text</strong> finds the {bundle.spaceNames.map(plural).join(", ")} whose
      meaning is closest to what you type. The words you use do not have to appear in a result.
    </p>
    <p>
      <strong>Find similar</strong> starts from something already in the library, for example a
      document or an author, and shows what lies closest to it. Choose a different kind of
      result to cross over, for example from a document to the authors nearest to it.
    </p>
  </section>
{/if}

<style>
  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
    padding: 1.4rem clamp(1rem, 4vw, 2.5rem) 0.6rem;
  }
  h1 {
    margin: 0;
    font: 600 1.7rem/1.2 var(--serif);
  }
  header p {
    margin: 0.3rem 0 0;
    max-width: 60ch;
  }
  .brand {
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    white-space: nowrap;
  }
  .muted {
    color: var(--muted);
  }
  .small {
    font-size: 0.85rem;
    margin: 0.4rem 0 0;
  }

  .controls {
    position: sticky;
    top: 0;
    z-index: 4;
    padding: 0.8rem clamp(1rem, 4vw, 2.5rem);
    background: var(--bg);
    border-bottom: 1px solid var(--line);
  }
  @media (max-width: 900px) {
    .controls {
      position: static;
    }
  }
  .modes {
    display: inline-flex;
    padding: 3px;
    border-radius: 8px;
    background: var(--surface-2);
    margin-bottom: 0.6rem;
  }
  .modes button {
    border: 0;
    border-radius: 6px;
    padding: 0.3rem 0.9rem;
    background: none;
    color: var(--muted);
  }
  .modes button[aria-selected="true"] {
    background: var(--surface);
    color: var(--ink);
    font-weight: 600;
    box-shadow: 0 1px 2px rgb(0 0 0 / 0.12);
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.6rem 0.9rem;
  }
  .options {
    margin-top: 0.6rem;
    font-size: 0.92rem;
    color: var(--muted);
  }
  .query {
    flex: 1 1 20rem;
    min-width: 0;
    padding: 0.55rem 0.8rem;
    font-size: 1.05rem;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--surface);
  }
  .go {
    padding: 0.55rem 1.2rem;
    border: 0;
    border-radius: 6px;
    background: var(--accent);
    color: var(--accent-ink);
    font-weight: 600;
  }
  select,
  .year {
    padding: 0.25rem 0.4rem;
    border: 1px solid var(--line);
    border-radius: 5px;
    background: var(--surface);
    color: var(--ink);
  }
  .year {
    width: 5.2rem;
  }

  .status,
  .error {
    margin: 1rem clamp(1rem, 4vw, 2.5rem);
  }
  .error {
    color: var(--warn);
  }
  .spinner {
    display: inline-block;
    width: 0.8em;
    height: 0.8em;
    margin-right: 0.5em;
    border: 2px solid var(--line);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .spinner {
      animation: none;
    }
  }

  main {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
    gap: 1.5rem;
    padding: 1rem clamp(1rem, 4vw, 2.5rem) 3rem;
    align-items: start;
  }
  main.busy {
    opacity: 0.55;
  }
  @media (max-width: 900px) {
    main {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  h2 {
    margin: 0 0 0.6rem;
    font: 600 1.05rem/1.35 var(--sans);
  }
  h2 em {
    font: italic 500 1.1rem var(--serif);
  }
  .note {
    margin: 0 0 0.6rem;
    color: var(--warn);
    font-size: 0.92rem;
  }
  ol {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .hit {
    display: grid;
    grid-template-columns: 1.8rem 0.6rem minmax(0, 1fr) auto;
    gap: 0.5rem;
    align-items: start;
    width: 100%;
    padding: 0.55rem 0.6rem;
    text-align: left;
    border: 0;
    border-bottom: 1px solid var(--line);
    background: none;
  }
  .hit:hover {
    background: var(--surface);
  }
  .hit.selected {
    background: var(--accent-soft);
  }
  .rank {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .dot {
    width: 0.55rem;
    height: 0.55rem;
    border-radius: 50%;
    margin-top: 0.5rem;
  }
  .what {
    display: grid;
    gap: 0.1rem;
    min-width: 0;
  }
  .title {
    font: 500 1.02rem/1.35 var(--serif);
    overflow-wrap: anywhere;
  }
  .plain {
    font-family: var(--sans);
  }
  .sub,
  .snippet {
    font-size: 0.88rem;
    color: var(--muted);
  }
  .snippet {
    color: var(--ink);
    opacity: 0.85;
  }
  .score {
    font-variant-numeric: tabular-nums;
    font-size: 0.88rem;
    line-height: 1.6;
    color: var(--muted);
  }

  aside {
    display: grid;
    gap: 1.2rem;
    position: sticky;
    top: 9.5rem;
    max-height: calc(100vh - 10.5rem);
    overflow-y: auto;
  }
  @media (max-width: 900px) {
    aside {
      position: static;
      max-height: none;
    }
  }
  .maphead {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
    color: var(--muted);
  }
  h3 {
    margin: 0;
    font: 600 1.2rem/1.3 var(--serif);
  }
  .maphead h3 {
    font: 600 0.95rem var(--sans);
    color: var(--ink);
  }
  h3.plain {
    font-family: var(--sans);
  }
  .detail {
    padding: 1rem 1.1rem;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
  }
  .detail > :first-child {
    margin-top: 0;
  }
  .kind {
    margin: 0 0 0.2rem;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
  }
  h4 {
    margin: 1rem 0 0.2rem;
    font: 600 0.8rem var(--sans);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .text {
    margin: 0;
    font: 1rem/1.6 var(--serif);
    white-space: pre-line;
    overflow-wrap: anywhere;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem 0.4rem;
    margin: 0.5rem 0 0;
    font-size: 0.9rem;
  }
  .chip {
    padding: 0.1rem 0.55rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--surface-2);
  }
  .chip:hover {
    border-color: var(--accent);
  }
  dl {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 0.15rem 0.8rem;
    margin: 1rem 0 0;
    font-size: 0.88rem;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
    overflow-wrap: anywhere;
  }
  a,
  .linklike {
    color: var(--accent);
  }
  .linklike {
    padding: 0;
    border: 0;
    background: none;
    font: inherit;
    text-align: left;
    text-decoration: underline;
    text-decoration-color: transparent;
  }
  .linklike:hover {
    text-decoration-color: currentColor;
  }
  .own {
    margin: 0;
    padding-left: 1.1rem;
    font: 0.95rem/1.45 var(--serif);
  }
  .own li {
    margin: 0.25rem 0;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
    margin-top: 1.1rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--line);
    font-size: 0.9rem;
  }
  .actions button {
    padding: 0.25rem 0.7rem;
    border: 1px solid var(--accent);
    border-radius: 6px;
    background: none;
    color: var(--accent);
  }
  .actions button:hover {
    background: var(--accent);
    color: var(--accent-ink);
  }
  .intro {
    max-width: 62ch;
    margin: 1.5rem clamp(1rem, 4vw, 2.5rem);
  }
  .intro p {
    color: var(--muted);
  }
  .intro strong {
    color: var(--ink);
  }
</style>

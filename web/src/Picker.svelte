<script>
  // A text box that finds one item of a space by typing part of its name.
  let { items, value = null, placeholder = "", onpick } = $props();

  let text = $state("");
  let open = $state(false);
  let active = $state(0);
  let lowered = $derived(items.map((item) => item.toLowerCase()));
  let matches = $derived.by(() => {
    const needle = text.trim().toLowerCase();
    if (!needle) return [];
    const starts = [];
    const contains = [];
    for (let i = 0; i < lowered.length && starts.length < 12; i++) {
      const at = lowered[i].indexOf(needle);
      if (at === 0) starts.push(i);
      else if (at > 0 && contains.length < 12) contains.push(i);
    }
    return [...starts, ...contains].slice(0, 12);
  });

  $effect(() => {
    text = value == null ? "" : items[value];
  });

  function pick(row) {
    open = false;
    text = items[row];
    onpick(row);
  }

  function onkeydown(event) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      open = true;
      active = Math.min(active + 1, matches.length - 1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      active = Math.max(active - 1, 0);
    } else if (event.key === "Enter" && open && matches.length) {
      event.preventDefault();
      pick(matches[active]);
    } else if (event.key === "Escape") {
      open = false;
    }
  }
</script>

<div class="picker">
  <input
    type="text"
    role="combobox"
    aria-expanded={open && matches.length > 0}
    aria-controls="picker-list"
    aria-autocomplete="list"
    autocomplete="off"
    spellcheck="false"
    {placeholder}
    bind:value={text}
    oninput={() => {
      open = true;
      active = 0;
    }}
    onfocus={(event) => event.target.select()}
    onblur={() => setTimeout(() => (open = false), 150)}
    {onkeydown}
  />
  {#if open && matches.length}
    <ul id="picker-list" role="listbox">
      {#each matches as row, i (row)}
        <li role="option" aria-selected={i === active}>
          <button type="button" class:active={i === active} onmousedown={() => pick(row)}>
            {items[row]}
          </button>
        </li>
      {/each}
    </ul>
  {:else if open && text.trim()}
    <p class="none">Nothing matches.</p>
  {/if}
</div>

<style>
  .picker {
    position: relative;
    flex: 1 1 18rem;
    min-width: 0;
  }
  input {
    width: 100%;
    padding: 0.5rem 0.7rem;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--surface);
  }
  ul,
  .none {
    position: absolute;
    z-index: 5;
    left: 0;
    right: 0;
    top: calc(100% + 4px);
    margin: 0;
    padding: 4px;
    list-style: none;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 6px;
    box-shadow: 0 8px 24px rgb(0 0 0 / 0.14);
    max-height: 22rem;
    overflow-y: auto;
  }
  .none {
    padding: 0.5rem 0.7rem;
    color: var(--muted);
  }
  li button {
    display: block;
    width: 100%;
    text-align: left;
    padding: 0.35rem 0.5rem;
    border: 0;
    border-radius: 4px;
    background: none;
  }
  li button.active,
  li button:hover {
    background: var(--accent-soft);
  }
</style>

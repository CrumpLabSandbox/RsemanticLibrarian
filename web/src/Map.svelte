<script>
  // Scatter plot of the current results: similar items sit close together.
  let { points, selected = null, showLabels = false, onselect } = $props();

  const W = 560;
  const H = 380;
  const PAD = 28;
  let hover = $state(null);

  let placed = $derived.by(() => {
    if (!points.length) return [];
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const sx = (W - 2 * PAD) / (x1 - x0 || 1);
    const sy = (H - 2 * PAD) / (y1 - y0 || 1);
    return points.map((p) => ({
      ...p,
      px: x1 === x0 ? W / 2 : PAD + (p.x - x0) * sx,
      py: y1 === y0 ? H / 2 : H - PAD - (p.y - y0) * sy,
    }));
  });
  let tip = $derived(placed.find((p) => p.row === hover) ?? null);
  const short = (text) => (text.length > 28 ? `${text.slice(0, 27)}…` : text);
</script>

<svg viewBox="0 0 {W} {H}" role="group" aria-label="Map of the results">
  <rect x="0.5" y="0.5" width={W - 1} height={H - 1} rx="8" class="frame" />
  {#each placed as p (p.row)}
    <g
      class="point"
      class:selected={p.row === selected}
      role="button"
      tabindex="0"
      aria-label="{p.rank}. {p.label}"
      onclick={() => onselect(p.row)}
      onkeydown={(event) => event.key === "Enter" && onselect(p.row)}
      onmouseenter={() => (hover = p.row)}
      onmouseleave={() => (hover = null)}
      onfocus={() => (hover = p.row)}
      onblur={() => (hover = null)}
    >
      <circle cx={p.px} cy={p.py} r="14" class="hit" />
      <circle cx={p.px} cy={p.py} r={p.row === selected ? 7 : 5} fill="var(--c{p.cluster % 6})" />
      {#if showLabels}
        {@const flip = p.px > W * 0.7}
        <text
          x={p.px + (flip ? -8 : 8)}
          y={p.py + 4}
          text-anchor={flip ? "end" : "start"}
          class="name"
        >
          {short(p.label)}
        </text>
      {/if}
    </g>
  {/each}
  {#if tip}
    {@const left = tip.px > W / 2}
    <text
      x={tip.px + (left ? -10 : 10)}
      y={tip.py - 10}
      text-anchor={left ? "end" : "start"}
      class="tip"
    >
      {tip.rank}. {short(tip.label)}
    </text>
  {/if}
</svg>

<style>
  svg {
    display: block;
    width: 100%;
    height: auto;
  }
  .frame {
    fill: var(--surface);
    stroke: var(--line);
  }
  .point {
    cursor: pointer;
    outline: none;
  }
  .hit {
    fill: transparent;
  }
  .point circle:not(.hit) {
    stroke: var(--surface);
    stroke-width: 1.5;
    opacity: 0.9;
  }
  .point.selected circle:not(.hit),
  .point:focus-visible circle:not(.hit) {
    stroke: var(--ink);
    stroke-width: 2;
    opacity: 1;
  }
  .name {
    font-size: 11px;
    fill: var(--muted);
    pointer-events: none;
  }
  .tip {
    font-size: 12px;
    font-weight: 600;
    fill: var(--ink);
    paint-order: stroke;
    stroke: var(--surface);
    stroke-width: 4px;
    pointer-events: none;
  }
</style>

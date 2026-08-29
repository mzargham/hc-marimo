---
name: marimo-talk-notebook
description: >-
  Use when authoring or editing talk.py (or any marimo notebook meant to be
  PRESENTED as slides) in the hc-marimo repo — the SciPy "Derivations, Not Just
  Simulations" talk. Presenting FROM a reactive notebook adds three failure
  modes on top of ordinary marimo authoring: the slides layout, the pre-run-vs-
  live-slider discipline, and self-contained WASM export. Invoke this BEFORE
  writing or changing any @app.cell in a presentation notebook, and whenever you
  touch the slides layout, decide whether a cell is static or slider-driven, or
  validate that the deck runs end-to-end. This complements the general
  authoring-marimo-notebooks skill; read both.
---

# Authoring a marimo notebook you will PRESENT as slides

`talk.py` is simultaneously the slide deck and the live demo — "eat your own
dogfood." That dual role is exactly where things break. This skill exists so you
build the deck with the presentation constraints already in mind, instead of
discovering them on stage.

Read the general `authoring-marimo-notebooks` skill first for the reactive-vs-
Jupyter fundamentals (strict scoping, `mo` injection, run-to-validate). This
skill adds only what's specific to *presenting*.

## The mental model that prevents 90% of the pain

A marimo notebook is **not** a Jupyter notebook and **not** a plain script:

- Cells run in **dependency order**, not file order — but **slides display in
  FILE order**. So the file order must match the narrative order you want to
  present, *and* the dependency graph must still be satisfiable. Keep them
  aligned: write cells top-to-bottom in beat order, and make sure each cell only
  depends on values defined in earlier cells.
- **One cell = one slide.** Resist multi-idea cells. If a slide has a heading, a
  figure, and a slider, that is fine — but it is *one* beat. Two beats = two cells.
- Reactivity is the "live" magic: a `mo.ui.slider` change re-runs downstream
  cells automatically. On stage that is your only interaction. It must be **fast**
  and **can't raise** — see the pre-run discipline below.

## 1. Slides layout (how `marimo run` becomes a deck)

marimo stores the slide arrangement in a JSON layout file and references it from
the `App(...)` constructor. Two ways to set it up:

**Preferred — via the editor UI** (records positions correctly):
1. `uv run marimo edit talk.py`
2. Switch to app view, open the layout dropdown, choose **Slides**.
3. marimo writes `layouts/talk.slides.json` AND rewrites the top of `talk.py` to
   `app = marimo.App(layout_file="layouts/talk.slides.json")`. Commit both.

**Scripted fallback** (when you can't drive the UI): the slides layout carries no
positional data — slides are just the cells in file order. So you can hand-create
the file and add the kwarg yourself:
- `layouts/talk.slides.json` → exactly `{"type": "slides", "data": {}}`
- Change the App line to `app = marimo.App(layout_file="layouts/talk.slides.json")`.

Present with `uv run marimo run talk.py` — it respects the saved layout. Arrow
keys advance slides. Because slides = file order, **reordering a beat = moving its
`@app.cell` block in the file** (keeping dependencies earlier than their uses).

**One cell = one slide, strictly.** marimo cannot group cells onto a slide. A cell
with no visual output still becomes a (blank) slide. So: **fold setup / import /
heavy-compute / slider-definition code INTO content cells** — upstream code into
the title cell, the numeric pipeline and slider definitions into the closing cell,
with the cell's markdown as its final expression. Execution is dependency-order,
not file-order, so earlier beats still receive those names, and the deck contains
no blank slides at all. Never interleave an output-less cell between two content
beats — that inserts a blank slide mid-deck.

### Live slider + reactive figure on ONE slide (the key idiom)

marimo's core rule: **interacting with a UI element re-runs every cell that reads
its `.value` EXCEPT the cell that defines it.** So a cell that both defines a slider
and reads `slider.value` will NOT update on interaction — fatal for "scrub and watch"
if you do it naively. The fix that keeps the control and its figure on the *same*
reveal.js slide:

- **Define the slider away from the cell that reads it** — in this deck, inside
  the closing (reveal) cell, whose output is its own markdown:
  ```python
  # inside the closing content cell, before its final mo.md(...) expression
  traj_slider = mo.ui.slider(0, 40, value=40, label="trajectory")
  t_slider = mo.ui.slider(0.0, 12.0, 0.1, value=12.0, label="time")
  ```
- **In the beat cell, display AND consume it** (the beat cell references `.value`
  but does NOT define the slider, so it re-runs on interaction):
  ```python
  @app.cell
  def _beat_simulate(mo, plt, physical_trajs, traj_slider, t_slider):
      fig = draw_chase(physical_trajs[traj_slider.value], t_slider.value)
      mo.vstack([traj_slider, t_slider, fig])   # widget rendered HERE, on this slide
      return
  ```
Interacting with the slider (rendered inside the beat slide) re-runs the beat cell →
the figure updates live, in place. The definition cell's blank slide lives harmlessly
at the end of the deck. This is how you honor "notebook-first slides" AND "scrub
sliders live" simultaneously — they only conflict if you define-and-read in one cell.

## 2. Load-time compute vs. slider-live discipline (the on-stage safety rule)

The safety boundary is **not** "avoid heavy compute" — it is *where* the heavy
compute lives. In marimo every cell runs when the app loads, so the plan is:
**start `uv run marimo run talk.py` before going on stage and let it finish
computing.** Heavy, dense, high-quality work (dense value-function grids, many
backward `solve_ivp` characteristics) belongs in **load-time cells** and is
welcome — the dense grids are what make the figures read well; do NOT downscale
them for speed. "Live" on stage then means *scrubbing sliders and revealing
outputs*, never *computing something hard for the first time in front of people*.

**Keep OUT of any slider-downstream cell** (they stall or can throw mid-talk):
- `solve_ivp` integrations, `sp.trigsimp`, heavy `sp.simplify`, `sp.diff` on big
  expressions — any multi-second call.

**So:**
- **Do the expensive work once, at load, upstream of the sliders.** Build the dense
  grid / integrate the trajectory fan in a load-time cell that returns cached NumPy
  arrays. Downscaling is the wrong fix — a slow *load* is fine (it happens before
  the talk); a slow *slider* is not.
- **Slider-driven cells must be cheap and total.** A `mo.ui.slider` may only index
  into the pre-integrated arrays, evaluate a `lambdify`'d function on a small grid,
  or redraw a matplotlib figure. If moving a slider would trigger a `solve_ivp`,
  pre-integrate across the slider's whole range at load and index into the result.
- **Derivation results are shown, not recomputed live.** Paste the *settled* result
  as pre-rendered LaTeX: capture `sp.latex(expr)` once during authoring and embed
  the string in `mo.md(r"...")`, OR compute it in a load-time cell — just never
  downstream of a slider. The audience sees the clean identity either way. (Matches
  the "show the derivation for effect, don't expect mastery" decision.)
- **A slider-driven cell must not raise for any slider value.** Test every slider at
  its extremes before the talk — a traceback on stage is the worst outcome.

Rule of thumb: if a cell would take more than ~100 ms to re-run, it does not belong
*downstream of a slider*. Push it upstream into a load-time cell whose dense output
the slider merely selects from.

**WASM/hosted backup caveat:** the Pyodide export recomputes on page load, so dense
load-time grids make the hosted deck slow to start. That is acceptable — the hosted
build is only the backup; the primary surface is local `marimo run`, started ahead
of time, where dense grids cost nothing on stage.

## 3. Self-contained export (works locally AND as hosted WASM)

`talk.py` must run under `marimo run` locally **and** `marimo export html-wasm`
(Pyodide in the browser). That forbids anything Pyodide or a fresh export can't see:

- **No cross-repo / sibling-file imports.** Do not import from
  `homicidal_chauffeur.py`, the paper's `plots.py`, or any local module. *Lift the
  specific figure code and hardcoded data you need into `talk.py`'s own cells.*
- **No `Path(__file__)`-relative file loading** — `__file__` differs between edit
  and export modes. Inline data as literals; embed images as data URIs if needed.
- **`mo` must come from a dedicated setup cell** and be passed as an arg to every
  other cell (it is not auto-injected during export):
  ```python
  @app.cell
  def _setup_marimo():
      import marimo as mo
      return (mo,)
  ```
- Only depend on packages in `pyproject.toml` (marimo, sympy, numpy, scipy,
  matplotlib) — all Pyodide-supported.

## 4. The reactive-scoping checklist (carried from the general skill)

These cause `MultipleDefinitionError` / `NameError` that look like "syntax is fine
but it won't run." Honor every one:

- Prefix every cell **function** name with `_` (`_slide_derive`) so it doesn't leak
  as an exported name.
- **Unique top-level names across the whole notebook.** Two cells both doing
  `fig, ax = plt.subplots()` at top level collide. Wrap each cell's body in an inner
  function, or `_`-prefix scratch/loop vars (`_fig`, `_ax`, `_i`).
- Pass every dependency in as an arg (`mo`, `np`, `plt`, `sp`, and any value from an
  earlier cell); return the names later cells need as a tuple.
- Import heavy libs once in an `_imports` cell and return them; don't re-import per
  cell (except `_`-aliased stdlib names scoped inside an inner function).

## 5. Validate by RUNNING, not by parsing

`marimo export script` only does AST analysis — it passes on code that would crash
at runtime. Always validate by executing every cell:

```bash
uv run marimo export html talk.py -o /tmp/talk.html   # exit 0 ⇒ every cell ran
grep -o "where does the ★ come from\|Simulate\|Lambdify" /tmp/talk.html | head
```

Then walk it live: `uv run marimo run talk.py`, step through all beats, move every
slider to both extremes, confirm no figure errors and the intended trajectory lands.

Keep the existing suite green — `uv run pytest` — and never edit
`homicidal_chauffeur.py`'s behavior while building the talk; lift from it, don't
mutate it.

## Quick reference

```bash
uv run marimo edit talk.py                              # author (set Slides layout here)
uv run marimo run talk.py                               # PRESENT (respects slides layout)
uv run marimo export html talk.py -o /tmp/talk.html     # full-execution validation
uv run marimo export html-wasm talk.py -o _site/index.html --mode run   # hosted deck
```

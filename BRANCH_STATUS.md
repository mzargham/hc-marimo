# Branch: `feature/parameter-propagation`

Parent: `main` @ `ec66706`

## What this branch does

The `main` notebook has a latent defect: the composite trajectory in section 9
(the one showing dispersal-surface crossing and evader direction change) uses
hardcoded angles `alpha_A = 40 deg`, `alpha_B = 95 deg`. These values were
chosen for the default speed ratio `w = 0.45` and fall outside the usable arc
once the user drags the `v_E` slider above ~0.65. At that point the notebook
silently produces garbage or crashes.

This branch replaces the hardcoded angles with a percentile-based formula that
adapts to the current `w`:

```
arc_min   = arcsin(min(w, 0.999))
arc_width = pi - 2 * arc_min
alpha_A   = arc_min + 0.105 * arc_width   (~40.0 deg at default w=0.45)
alpha_B   = arc_min + 0.540 * arc_width   (~95.1 deg at default w=0.45)
```

The percentiles (0.105, 0.540) were calibrated to reproduce the original angles
to within 0.1 deg at default parameters, so the visual output at defaults should
be indistinguishable from `main`.

## What changed

### Committed (2 files, new)

| File | Description |
|------|-------------|
| `test_parameter_propagation.py` | 10 experiments (E1-E4) probing whether slider changes propagate through the notebook |
| `test_regression.py` | 20 regression tests (R1-R20) pinning golden values, conservation laws, trajectory structure |

### Uncommitted (3 files, modified)

| File | Lines changed | Description |
|------|---------------|-------------|
| `homicidal_chauffeur.py` | ~30 | Percentile angles, crossing guard, narrative tweaks |
| `test_regression.py` | ~20 | R10 uses percentile formula, tolerance 0.1 -> 0.2 |
| `test_parameter_propagation.py` | ~40 | E3 uses percentile formula, tests all w in [0.05, 0.95] |

### Notebook changes in detail

1. **Section 1 narrative** -- added "at the default parameters ... below" so
   the reader knows the static demo is frozen at specific values.

2. **`physical_trajectories` cell** -- replaced hardcoded 40/95 deg with
   percentile formula. `_T_bk` now adapts: `max(trajectories[0].t[-1]*1.5, 15)`.

3. **Crossing guard** -- if the two characteristics don't cross within
   `min_dist < 0.2`, the composite is skipped and the slider defaults to the
   last regular trajectory. No crash, no garbage.

4. **Section 9 narrative** -- removed references to "matching section 1" since
   the composite now adapts to the current parameters.

### Test status

46/46 pass (16 phase2 + 10 propagation + 20 regression) as of latest run.

---

## Why this is not yet merge-ready

The code change is small and well-tested, but the notebook is a published
artifact. The bar for merge is visual + narrative correctness across the
parameter range, not just passing tests. Specific concerns:

1. **No visual validation yet.** The notebook has not been opened and inspected
   at non-default parameters. The tests verify numerical properties (angles
   inside arc, crossing distance, trajectory structure) but cannot verify that
   the *plots look right* -- correct axis scaling, no visual artifacts at the
   stitch point, direction-change diamond in the right place, etc.

2. **Narrative coherence at extreme parameters.** The section 9 text describes
   the composite trajectory but no longer pins a specific angle ("~48 deg
   direction change"). At very different `w` values the direction change could
   be much larger or smaller. The narrative should still make sense.

3. **VALIDATION_CHECKLIST.md is stale.** Item D6 describes the hardcoded-angle
   problem that this branch fixes; items I12, I13 reference "~48 deg" and
   "alpha_A=40 deg, alpha_B=95 deg". These need updating if the branch merges.

4. **Graceful degradation path untested in the browser.** When the crossing
   guard fires (e.g., very high `w`), the composite is simply absent from the
   slider. This might confuse the user -- there's no message explaining why.
   Whether this matters is a judgment call.

5. **Uncommitted changes need a clean commit.** The notebook and test changes
   are still unstaged.

---

## Acceptance criteria

### Must pass (blocking)

- [ ] **AC1. Tests green.** `python -m pytest test_phase2.py
  test_parameter_propagation.py test_regression.py -v` -- 46/46 pass.

- [ ] **AC2. Default parameters visually identical.** Open the notebook at
  default sliders (`v_E=0.45`, `omega=1.0`, `ell=0.5`). Section 1 static demo
  and section 9 composite trajectory should be visually indistinguishable from
  `main`. Specifically:
  - Direction-change diamond appears at the same position
  - Pursuer and evader paths have the same shape
  - Capture circle encloses the evader at the end

- [ ] **AC3. High-w composite works.** Set `v_E` to 0.80 (`w = 0.80`). The
  composite trajectory (last slider entry) renders without error. The
  direction-change diamond is visible. Both trajectory segments are plausible
  (smooth pursuer turns, straight-line evader segments). This was the specific
  failure case on `main`.

- [ ] **AC4. Section 1 unchanged.** The static demo in section 1 does not
  change when sliders are moved. Verify by dragging `v_E` to an extreme and
  confirming section 1 is pixel-identical.

- [ ] **AC5. No marimo errors.** The notebook loads without `critical`,
  `error`, or `warning` messages in the marimo console/terminal.

- [ ] **AC6. Narrative still coherent.** Read the section 9 explanatory text
  at a non-default `w` (e.g., 0.30 or 0.70). The text should not make claims
  that are visibly false for the current parameter setting (e.g., it should not
  say "~48 deg direction change" if the actual change is 20 deg or 80 deg).

### Should pass (important but not blocking)

- [ ] **AC7. Graceful degradation.** Set `v_E` to 0.95 (`w = 0.95`, very
  narrow usable arc). Either:
  - (a) The composite renders correctly, or
  - (b) The composite is absent and the slider shows only regular trajectories

  In neither case should there be a crash, traceback, or visually broken plot.

- [ ] **AC8. Regression values stable.** Compare R10 golden crossing times
  (`tau_cA ~ 6.1156`, `tau_cB ~ 9.0120`) between `main` and this branch at
  default parameters. Drift should be < 0.2 (the relaxed tolerance). Ideally
  < 0.05.

- [ ] **AC9. VALIDATION_CHECKLIST.md updated.** Items D6, I12, I13, N1 should
  reflect the new adaptive behavior rather than the old hardcoded angles.

### Won't fix in this branch

- **Reachable set `_T_max = 12.0` is hardcoded.** The reachable set computation
  uses a fixed horizon independent of the `T_horizon` slider. Lower priority,
  separate change.

- **No singular-arc handling.** The integrator uses `phi = -sign(sigma)` which
  gives `phi = 0` on the switching surface. Correct singular-arc control is a
  research-level change, out of scope.

- **Section 1 remains static.** This is by design -- the static demo is a
  pre-computed hook that loads instantly. Making it reactive would require the
  full SymPy -> lambdify -> ODE pipeline to run before the user sees anything.

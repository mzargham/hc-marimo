# Validation Checklist — Phases 1 & 2

Reviewer: ______________________  Date: ___________

## Status Key

- [ ] = not yet checked
- [x] = verified correct
- [!] = discrepancy found (describe in notes)

---

## A. Machine-Verified (Claude + SymPy, independently confirmed)

These were verified by an independent symbolic computation (not the notebook
code itself). No human action needed unless you want to double-check.

- [x] **A1.** Reduced dynamics algebra: differentiating the body-frame
  transformation, substituting absolute dynamics, applying trig identities,
  and normalizing yields f1 = -phi*x2 + w*sin(psi),
  f2 = phi*x1 + w*cos(psi) - 1.
- [x] **A2.** Hamiltonian H = p1*f1 + p2*f2 + 1 expands to
  phi*(p2*x1 - p1*x2) + w*(p1*sin(psi) + p2*cos(psi)) - p2 + 1.
- [x] **A3.** H is linear in phi (no phi^2 or higher terms).
- [x] **A4.** Switching function sigma = p2*x1 - p1*x2 (coefficient of phi).
- [x] **A5.** dH/dpsi = w*(p1*cos(psi) - p2*sin(psi)); setting to zero gives
  psi* = atan2(p1, p2).
- [x] **A6.** Second derivative d2H/dpsi2 at psi* = -w*||p|| < 0, confirming
  maximum (not minimum).
- [x] **A7.** Costate ODEs: p1_dot = -phi*p2, p2_dot = phi*p1.
- [x] **A8.** d/dt(||p||^2) = 2*(p1*p1_dot + p2*p2_dot) = 0 identically.
- [x] **A9.** No phi-psi cross terms in H (Isaacs separability condition).
- [x] **A10.** Physical sanity: w=0 (stationary evader), phi=0 gives
  x1_dot=0, x2_dot=-1 (straight-line pursuit at unit speed).
- [x] **A11.** Physical sanity: w=0, phi=1, evader at (1,0) gives
  x1_dot=x2_dot=0 (evader sits at center of turning circle, fixed point).

---

## B. Requires Human Verification Against Source Material

These are the *premises* — things the notebook asserts but SymPy cannot
validate. Each needs a grad student to open the cited source and compare
sign-by-sign.

### B1. Absolute dynamics (§2)

- [ ] **B1a.** Confirm that the Dubins vehicle model
  (x_P_dot = v_P cos theta, y_P_dot = v_P sin theta, theta_dot = phi)
  matches Isaacs (1965) Chapter 10 or Merz (1971) Chapter 2.
  *Specific concern:* Is theta measured CCW from the x-axis? Some sources
  use CW or measure from the y-axis.

- [ ] **B1b.** Confirm the evader kinematics
  (x_E_dot = v_E cos psi_lab, y_E_dot = v_E sin psi_lab) use the same
  angle convention as the pursuer.

### B2. Body-frame transformation (§3)

- [ ] **B2a.** Confirm the rotation convention:
  x1 = -(x_E - x_P)sin(theta) + (y_E - y_P)cos(theta)  (perpendicular)
  x2 =  (x_E - x_P)cos(theta) + (y_E - y_P)sin(theta)  (along heading)
  Compare against Isaacs (1965) pp. 297-300 or Merz (1971) eq. (2.X).
  *Specific concern:* Which axis is "along velocity" — x1 or x2? The
  notebook uses x2. Isaacs and some later papers use x1. A transposition
  here would swap sin and cos throughout all subsequent results.

- [ ] **B2b.** Confirm sign of x1. The notebook defines x1 as positive
  to the *left* of P's heading. Check this matches the source convention.
  If the source uses "positive right," every sigma sign flips.

### B3. Control bounds and normalization (§3)

- [ ] **B3a.** The notebook normalizes v_P = 1 and R_min = 1 (equivalently
  omega_max = 1), so phi in [-1, 1]. Confirm this matches Isaacs'
  normalization. Some sources normalize differently (e.g., v_P = 1 but
  R_min != 1).

- [ ] **B3b.** Confirm the dimensionless capture radius is
  ell_tilde = ell / R_min = ell * omega_max / v_P.

### B4. Hamiltonian sign convention (§5)

- [ ] **B4a.** The notebook uses H = p dot f + 1 with P minimizing and
  E maximizing. Confirm this matches the source. Alternative conventions:
  some authors use H = p dot f - 1 (if E minimizes escape time), or
  define the value function with opposite sign.

- [ ] **B4b.** Confirm that phi* = -sign(sigma) (not +sign(sigma)).
  This depends on whether P minimizes or maximizes H, and whether the
  costate points in the gradient direction or the negative gradient
  direction. Cross-reference: Isaacs (1965) eq. (10.3.X), Merz (1971)
  eq. (3.X), or Patsko & Turova (2011) Section 3.

### B5. Optimal evader heading (§5)

- [ ] **B5a.** Confirm psi* = atan2(p1, p2) rather than atan2(p2, p1)
  or atan2(-p1, -p2). The argument order in atan2 depends on which axis
  is which. This is directly coupled to B2a (axis convention).

### B6. Transversality conditions (§6, stated but not derived)

- [ ] **B6a.** The notebook states: at the terminal surface
  x1^2 + x2^2 = ell^2, the costate satisfies p(T) = lambda * x(T)
  for lambda > 0. Verify this against Isaacs (1965) pp. 310-312 or
  standard optimal control transversality conditions for a target set
  defined by g(x) = x1^2 + x2^2 - ell^2 = 0.
  *Specific concern:* The sign of lambda. If V increases away from the
  target, p = nabla V should point outward, so lambda > 0. But confirm.

### B7. Narrative claims (§1, §4)

- [ ] **B7a.** "Isaacs at the RAND Corporation, 1951" — verify the date
  and institutional affiliation. The RAND paper is P-257.

- [ ] **B7b.** "Torpedo pursuing a maneuvering ship" as the original
  inspiration — verify against Isaacs (1965) preface or Chapter 10
  introduction.

- [ ] **B7c.** The HJI equation as stated:
  min_phi max_psi [nabla V dot f(x, phi, psi)] + 1 = 0.
  Verify the +1 and the min/max ordering match the source.

---

## C. Cross-Reference Checks (Phase 2 prep)

These cannot be fully validated until the numerical integrator is built,
but a reviewer can check the theoretical claims now.

- [ ] **C1.** The costate rotation structure dp/dt = phi * J * p (where J
  is the 90-degree rotation matrix) implies p is fixed in the lab frame.
  This is a known result — confirm in Merz (1971) or Patsko & Turova
  (2011).

- [ ] **C2.** The notebook claims all four types of singular surfaces
  (dispersal, universal, equivocal, focal) appear for certain (w, ell).
  Verify this is stated in Merz (1971) or Patsko & Turova (2011).
  If the claim is more nuanced (e.g., only for specific parameter
  ranges), note the precise conditions.

- [ ] **C3.** Identify a published figure (e.g., Patsko & Turova 2011
  Fig. X, or Merz 1971 Fig. Y) showing specific trajectories for
  specific (w, ell) values. Record the exact parameter values and
  expected trajectory shape here — this will be the ground truth for
  Phase 2 numerical validation.

  Source: ________________  Figure: ________  w = ______  ell = ______
  Expected behavior: ________________________________________________

---

## E. Phase 2 Machine-Verified (test_phase2.py)

These are verified by the Phase 2 test suite and the notebook's built-in
Hamiltonian check cell.

- [x] **E1.** Lambdified RHS matches hand-coded numpy at 50 random
  (x, p, w) points (test T1). Confirms sp.lambdify produced correct code.
- [x] **E2.** Hamiltonian H* = -|sigma| + w||p|| - p2 + 1 stays near 0
  along numerically integrated trajectories, max |H*| < 1e-6 (test T2).
- [x] **E3.** ||p||^2 conserved along trajectories, max drift < 1e-8
  (test T3). Independent of the symbolic proof in §6.
- [x] **E4.** Terminal capture condition: x1^2 + x2^2 = ell_tilde^2 at
  terminal time for all computed trajectories (test T4).
- [x] **E5.** Transversality: p(T)/||p(T)|| = x(T)/||x(T)|| at terminal
  surface, with lambda > 0 (test T5).
- [x] **E6.** Stationary evader (w=0): backward trajectory from (0, ell)
  reaches (0, d) in time d - ell (test T6). Cross-checks that the dynamics
  reduce to straight-line pursuit in the degenerate case.

---

## F. Phase 2 — Requires Human Verification

- [ ] **F1.** Usable part condition sin(alpha) > w. Verify against
  Isaacs (1965) pp. 310-312 or Merz (1971) Section 3.4. The notebook
  derives this from transversality + H* = 0; confirm the derivation
  chain is correct.

- [ ] **F2.** Direction of backward trajectories. Do the spirals in the
  trajectory plot rotate in the correct direction (CW vs CCW) compared
  to published figures? This is sensitive to the sign conventions in B2.

- [ ] **F3.** Reachable set shape. Compare the isochrone structure against
  Patsko & Turova (2011) figures for matching (w, ell_tilde) values.
  Specifically: set the sliders to a known published configuration and
  visually compare the level set shapes.
  Source: ________________  Figure: ________  w = ______  ell = ______
  Match: [ ] yes  [ ] no  Notes: ________________________________________

- [ ] **F4.** Barrier curve. For parameter values where the evader can
  escape, does the reachable set boundary remain open (not closing)?
  Compare against known results for the barrier/dispersal line.

---

## G. Notebook Structure — Machine-Verified (test_phase2.py)

These tests guard against marimo DAG regressions discovered during
notebook integration.

- [x] **G1.** No marimo cell variable redefinitions (test T8). Parses the
  notebook AST and verifies that no two `@app.cell` functions assign the
  same non-private variable at cell scope. Private variables (prefixed
  with `_`) are excluded — marimo treats these as cell-local.

- [x] **G2.** Canonical dynamics `f1`, `f2` contain only reduced symbols
  (test T9). Verifies the expressions use `{phi, psi, w, x_1, x_2}` and
  no `AppliedUndef` objects (e.g., `theta(t)`, `x_P(t)`). This catches the
  Function-object leakage bug where the body-frame derivation's intermediate
  terms survived into the lambdified ODE.

- [x] **G3.** Full symbolic derivation (body-frame differentiation,
  derivative substitution, trig simplification, normalization) produces
  expressions that agree with the canonical form `f1 = -phi*x2 + w*sin(psi)`,
  `f2 = phi*x1 + w*cos(psi) - 1` at 20 random numerical test points
  (test T10). This verifies the derivation chain end-to-end, including
  correct cancellation of `theta(t)` after the `psi_lab = psi + theta(t)`
  substitution.

---

## H. Symmetry & Robustness — Machine-Verified (test_phase2.py)

- [x] **H1.** Dynamics reflection symmetry (test T11). Under x1 → -x1,
  p1 → -p1 the RHS transforms as (f1, f2, g1, g2) → (-f1, f2, -g1, g2).
  Verified at 50 random state-costate-parameter points.

- [x] **H2.** Isochrone mirror symmetry (test T12). Backward trajectories
  from symmetric terminal angles (alpha, pi-alpha) produce paths that are
  mirror images about the x2-axis, to within 1e-8 tolerance.

- [x] **H3.** Small-w integration stability (test T13). For w = 0.01
  (near-stationary evader), backward integration over T = 10 maintains
  H* < 1e-6 and ||p||^2 drift < 1e-8. Confirms numerical stability in
  the degenerate regime.

- [x] **H4.** Usable part monotonicity (test T14). The arc width
  pi - 2*arcsin(w) monotonically decreases as w increases from 0 to 1,
  confirmed at 50 sample points.

---

## I. Visualization Verification (Phase 5)

### Static diagrams

- [ ] **I1.** Problem geometry (V1): Pursuer heading arrow points in the
  theta direction (CCW from x-axis). Minimum turning circle has correct
  radius relative to the pursuer. Capture radius circle is visibly
  smaller than turning circle. Evader velocity arrow points at angle psi.

### Reactive plots

- [ ] **I2.** Usable part (V3): Increasing w via slider visibly shrinks
  the red usable arc. At w = 0.05, the arc spans nearly the full upper
  semicircle. At w = 0.45, it shrinks to a small arc near alpha = pi/2.

- [ ] **I3.** Usable part (V3): The costate arrows (blue) point radially
  outward on the usable arc. Arrow length increases toward the endpoints
  of the usable arc (where lambda diverges).

- [ ] **I4.** Vector field (V6): The switching surface at x1 = 0 is visible
  as a discontinuity in the quiver plot (arrows change direction across
  this line). For x1 > 0 the field rotates clockwise (phi = -1); for
  x1 < 0 it rotates counterclockwise (phi = +1).

- [ ] **I5.** Vector field (V6): Changing w via slider shows the evader's
  influence: at w ~ 0, the field is dominated by the pursuer's rotation;
  at w ~ 0.5, the field visibly tilts (evader contribution along psi = 0).

### Animation

- [ ] **I6.** Time animation (V4): At tau = 0, all trajectory dots sit on
  the usable arc of the terminal circle. At tau = T_horizon, the full
  trajectories are visible and match the static trajectory_plot.

- [ ] **I7.** Time animation (V4): Scrubbing the slider smoothly reveals
  trajectories building outward. No visual glitches, jumps, or missing
  segments. Bang-bang switching kinks appear as trajectories cross x1 = 0.

### Conservation plots

- [ ] **I8.** Conservation (V5): The H*(t) plot shows all lines clustered
  within +/- 1e-6 of zero. No outliers or drift trends.

- [ ] **I9.** Conservation (V5): The ||p_0||^2 - ||p||^2 plot shows all
  lines clustered near zero. Any drift is consistent with numerical
  integration error on high-lambda trajectories (see narrative).

### Forward-time physical chase (§9)

- [ ] **I10.** Physical chase: At t=0 (start), pursuer and evader are far
  apart. At t=T (capture), the pursuer's capture circle encloses the evader.

- [ ] **I11.** Physical chase: Pursuer triangles point in a physically
  reasonable heading direction. Turning is smooth and respects the minimum
  turning radius (no instantaneous direction reversals in heading).

- [ ] **I12.** Physical chase: Single-characteristic trajectories show
  straight-line evader paths. The composite trajectory (last slider position)
  shows a visible direction change (~48deg) at the dispersal surface.

- [ ] **I13.** Physical chase: The composite trajectory (last slider
  position, default) visually matches the static demo in §1 — same
  general shape, same direction-change behavior. (Both use w=0.45,
  ℓ̃=0.5, alpha_A=40deg, alpha_B=95deg.)

### Static demo (§1)

- [ ] **I14.** Static demo: The evader path has a visible kink (diamond
  marker) where the heading changes by ~48deg. Both segments before and
  after the kink are approximately straight lines.

- [ ] **I15.** Static demo: The pursuer path shows smooth wide turns
  consistent with a minimum turning radius constraint. The capture circle
  encloses the evader at the final position.

- [ ] **I16.** Static demo: Annotations (Pursuer start, Evader start,
  Direction change, Capture) are legible and correctly positioned.

### Narrative coherence

- [ ] **N1.** The static demo (§1) says "We will return to this
  forward-time view with interactive controls in §9." Confirm that §9
  header exists and the last slider position reproduces the static case.

- [ ] **N2.** The §9 narrative explains *both* the straight-line result
  (constant ψ_lab along single characteristics) and the direction-change
  mechanism (dispersal surface crossing). Read both paragraphs and confirm
  the explanation is clear and non-contradictory.

- [ ] **N3.** The "why backward" section (§4) references "§9" for the
  forward-time lift. Confirm this section actually appears as §9.

- [ ] **N4.** Section numbering is sequential (§1–§12) with no gaps or
  duplicates. Spot-check: §9 = Forward-Time Physical Chase, §10 = Backward
  Reachable Set, §11 = Verification & Validation, §12 = Extensions.

---

## D. Known Limitations & Open Questions

1. **Test R1 is self-referential.** The test re-derives the dynamics using
   the same transformation as the notebook, then checks the Hamiltonian
   structure. It does NOT compare against a hardcoded "Isaacs says this"
   ground truth. A stronger test would assert f1 == -phi*x2 + w*sin(psi)
   directly after the full symbolic derivation. *Mitigated:* Test T10 now
   verifies the full derivation chain numerically, and test T9 verifies the
   canonical form contains only reduced symbols.

2. **Singular surface handling.** The current integrator uses -sign(sigma)
   for phi*, which produces phi*=0 when sigma=0 (the singular surface).
   This is not physically correct — singular arcs require a separate
   control law. For trajectories that cross or approach the switching
   surface, the integration may lose accuracy. §7 documents the four
   types of singular surfaces but does not implement explicit handling.

3. **No published figure reproduction yet.** F3 is the critical human
   check. Until a reviewer identifies a specific published figure and
   confirms the notebook reproduces it, the numerical results should be
   treated as plausible but not validated against ground truth.

4. **Body-frame derivation vs canonical form.** The SymPy derivation
   starting from `sp.Function` objects produces expressions containing
   `theta(t)`, `x_P(t)`, etc. — SymPy cannot symbolically recognize
   these as the reduced coordinates `x_1`, `x_2`. The notebook therefore
   constructs `f1`, `f2` in canonical form using the reduced symbols
   directly. Test T10 confirms the derivation and canonical form agree
   numerically. This is a SymPy limitation, not a mathematical error.

5. **Composite trajectory is approximate.** The composite trajectory
   (static demo + last dynamic slider position) stitches two backward
   characteristics at their crossing in (x1, x2) space. This is a
   first-order approximation of the true dispersal-surface crossing —
   exact matching would require iterating to find the precise point where
   both characteristics yield identical capture times (equal value
   function). The current approach finds the closest spatial crossing
   (dist ~0.008), which is sufficient for demonstration purposes but
   may show small geometric artifacts.

6. **Slider default vs static demo.** The v_E slider defaults to 0.45
   (w=0.45) to match the static demo in §1. If the user changes w,
   the composite trajectory recomputes with the new w but the hardcoded
   alpha_A=40deg, alpha_B=95deg may no longer produce a meaningful
   crossing. The composite is most informative near the default parameters.

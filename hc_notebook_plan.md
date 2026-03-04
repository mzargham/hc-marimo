# Homicidal Chauffeur Differential Game — Marimo Notebook Plan

## 1. Goals (Restated & Refined)

**High-level:** Demonstrate that Claude + SymPy symbolic computation is a
powerful combination for rigorous mathematical exploration — the kind of
thing that usually lives in textbooks but can now be made *alive* and
*interactive*.

**Mid-level:** Educate readers about differential game theory, using the
Homicidal Chauffeur as a worked example. The narrative arc should take a
reader from "what is a differential game?" to "here is the Hamiltonian, here
are the optimal controls, here is what the trajectories look like" — with
every step symbolically derived and numerically verified.

**Low-level:** Produce a single `homicidal_chauffeur.py` marimo notebook that:
- Runs locally via `marimo edit homicidal_chauffeur.py`
- Can be deployed as a read-only app via `marimo run homicidal_chauffeur.py`
- Is git-friendly (pure Python, no JSON)
- Uses interactive sliders for the two dimensionless parameters (w, ℓ)
- Contains both symbolic derivations (SymPy) and numerical simulations
  (SciPy + matplotlib)

## 2. Design Constraints

| Constraint | Rationale |
|---|---|
| Single-file `.py` notebook | Marimo native format; git-friendly; shareable |
| Dependencies: sympy, scipy, numpy, matplotlib, marimo | Scientific Python stack; no exotic deps |
| Each cell = one logical step | Marimo's DAG requires no variable mutation across cells |
| Symbolic before numerical | Derive first, simulate second — enforce correctness |
| All parameters dimensionless | Follow Isaacs' normalization: v_P = 1, R = 1 |
| TDD mindset | Symbolic results verified against known closed-form; numerical results verified against symbolic |

## 3. Notebook Sections (Cell Groups)

### §0 — Preamble & Imports
- `import marimo as mo`, sympy, numpy, scipy, matplotlib
- Brief title card with problem statement

### §1 — Historical Context & Problem Statement
- Markdown narrative: Isaacs (1951), RAND, the torpedo-vs-ship origin
- The two players: P (fast, constrained turning) vs E (slow, omnidirectional)
- What "capture" means (distance ≤ ℓ)
- Why it matters (missile defense, UAV guidance, autonomous vehicles)

### §2 — Kinematics in Absolute Coordinates
- SymPy: define symbols for positions (x_P, y_P, x_E, y_E), heading θ,
  speed v_P, turn rate φ, evader speed v_E, evader heading ψ
- State the 5-DOF absolute dynamics
- Explain why 5D is too many — motivate reduced coordinates

### §3 — Reduction to Relative Coordinates
- SymPy: derive the 2D reduced dynamics (Isaacs' body-fixed frame)
  - ẋ₁ = −φ x₂ + w sin ψ
  - ẋ₂ = φ x₁ + w cos ψ − 1
- Show the coordinate transformation explicitly
- Identify the two free parameters: w = v_E/v_P, ℓ = capture_radius/R_min
- **Interactive sliders** for w ∈ (0, 1) and ℓ > 0

### §4 — The Differential Game Formulation
- Markdown: define the zero-sum game structure
  - P minimizes capture time T
  - E maximizes capture time T
  - Terminal set: x₁² + x₂² ≤ ℓ²
- Isaacs' main equation and the concept of Value function V(x)
- Connection to Hamilton-Jacobi-Isaacs (HJI) PDE

### §5 — Hamiltonian & Isaacs' Condition
- SymPy: construct the Hamiltonian
  H(x, p, φ, ψ) = p₁ f₁(x, φ, ψ) + p₂ f₂(x, φ, ψ) + 1
  (the +1 comes from minimizing time)
- Derive the saddle-point condition:
  min over φ, max over ψ of H = max over ψ, min over φ of H
- SymPy: solve for optimal controls
  - φ*(x, p) = sign(p₁ x₁ + p₂ x₂)  (bang-bang in turn rate)
  - ψ*(x, p) = atan2(p₁, p₂)  (evader aligns velocity with costate gradient)
- Verify: substitute back, confirm saddle-point holds

### §6 — Costate (Adjoint) Equations
- SymPy: derive ṗ₁, ṗ₂ from ∂H/∂x
- Write the full 4D ODE system: (x₁, x₂, p₁, p₂)
- Discuss transversality conditions at the terminal surface

### §7 — Singular Surfaces (Conceptual)
- Markdown: explain dispersal lines, universal lines, equivocal lines,
  focal lines — the taxonomy from Merz (1971)
- Why the HC game is special: all four types appear for certain (w, ℓ)
- The "usable part" boundary condition on the terminal circle
- This section is primarily narrative/educational; full computation of
  singular surfaces is a research problem

### §8 — Numerical Trajectory Simulation
- Use scipy.integrate.solve_ivp on the 4D system
- **Interactive**: sliders for (w, ℓ), initial conditions (x₁₀, x₂₀)
- Plot trajectories in the (x₁, x₂) plane with terminal circle
- Show both P-optimal and E-optimal plays
- Overlay: show what happens when one player deviates from optimal

### §9 — Backward Reachable Set (Level Sets of V)
- Compute isochrones: for a given time T, what initial states lead to
  capture in exactly T?
- Use the numerical ODE integration backward from the terminal circle
- Plot nested level sets, reproducing the classic figures from
  Patsko & Turova
- **Interactive**: slider for time horizon, parameter values

### §10 — Verification & Validation
- Sanity checks:
  - When w → 0 (stationary evader), P captures from everywhere (Dubins path)
  - When w → 1 (equal speeds), capture only if ℓ large enough
  - Hamiltonian is constant along optimal trajectories (numerical check)
  - Symmetry w.r.t. x₂ axis when terminal set is centered
- SymPy symbolic checks cross-referenced against numerical integration

### §11 — Extensions & Further Reading
- Acoustic HC (Bernhard): evader speed depends on distance
- Multiplayer generalizations (Kumkov, Le Ménec, Patsko 2017)
- Reversed game (Lewin 1973): evader escapes detection zone
- Connection to Dubins car optimal control
- Pointers to: Isaacs (1965), Merz (1971), Patsko & Turova (2011),
  Coates & Pachter (2018)

## 4. Implementation Strategy

**Phase 1 — Symbolic Foundation (§0–§6)**
Build the SymPy derivations cell by cell. Each cell produces symbolic
objects that downstream cells consume (marimo's DAG handles this
automatically). Verify each derivation before proceeding.

**Phase 2 — Numerical Engine (§8–§9)**
Lambdify the symbolic expressions for the RHS of the ODE system.
Build the trajectory integrator and level-set computation. Wire up
marimo sliders for interactivity.

**Phase 3 — Narrative & Polish (§1, §4, §7, §10, §11)**
Write the educational prose. Add verification cells. Review flow.

**Phase 4 — Testing**
- Symbolic: assert that optimal controls satisfy saddle-point
- Numerical: assert Hamiltonian conservation along trajectories
- Boundary: assert capture condition at terminal time
- Regression: known parameter values reproduce known figures

## 5. Dependencies

```
marimo>=0.9.0
sympy>=1.13
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
```

## 6. Resolved Design Decisions

1. **Singular surface analysis (§7):** Keep conceptual and narrative only.
   Full computation (Merz's 20-subregion classification) is a research
   effort beyond notebook scope. Computational effort concentrates on
   §5–§6 (symbolic) and §8–§9 (numerical).

2. **Level-set computation method (§9):** Use the simpler characteristic
   shooting approach (backward integration from the terminal circle).
   Describe and cite the more rigorous grid-based HJI PDE method
   (Patsko & Turova 2001, Bardi, Falcone & Soravia 1999) and leave it
   as an exercise for the motivated reader.

3. **Interactivity — physical slider variables:**
   - `v_E` — evader max speed, constrained to `v_E ∈ (0, v_P/2]`
   - `ω_max` — pursuer max angular velocity (= v_P / R_min)
   - `ℓ` — capture radius
   - `(x₁₀, x₂₀)` — initial conditions (Phase 2)

   Sliders are on *physical* quantities; the notebook shows the explicit
   map to Isaacs' dimensionless parameters `w = v_E/v_P` and
   `ℓ̃ = ℓ / R_min = ℓ · ω_max / v_P`.

   **Domain constraint rationale:** `v_E ≤ v_P/2` keeps the problem in
   the regime where the interplay between speed advantage and turning
   constraint is the deciding factor — the "interesting" part of the
   parameter space. This constraint is made explicit in the notebook
   with a note that we've restricted the domain deliberately; the full
   space includes degenerate cases (e.g., `w → 1` where capture is
   trivially avoidable). Good for sensitivity analysis and for
   categorizing which type of limit cycle / singular structure the
   solution resolves to.

4. **Phase gating:** Build is phased with review gates:
   - **Phase 1** (§0–§6): Symbolic foundation — **COMPLETE, REVIEWED**
   - **Phase 2** (§8–§9): Numerical engine + interactive visualization — **COMPLETE, REVIEWED**
   - **Phase 3** (§7, §10, §11): Narrative, V&V, extensions — **COMPLETE**
   - **Phase 4**: Final testing and polish — **COMPLETE** (20 tests: 6 Phase 1 + 14 Phase 2)

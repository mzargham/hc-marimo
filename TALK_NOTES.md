# Talk notes — *Derivations, Not Just Simulations* (SciPy 2026)

**Slot:** ~20 min → aim ~17 min talk + Q&A. **Arc:** immersion → reveal (the six-stage
pattern is *felt*, unlabeled, until beat 10). **Surface:** present live from the marimo
notebook in slides layout.

## Run it

```bash
uv run marimo run talk.py          # PRESENT — start this BEFORE going on stage; let it
                                   # finish the dense backward-integration load (~few s)
```
Arrow keys advance slides. Every cell is a content slide (17 slides across the 10
beats); the deck ends at the reveal — no blank slides after it.

**Backup:** the deployed deck at mzargham.github.io/hc-marimo/slides/ open in a second
tab, plus the full companion notebook `homicidal_chauffeur.py` (hosted at
mzargham.github.io/hc-marimo/notebook/) linked on the reveal.

## What's live vs. pre-run

- Heavy dense compute (80 backward characteristics, value-function scatter, forward-lift)
  runs **once at load** at the canonical operating point `w=0.45, ℓ̃=0.5`. Never scrubbed.
- **Live sliders** (they update their figure in place — one reveal.js slide each):
  - Beat 4: `v_E`, `ω`, `ℓ` → the derived-config readout + schematic (decoupled from the
    sims, so scrubbing is instant; say "the whole game collapses to two numbers").
  - Beat 7: `trajectory` + `forward time` → the chase playback. Last index = the ★ chase.
- Everything else is pre-rendered markdown / static figures.

## Beat sheet (cumulative timing)

| # | Beat | ~min | Say this |
|---|------|------|----------|
| 1 | Title | 0:15 | "This is a marimo notebook. It's also these slides." (plant the seed) |
| 2 | The ★ puzzle | 2:00 | Tag: fast-clumsy car vs slow-agile runner. Two heuristics (11.3s, 6.2s) — but the *provably optimal* escape survives 12.1s with a sharp kink (★) no intuition predicts. Say aloud: heuristic panels face a pure-pursuit chaser; 12.1s is optimal-vs-optimal. **"Where does the ★ come from? Watch it get computed."** |
| 3 | Textbooks | 1:15 | The answer's in Isaacs/Kirk/Bryson & Ho — as static prose. Following ≠ computing. "Let's compute with the derivation instead." |
| 4a | Symbolize — physical | 1:15 | The physical (lab) picture: 5-DOF state `(x_P,y_P,θ,x_E,y_E)`. P fast but min turning radius; E slow but instantly agile. Scrub the physical sliders. |
| 4b | Reduce to relative frame | 1:15 | Pin the frame to the pursuer → only relative `(x₁,x₂)` matters. Three different lab configs are the *same* reduced problem (5 DOF → 2 DOF). Whole game depends on just `(w, ℓ̃)`. |
| 4c | Two players, two controls | 0:45 | Define the controls *before* they appear in the dynamics. **Asymmetry**: pursuer committed forward, only a bounded **turn rate `φ`** (`\|φ\|≤1`, no stop/sidestep); evader a **velocity in a disk** radius `w`, optimal on the rim so its knob is **heading `ψ`**. **Notation**: capital = the state-dependent **policy** (`Φ(x)` turn-rate, `Ψ(x)` heading); lowercase = the **action** it outputs (`φ=Φ(x)`, `ψ=Ψ(x)`). Narrate: evader could slow down, it just never optimally does. |
| 5a | Derive (manipulations) | 1:30 | `sp.diff` + `sp.trigsimp` do the work SymPy eliminates: differentiate the body-frame coords, substitute the lab dynamics, and θ cancels → Isaacs' canonical form. "Two pages of trig, two lines of code." |
| 5b | Objective (value fn) | 1:00 | **Build `V → V*` carefully** (this confused a test listener). Fix both **policies** `Φ,Ψ`; the value is a **bare integral of the running cost** to capture: `V(x\|Φ,Ψ) = ∫₀^{T(x,Φ,Ψ)} 1 dt` (running cost 1 = time). `V*(x) = min_Φ max_Ψ V` is the saddle **over policies** (star = "under optimal play by both"), `V*=0` on the capture circle. Optimum = **optimal policies `Φ*,Ψ*`**, which output **actions `φ*(x),ψ*(x)`**. **Don't conflate the action (instantaneous turn) with the policy that produces it.** Text-only. |
| 5c | Hamiltonian (the plan) | 0:45 | `H = dV*/dt + 1` from two rates: (1) chain rule `dV*/dt = ∇V*·ẋ = p·f` (costate `p=∇V*`, dynamics `f=ẋ`); (2) the `+1` is the running-cost integrand **back by the fundamental theorem** (`d/dt ∫₀ᵗ 1 = 1`) — "that's why the 1 pops back out." So `H = p·f + 1`. Optimal **actions** = *pointwise* saddle: `φ*=argmin_φ H`, `ψ*=argmax_ψ H` = the feedback `Φ*,Ψ*`. Text-only. |
| 5d | Costate & dynamics | 1:00 | Define `H`'s two ingredients: the **dynamics** `f = (ẋ₁, ẋ₂)` (recall from 5a) and the **costate** `p = ∇V*` (gradient of the value fn). Then `ṗ = -∂H/∂x` (rotation), `d/dt‖p‖²=0`. **Narrate**: `p` prices each direction; the ODE is one `sp.diff`, the conservation one `sp.simplify` (the cancellation Isaacs did by hand); `p` spins, never changes length; **both players read their move off `p`**. Figure: costate rotating on the fixed-‖p‖ circle. |
| 5e | Pursuer (bang-bang) | 1:00 | `H` is *linear* in `φ` so `.coeff(phi)` gives `σ`. Minimising a linear function on `[-1,1]` lands on the boundary opposite the slope, so `φ*=-sign(σ)`, forced. Hard left/right, flips **each time** `σ=0`, often several times per chase (**not** the ★, that is a repeated switch). |
| 5f | Evader (up the gradient) | 1:00 | Maximise `w(p₁sinψ+p₂cosψ)` so `ψ*=atan2(p₁,p₂)`: velocity along `p`. Since `p=∇V*`, **the evader runs straight up the value gradient**, the one intuition the whole talk trades on. Value = `w‖p‖`. **The ★ lives here**: at a dispersal surface `p` is multivalued, the evader commits to one escape family once, and that single kink is the ★ from slide 2. |
| 6 | Lambdify | 1:15 | `sp.lambdify` turns those exact expressions into NumPy — zero transcription, zero silent sign-bugs. Cross-check vs hand-coded RHS agrees to machine precision. |
| 7 | Simulate | 2:00 | `solve_ivp` on the lambdified dynamics. Scrub to the ★ chase. "Same numbers as slide 2 — now generated, not asserted." |
| 8 | Payoff 1: one step, one script | 1:15 | The real punchline of the title. Show `lemma_04_bangbang.py`: `AXIOMS_USED = [A3, A7, I2]`, `derive` returns `-sp.sign(sigma)`, `verify` asserts `sigma*phi* == -Abs(sigma)`. "Not prose that *claims* φ\*=-sign(σ). A script that derives it, checks it, and names what it assumed. The pursuer's forced hard-over turn." |
| 9 | Payoff 2: verify (machine) vs validate (human) | 1:45 | Draw the V&V line. **Verify** = "did we build it *right*?": `python derivation/run.py` checks L1..L8 as symbolic identities, green in CI. **Validate** = "did we build the *right thing*?": the same run prints the ledger, **derived vs assumed** (Posited: the game; Inherited: PMP/HJB, Isaacs, viscosity, cited not proven; Leaned-on: V\*∈C¹, which breaks at the ★). "Verification is automatic. Weighing that ledger and calling it appropriate is the human act." Don't make the human claim yet: set it up. |
| 10 | THE REVEAL | 2:00 | Name the six stages as earned insight; **end on Validate, the human step**. Say aloud that these assumptions were a contextually appropriate choice for this problem (that is the validation, and it is mine to make). Machine verification runs underneath all six. SymPy extends the notebook ecosystem to the *irreducibly symbolic* subjects. "And this whole talk *was* the notebook. Fork it." |
| — | **Invitation (oral)** | 0:45 | Over the reveal slide: invite collaboration on open-source educational content in advanced applied math. *Not built into the notebook.* |

## Beat 4a — live slider demo choreography (verified behavior)

The schematic reads three sliders. What each visibly does (tested at min/mid/max):

- **`v_E`** — lengthens/shortens the **red evader velocity arrow**, on the *same scale*
  as the blue `v_P` arrow. At `v_E→0.95` the two arrows are nearly equal length.
- **`ω_max`** — resizes the **blue dashed turning circle** (`R_min = v_P/ω`): low ω = big
  sweeping circle (sluggish), high ω = tight circle (agile). Stays in-frame across the range.
- **`ℓ`** — resizes the **gray dotted capture circle** around P.

Suggested sequence (drag one slider at a time; pause on each):

1. **Defaults** (`0.45, 1.0, 0.5`): "Blue is the car — its velocity, and the circle it
   *cannot* turn tighter than. Red is the runner."
2. **`ω_max` down → up**: "Make it sluggish — huge turning circle, committed to wide sweeps.
   Now tighten it — nearly a point-turn vehicle, and the game gets easy." (No clipping now.)
3. **`v_E` up toward 0.95**: "Speed up the runner — watch its arrow grow toward the car's.
   When the speeds match, the car's whole advantage is gone — capture becomes impossible."
4. **`ℓ` up**: "This is just how close counts as caught — bigger capture radius favors P."
5. **Punchline → advance to 4b**: "Three physical knobs — but the *game* only cares about two
   ratios, `w` and `ℓ̃`. Here's why they're the same problem…"

Minor cosmetic notes (not bugs): at `ω=3.0` the `R_min` label sits close to the tiny circle;
at `ℓ=1.5` the capture circle overlaps the turning circle (physically correct). `v_P` is fixed
at 1 (no slider) — it's the reference speed the other arrow is measured against.

## If running long (cut buffer, drop in this order)

1. Beat 5e: compress to just "linear in φ, so φ*=-sign(σ), forced bang-bang" + the conservation line.
2. Payoff 2: show the chain + audit but skip reading each L1..L8 row; land only the C¹-breaks-at-★ headline.
3. Beat 4: mention the `(w, ℓ̃)` collapse verbally without dwelling on the sliders.

## Q&A backup (not in the deck)

- Merz's singular-surface taxonomy (dispersal / universal / equivocal / focal).
- Extensions: multiplayer pursuit, acoustic HC, the Dubins-path connection.
- Why pure-pursuit (not optimal) pursuer in the beat-2 heuristics: the start sits on a
  dispersal singularity where σ is multi-valued; pure-pursuit avoids chatter and the
  saddle-point bound still holds. The full companion notebook has the optimal-feedback field.

## Pre-flight checklist

- [ ] `uv run marimo run talk.py` starts clean; walk all 10 slides.
- [ ] Move every slider (beats 4, 7) to both extremes: no figure errors.
- [ ] Beat 7 default lands on the ★ (composite) chase.
- [ ] `uv run python derivation/run.py` prints L1..L8 PASS + the assumption audit; exit 0.
- [ ] `uv run --group dev pytest -q` → 30 passed.
- [ ] Backup WASM tab open; laptop on a timer for a ≤17-min dry run.

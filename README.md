# The Homicidal Chauffeur: A Differential Game

**Symbolic Derivation & Interactive Simulation with SymPy**

[**Live demo**](https://mzargham.github.io/hc-marimo/)

---

An interactive [marimo](https://marimo.io) notebook exploring Rufus Isaacs' foundational pursuit-evasion problem from his 1951 RAND Corporation research. Every equation is derived symbolically with SymPy, then brought to life through numerical simulation with interactive parameter sliders.

The **Homicidal Chauffeur problem** asks: *Can a fast but clumsy car catch a slow but agile pedestrian?* A pursuer (high speed, constrained turning radius) chases an evader (low speed, unlimited maneuverability) on an unbounded plane — a canonical model for missile guidance, UAV pursuit, and autonomous vehicle collision avoidance.

## What's Inside

- **Problem history** — Isaacs, RAND, and the torpedo-vs-ship origin (1951)
- **Absolute & reduced kinematics** — 5-DOF dynamics reduced to a 2-DOF body-frame system
- **Differential game formulation** — zero-sum structure, value function, Hamilton-Jacobi-Isaacs PDE
- **Hamiltonian & optimal controls** — symbolic derivation of bang-bang pursuer and gradient-aligned evader strategies
- **Costate equations** — adjoint system, costate norm conservation proof
- **Singular surfaces** — Merz's taxonomy: dispersal, universal, equivocal, and focal lines
- **Numerical trajectory simulation** — interactive sliders for evader speed, turn rate, and capture radius
- **Backward reachable sets** — isochrones computed via method of characteristics
- **Verification & validation** — 20 automated tests (symbolic + numerical) cross-checking the derivations

## Run Locally

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mzargham/hc-marimo.git
cd hc-marimo
uv sync
uv run marimo edit homicidal_chauffeur.py
```

For read-only app mode (sliders work, code hidden):

```bash
uv run marimo run homicidal_chauffeur.py
```

## Dependencies

marimo, SymPy, NumPy, SciPy, matplotlib — see [pyproject.toml](pyproject.toml) for pinned versions.

## References

- R. Isaacs, *Games of Pursuit*, RAND Corporation P-257 (1951)
- R. Isaacs, *Differential Games*, John Wiley & Sons (1965), pp. 297–350
- A.W. Merz, *The Homicidal Chauffeur — a Differential Game*, PhD Thesis, Stanford (1971)
- V.S. Patsko & V.L. Turova, "Homicidal Chauffeur Game: History and Modern Studies," *Advances in Dynamic Games*, Annals of the ISDG Vol. 11 (2011)
- S. Coates & M. Pachter, "The Classical Homicidal Chauffeur Game," *Dynamic Games and Applications* 9(1), 2019

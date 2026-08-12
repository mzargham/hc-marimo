"""The axiom manifest: the single source of truth for what the derivation assumes.

Three tags:
  POSITED     - a modeling choice we simply declare (the structure of the game).
  INHERITED   - an external theorem we USE and CITE but do not prove here
                (PMP / HJB / verification). These are the machinery, not our result.
  REGULARITY  - a smoothness assumption the smooth derivation rests on. R* (V* in C^1)
                is exactly what FAILS on singular surfaces; the dispersal surface (the
                star in the talk) is where it breaks and the costate becomes multivalued.
"""

from dataclasses import dataclass

import sympy as sp

from derivation import symbols as S

POSITED = "POSITED"
INHERITED = "INHERITED"
REGULARITY = "REGULARITY"


@dataclass(frozen=True)
class Axiom:
    name: str
    tag: str
    statement: str          # human-readable claim
    relation: object = None  # optional SymPy relation making it machine-checkable
    citation: str = ""       # required for INHERITED

    def __str__(self):
        cite = f"  [{self.citation}]" if self.citation else ""
        return f"{self.name} ({self.tag}): {self.statement}{cite}"


# Convenience symbols for stating relations.
_ell = sp.symbols("ell_tilde", positive=True)          # capture radius
_x1, _x2, _p1, _p2 = S.x1, S.x2, S.p1, S.p2
_phi, _w = S.phi, S.w

AXIOMS = {
    # ---------------------------------------------------------------- POSITED
    "A1": Axiom(
        "A1", POSITED,
        "pursuer Dubins kinematics: x_P'=cos(theta), y_P'=sin(theta), theta'=phi (v_P=1)",
    ),
    "A2": Axiom(
        "A2", POSITED,
        "evader simple motion: x_E'=w*cos(psi_lab), y_E'=w*sin(psi_lab) (v_E=w)",
    ),
    "A3": Axiom(
        "A3", POSITED,
        "pursuer control bound |phi| <= 1 (turn radius normalized to R_min=1)",
        relation=sp.Le(sp.Abs(_phi), 1),
    ),
    "A4": Axiom(
        "A4", POSITED,
        "speed ratio 0 < w < 1, with normalization v_P=1, v_E=w",
        relation=sp.And(_w > 0, _w < 1),
    ),
    "A5": Axiom(
        "A5", POSITED,
        "capture set is the disk x_1^2 + x_2^2 <= ell_tilde^2",
        relation=sp.Le(_x1**2 + _x2**2, _ell**2),
    ),
    "A6": Axiom(
        "A6", POSITED,
        "running cost L=1 (minimize time to capture), terminal cost 0",
        relation=sp.Eq(sp.Symbol("L"), 1),
    ),
    "A7": Axiom(
        "A7", POSITED,
        "zero-sum objective: V* = min over Phi of max over Psi of the value (saddle over policies)",
    ),
    # -------------------------------------------------------------- INHERITED
    "I1": Axiom(
        "I1", INHERITED,
        "Pontryagin/HJB: the costate is p = grad V*, and the adjoint evolves as p' = -dH/dx",
        citation="Isaacs 1965; Pontryagin et al. 1962",
    ),
    "I2": Axiom(
        "I2", INHERITED,
        "verification / Isaacs' condition: the POINTWISE Hamiltonian saddle is globally "
        "optimal (needs H separable in phi,psi; that sub-fact is checked in L3/lemma_03)",
        citation="Isaacs 1965; Bardi & Capuzzo-Dolcetta 1997",
    ),
    "I3": Axiom(
        "I3", INHERITED,
        "the value V* exists as the (viscosity) solution of the HJI equation, so H=0 holds",
        citation="Bardi, Falcone & Soravia 1994; Evans & Souganidis 1984",
    ),
    # ------------------------------------------------------------- REGULARITY
    "R*": Axiom(
        "R*", REGULARITY,
        "V* in C^1 wherever we differentiate (so grad V* exists and the chain rule holds). "
        "FAILS on singular surfaces: the dispersal surface (the star) is where the costate "
        "is multivalued and the smooth derivation does not apply.",
    ),
}


def get(name):
    return AXIOMS[name]


def by_tag(tag):
    return [a for a in AXIOMS.values() if a.tag == tag]

"""L6 - Costate dynamics (closes a gap R5 leaves implicit).

The adjoint equation p' = -dH/dx (I1) makes the costate a pure rotation:
    p1' = -dH/dx1 = -phi*p2,   p2' = -dH/dx2 = +phi*p1.
R5 uses these derivatives inside its conservation check but never states them; here we
assert the explicit closed form.
"""

import sympy as sp

from derivation import symbols as S

TITLE = "Costate dynamics: p' = -dH/dx = (-phi*p2, phi*p1)"
STATEMENT = "p1' = -phi*p2,  p2' = phi*p1  (a pure rotation of p)"
AXIOMS_USED = ["I1", "R*"]


def derive(ctx):
    H = ctx["H"]
    x1, x2, p1, p2, phi = S.x1, S.x2, S.p1, S.p2, S.phi

    p1_dot = -sp.diff(H, x1)
    p2_dot = -sp.diff(H, x2)

    ctx["p1_dot"] = sp.simplify(p1_dot)
    ctx["p2_dot"] = sp.simplify(p2_dot)
    return {"p1_dot": ctx["p1_dot"], "p2_dot": ctx["p2_dot"]}


def verify(ctx):
    p1, p2, phi = S.p1, S.p2, S.phi
    assert sp.simplify(ctx["p1_dot"] - (-phi * p2)) == 0, f"L6 FAIL: p1' = {ctx['p1_dot']}"
    assert sp.simplify(ctx["p2_dot"] - (phi * p1)) == 0, f"L6 FAIL: p2' = {ctx['p2_dot']}"

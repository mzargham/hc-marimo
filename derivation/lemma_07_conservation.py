"""L7 - The costate norm is conserved (the cancellation Isaacs did by hand).

From the rotation p1' = -phi*p2, p2' = phi*p1 (L6),
    d/dt (p1^2 + p2^2) = 2(p1 p1' + p2 p2') = 2(p1(-phi p2) + p2(phi p1)) = 0.
||p|| is constant along the optimal flow: p spins, never changes length.
"""

import sympy as sp

from derivation import symbols as S

TITLE = "Conservation: d/dt ||p||^2 = 0"
STATEMENT = "2(p1 p1' + p2 p2') = 0  along the adjoint flow"
AXIOMS_USED = []  # algebraic consequence of L6


def derive(ctx):
    p1, p2 = S.p1, S.p2
    d_norm_sq = sp.simplify(2 * (p1 * ctx["p1_dot"] + p2 * ctx["p2_dot"]))
    ctx["d_norm_sq"] = d_norm_sq
    return {"d_norm_sq": d_norm_sq}


def verify(ctx):
    assert ctx["d_norm_sq"] == 0, f"L7 FAIL: d/dt ||p||^2 = {ctx['d_norm_sq']}, expected 0"

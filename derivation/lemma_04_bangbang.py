"""L4 - Pursuer plays bang-bang (closes a gap R3 leaves open).

The pursuer minimizes H over the action set |phi| <= 1 (A3). Since the only phi-dependence
is the linear term sigma*phi, minimizing a linear function on [-1,1] lands on the boundary
opposite the slope:
    phi* = -sign(sigma),   with minimum value  -|sigma|.
The flip at sigma=0 is the star (the dispersal surface) from the talk.

R3 only produces sigma; here we assert the ARGMIN step symbolically:
    a linear objective on [-1,1] is minimized at an endpoint, giving -sign(slope), value
    -|slope|, and that value is attained by phi* = -sign(sigma).
"""

import sympy as sp

from derivation import symbols as S

TITLE = "Pursuer bang-bang: phi* = -sign(sigma)"
STATEMENT = "argmin_{|phi|<=1} (sigma*phi) = -sign(sigma), value = -|sigma|"
AXIOMS_USED = ["A3", "A7", "I2"]


def derive(ctx):
    sigma = ctx["sigma"]
    phi_star = -sp.sign(sigma)
    ctx["phi_star"] = phi_star
    return {"phi_star": phi_star}


def verify(ctx):
    sigma = ctx["sigma"]
    phi = S.phi
    phi_star = ctx["phi_star"]

    # (a) The phi-dependent part of H is exactly the linear term sigma*phi.
    linear_in_phi = ctx["H"].coeff(phi) * phi
    assert sp.simplify(linear_in_phi - sigma * phi) == 0, "L4 FAIL: phi-part is not sigma*phi"

    # (b) Abstract argmin of a LINEAR objective on the feasible interval [-1,1] (A3):
    # a real slope s gives min_{|phi|<=1} s*phi = min(s*(+1), s*(-1)) = -|s|, at phi=-sign(s).
    s = sp.Symbol("s", real=True)
    endpoint_min = sp.Min(s * 1, s * (-1)).rewrite(sp.Piecewise)
    assert sp.simplify(endpoint_min - (-sp.Abs(s)).rewrite(sp.Piecewise)) == 0, \
        "L4 FAIL: min on [-1,1] != -|s|"
    assert sp.simplify((s * (-sp.sign(s))) - (-sp.Abs(s))) == 0, \
        "L4 FAIL: phi=-sign(s) does not attain -|s|"

    # (c) Instantiate at slope = sigma: phi* attains the minimum value -|sigma|.
    assert sp.simplify(sigma * phi_star - (-sp.Abs(sigma))) == 0, \
        f"L4 FAIL: sigma*phi* = {sp.simplify(sigma * phi_star)}, expected -|sigma|"

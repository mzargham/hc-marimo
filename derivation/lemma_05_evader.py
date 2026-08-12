"""L5 - Evader runs up the value gradient (closes a gap R4 leaves open).

The evader maximizes the psi-dependent part of H,
    g(psi) = w*(p1*sin(psi) + p2*cos(psi)),
over the heading psi. Stationarity gives psi* = atan2(p1, p2): the velocity points along
p = grad V*, i.e. straight up the value gradient. The optimized contribution is w*||p||.

R4 only checks the value numerically. Here we assert the argmax SYMBOLICALLY. To avoid
fragile atan2 simplification we represent the maximizer by its sine/cosine on the unit
circle, (sin psi*, cos psi*) = (p1, p2)/||p|| (which IS atan2(p1,p2)), and verify:
  * stationarity: d g/d psi = 0 there;
  * it is the MAX (not the min): value = +w*||p|| > -w*||p|| at the antipode.
"""

import sympy as sp

from derivation import symbols as S

TITLE = "Evader up-gradient: psi* = atan2(p1, p2), contribution w*||p||"
STATEMENT = "argmax_psi w*(p1 sin psi + p2 cos psi) at psi*=atan2(p1,p2), value = w*||p||"
AXIOMS_USED = ["A4", "A7", "I2"]


def derive(ctx):
    p1, p2, w, psi = S.p1, S.p2, S.w, S.psi
    norm = S.norm_p

    g = w * (p1 * sp.sin(psi) + p2 * sp.cos(psi))

    # The maximizer, given by its sine and cosine (this is exactly atan2(p1, p2)).
    sin_star = p1 / norm
    cos_star = p2 / norm
    value = sp.simplify(w * (p1 * sin_star + p2 * cos_star))

    ctx["g_psi"] = g
    ctx["sin_psi_star"] = sin_star
    ctx["cos_psi_star"] = cos_star
    ctx["psi_star"] = sp.atan2(p1, p2)
    ctx["evader_value"] = value
    return {"psi_star": ctx["psi_star"], "value": value}


def verify(ctx):
    p1, p2, w, psi = S.p1, S.p2, S.w, S.psi
    norm = S.norm_p
    g = ctx["g_psi"]
    sin_star, cos_star = ctx["sin_psi_star"], ctx["cos_psi_star"]

    # The maximizer really lies on the unit circle.
    assert sp.simplify(sin_star**2 + cos_star**2 - 1) == 0, "L5 FAIL: (sin*,cos*) not unit"

    # Stationarity: dg/dpsi = w*(p1 cos psi - p2 sin psi); substituting sin*/cos* -> 0.
    dg = sp.diff(g, psi)
    dg_at_star = dg.subs({sp.sin(psi): sin_star, sp.cos(psi): cos_star})
    assert sp.simplify(dg_at_star) == 0, f"L5 FAIL: not stationary, dg = {dg_at_star}"

    # Value at the maximizer is +w*||p|| (the true max; the antipode gives -w*||p||).
    val_max = w * (p1 * sin_star + p2 * cos_star)
    assert sp.simplify(val_max - w * norm) == 0, f"L5 FAIL: value = {sp.simplify(val_max)}"
    val_min = w * (p1 * (-sin_star) + p2 * (-cos_star))
    assert sp.simplify(val_min + w * norm) == 0, "L5 FAIL: antipode is not the minimum"

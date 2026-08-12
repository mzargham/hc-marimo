"""L3 - Switching function and separability.

H is LINEAR in the pursuer turn rate phi (no phi^2), and its phi-coefficient is the
switching function
    sigma = p2*x1 - p1*x2.
Also: H has no phi*psi cross terms, so it is separable in the two controls. That
separability is the SymPy-checkable sub-fact underpinning Isaacs' condition (I2).
"""

import sympy as sp

from derivation import symbols as S

TITLE = "Switching function sigma = p2*x1 - p1*x2, and (phi,psi)-separability"
STATEMENT = "sigma = coeff_phi(H) = p2*x1 - p1*x2;  no phi*psi cross terms"
AXIOMS_USED = []  # pure algebra on the L2 object; substantiates a sub-fact of I2


def derive(ctx):
    H = ctx["H"]
    x1, x2, p1, p2, phi = S.x1, S.x2, S.p1, S.p2, S.phi

    sigma = H.coeff(phi)
    ctx["sigma"] = sp.expand(sigma)
    return {"sigma": ctx["sigma"]}


def verify(ctx):
    H = ctx["H"]
    x1, x2, p1, p2, phi, psi = S.x1, S.x2, S.p1, S.p2, S.phi, S.psi
    sigma = ctx["sigma"]

    # Linear in phi (bang-bang structure): no quadratic term, nonzero linear term.
    assert H.coeff(phi, 2) == 0, "L3 FAIL: H has a phi^2 term"
    assert H.coeff(phi, 1) != 0, "L3 FAIL: H has no phi term"

    # The switching function is exactly p2*x1 - p1*x2.
    assert sp.simplify(sigma - (p2 * x1 - p1 * x2)) == 0, f"L3 FAIL: sigma = {sigma}"

    # Separability: the phi-carrying part carries no psi, and the rest carries no phi.
    phi_part = H.coeff(phi) * phi
    rest = sp.expand(H - phi_part)
    assert rest.coeff(phi) == 0, "L3 FAIL: residual phi dependence outside sigma*phi"
    assert phi_part.coeff(sp.sin(psi)) == 0, "L3 FAIL: phi*sin(psi) cross term"
    assert phi_part.coeff(sp.cos(psi)) == 0, "L3 FAIL: phi*cos(psi) cross term"

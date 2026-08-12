"""L8 - The characteristic (retrograde) system.

Substitute the optimal actions phi* = -sign(sigma) (L4) and the evader optimum
sin psi* = p1/||p||, cos psi* = p2/||p|| (L5) into the dynamics (L1) and costate (L6) to
get the closed 4D system that gets lambdified and integrated:
    x1' = -phi* x2 + w p1/||p||
    x2' =  phi* x1 + w p2/||p|| - 1
    p1' = -phi* p2
    p2' =  phi* p1,          phi* = -sign(p2 x1 - p1 x2).

The lambdify seam is inherently numeric, so this lemma's check is the one numeric endpoint
the chain allows (mirrors T1): the lambdified RHS agrees with the hand-coded field to
machine precision.
"""

import numpy as np
import sympy as sp

from derivation import symbols as S

TITLE = "Characteristic system: sub phi*, psi* -> closed 4D ODE"
STATEMENT = "(x1', x2', p1', p2') with phi*=-sign(sigma), psi*=atan2(p1,p2)"
AXIOMS_USED = []  # assembly of L1/L4/L5/L6 outputs


def derive(ctx):
    x1, x2, p1, p2, w = S.x1, S.x2, S.p1, S.p2, S.w
    norm = S.norm_p
    phi_star = ctx["phi_star"]
    sin_star, cos_star = ctx["sin_psi_star"], ctx["cos_psi_star"]

    # f1, f2 with the evader optimum substituted (sin/cos psi* -> p/||p||).
    f1_opt = S.f1_canonical.subs({sp.sin(S.psi): sin_star})
    f2_opt = S.f2_canonical.subs({sp.cos(S.psi): cos_star})

    rhs_x1 = f1_opt.subs(S.phi, phi_star)
    rhs_x2 = f2_opt.subs(S.phi, phi_star)
    rhs_p1 = ctx["p1_dot"].subs(S.phi, phi_star)
    rhs_p2 = ctx["p2_dot"].subs(S.phi, phi_star)

    rhs = [rhs_x1, rhs_x2, rhs_p1, rhs_p2]
    ctx["rhs"] = rhs
    ctx["rhs_fn"] = sp.lambdify([x1, x2, p1, p2, w], rhs, modules=["numpy"])
    return {"rhs": rhs}


def _hand_coded(state, w_val):
    """The independent hand-written field (same as test_phase2.rhs_forward_numpy)."""
    x1, x2, p1, p2 = state
    norm_p = np.sqrt(p1**2 + p2**2)
    sigma = p2 * x1 - p1 * x2
    phi_star = -np.sign(sigma)
    return [
        -phi_star * x2 + w_val * p1 / norm_p,
        phi_star * x1 + w_val * p2 / norm_p - 1.0,
        -phi_star * p2,
        phi_star * p1,
    ]


def verify(ctx):
    fn = ctx["rhs_fn"]
    rng = np.random.default_rng(20260715)
    for _ in range(50):
        x1 = rng.uniform(-5, 5)
        x2 = rng.uniform(-5, 5)
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.1, 3.0)
        p1, p2 = r * np.cos(ang), r * np.sin(ang)
        w_val = rng.uniform(0.05, 0.5)
        state = [x1, x2, p1, p2]
        lamb = fn(*state, w_val)
        hand = _hand_coded(state, w_val)
        for j in range(4):
            assert abs(float(lamb[j]) - hand[j]) < 1e-10, \
                f"L8 FAIL: component {j} mismatch at {state}, w={w_val}"

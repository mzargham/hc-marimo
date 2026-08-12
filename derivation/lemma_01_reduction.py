"""L1 - Body-frame reduction (closes the gap R1 leaves open).

Claim: pinning the frame to the pursuer collapses the 5-DOF lab problem to Isaacs'
2-DOF reduced dynamics
    f1 = -phi*x2 + w*sin(psi),   f2 = phi*x1 + w*cos(psi) - 1,
and the pursuer heading theta CANCELS out entirely.

R1 in test_phase1.py only spot-checks this numerically (the Function(t) objects make a
direct symbolic compare awkward). Here we assert it as a SYMBOLIC IDENTITY:
    simplify(f_derived - f_canonical) == 0.
"""

import sympy as sp

from derivation import symbols as S

TITLE = "Body-frame reduction: 5 DOF -> 2 DOF, theta cancels"
STATEMENT = "f = (-phi*x2 + w*sin(psi),  phi*x1 + w*cos(psi) - 1)"
AXIOMS_USED = ["A1", "A2", "A4"]  # A3 (|phi|<=1) is NOT used to DERIVE f; it enters at L4.


def derive(ctx):
    x1, x2 = S.x1, S.x2
    phi, psi, w = S.phi, S.psi, S.w
    theta, Dx, Dy = S.theta, S.Dx, S.Dy

    # Isaacs' body-frame coordinates in terms of lab relative position (Dx, Dy) and heading.
    x1_def = -Dx * sp.sin(theta) + Dy * sp.cos(theta)
    x2_def = Dx * sp.cos(theta) + Dy * sp.sin(theta)

    # Lab-frame relative velocities. A1: pursuer (v_P=1); A2: evader (v_E=w); with
    # psi_lab = psi + theta (evader heading measured relative to the pursuer), and A4's
    # normalization v_P=1, v_E=w already applied.
    Dx_dot = w * sp.cos(psi + theta) - sp.cos(theta)   # x_E' - x_P'
    Dy_dot = w * sp.sin(psi + theta) - sp.sin(theta)   # y_E' - y_P'
    theta_dot = phi                                    # A1

    # Total time derivative via the chain rule (Dx, Dy, theta all move).
    x1_dot = (sp.diff(x1_def, Dx) * Dx_dot
              + sp.diff(x1_def, Dy) * Dy_dot
              + sp.diff(x1_def, theta) * theta_dot)
    x2_dot = (sp.diff(x2_def, Dx) * Dx_dot
              + sp.diff(x2_def, Dy) * Dy_dot
              + sp.diff(x2_def, theta) * theta_dot)

    # Re-express the lab relative position through the body-frame state (invert the defs):
    #   Dx = -x1*sin(theta) + x2*cos(theta),  Dy = x1*cos(theta) + x2*sin(theta).
    Dx_sub = -x1 * sp.sin(theta) + x2 * sp.cos(theta)
    Dy_sub = x1 * sp.cos(theta) + x2 * sp.sin(theta)

    f1 = sp.trigsimp(sp.expand_trig(x1_dot.subs({Dx: Dx_sub, Dy: Dy_sub})))
    f2 = sp.trigsimp(sp.expand_trig(x2_dot.subs({Dx: Dx_sub, Dy: Dy_sub})))

    ctx["f1"], ctx["f2"] = f1, f2
    return {"f1": f1, "f2": f2}


def verify(ctx):
    f1, f2 = ctx["f1"], ctx["f2"]
    # The whole point: theta has cancelled and the result equals Isaacs' canonical form.
    assert sp.simplify(f1 - S.f1_canonical) == 0, f"L1 FAIL: f1 = {f1}"
    assert sp.simplify(f2 - S.f2_canonical) == 0, f"L1 FAIL: f2 = {f2}"
    assert S.theta not in f1.free_symbols, "L1 FAIL: theta did not cancel from f1"
    assert S.theta not in f2.free_symbols, "L1 FAIL: theta did not cancel from f2"

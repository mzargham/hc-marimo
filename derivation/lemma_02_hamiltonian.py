"""L2 - The Hamiltonian falls out of the value function.

The value under optimal play is the state-dependent time-to-go, the bare integral of the
running cost to capture,
    V*(x(t)) = integral_t^{T} 1 ds.
Its gradient p = grad V* prices every direction: moving with the dynamics f = x' changes
the value at rate
    dV*/dt = grad V* . x' = p . f          [needs p = grad V*, R*].
The value is a clock, so on an optimal path it must fall at the running cost rate,
dV*/dt = -1 (the running cost 1, back by the fundamental theorem). Optimality is exactly
the condition that the two rates agree; that is the Hamilton-Jacobi-Isaacs equation
    H := p . f + 1 = 0        (= p . f - (-1)).
H=0 is INHERITED (I1/I3), not proven here: it is the definition of the optimal value, not a
consequence of the smooth algebra below.

We assemble H as a lemma object and SYMBOLICALLY verify the FTC step that regenerates the
"+1" (the "where does the 1 come from" question, closed). We do NOT assert H=0; that is the
inherited HJI condition (I1, I3), cited only.
"""

import sympy as sp

from derivation import symbols as S

TITLE = "Hamiltonian from the value function: H = p.f + 1"
STATEMENT = "H = p1*f1 + p2*f2 + 1   (=0 on optimal play, by HJB)"
AXIOMS_USED = ["A5", "A6", "A7", "I1", "I3", "R*"]


def derive(ctx):
    p1, p2 = S.p1, S.p2
    f1, f2 = ctx["f1"], ctx["f2"]

    # FTC step: cost-to-go of the running cost L=1 from time tau to capture time T.
    s, tau, T = sp.symbols("s tau T", positive=True)
    L = sp.Integer(1)                       # A6: running cost
    V = sp.integrate(L, (s, tau, T))        # = T - tau  (bare integral, time-to-go)
    dV_dtau = sp.diff(V, tau)               # = -1, the integrand back by the FTC

    # Assemble H = grad V* . f + L  with grad V* = p (I1) and L=1 (A6).
    H = p1 * f1 + p2 * f2 + 1

    ctx["H"] = sp.expand(H)
    ctx["dV_dtau"] = dV_dtau
    return {"H": ctx["H"], "dV_dtau": dV_dtau}


def verify(ctx):
    H = ctx["H"]
    p1, p2 = S.p1, S.p2
    f1, f2 = ctx["f1"], ctx["f2"]

    # The FTC that regenerates the running-cost term: d/dtau of the time-to-go is -1,
    # so the running cost L=1 reappears as the "+1" in H.
    assert ctx["dV_dtau"] == -1, f"L2 FAIL: FTC gave dV/dtau = {ctx['dV_dtau']}, expected -1"

    # H is exactly p.f + 1 (the inner product of costate and dynamics plus the running cost).
    assert sp.expand(H - (p1 * f1 + p2 * f2 + 1)) == 0, "L2 FAIL: H is not p.f + 1"

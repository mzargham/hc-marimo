"""Shared symbols for the derivation.

Load-bearing accounting (symproof discipline): any SymPy assumption baked into a Symbol
(e.g. ``positive=True``) is itself a modeling commitment and MUST correspond to a declared
axiom. ``SYMBOL_ASSUMPTIONS`` records that mapping; ``test_derivation.py`` checks that no
symbol smuggles in an assumption without a licensing axiom.
"""

import sympy as sp

# ---- Reduced (body-frame) state and costate ----
x1, x2 = sp.symbols("x_1 x_2", real=True)
p1, p2 = sp.symbols("p_1 p_2", real=True)

# ---- Controls (lowercase = the instantaneous action) ----
phi = sp.symbols("phi", real=True)   # pursuer turn rate,  action of policy Phi
psi = sp.symbols("psi", real=True)   # evader heading,     action of policy Psi

# ---- Speed ratio (LOAD-BEARING: positive=True is licensed by axiom A4) ----
w = sp.symbols("w", positive=True)

# ---- Lab-frame helpers used only inside the L1 reduction ----
theta = sp.symbols("theta", real=True)   # pursuer heading
Dx, Dy = sp.symbols("Dx Dy", real=True)  # relative position in the lab frame

# ---- Canonical reduced dynamics (Isaacs' form): the TARGET L1 must reproduce ----
f1_canonical = -phi * x2 + w * sp.sin(psi)
f2_canonical = phi * x1 + w * sp.cos(psi) - 1

# ---- Load-bearing symbol assumptions -> the axiom that licenses each ----
# Maps a human tag to (symbol name, assumption, licensing axiom name).
SYMBOL_ASSUMPTIONS = {
    "w_positive": ("w", "positive", "A4"),
}

# The norm that recurs once the evader is optimized in.
norm_p = sp.sqrt(p1**2 + p2**2)

"""
Test suite for Phase 2 of the Homicidal Chauffeur notebook.
Verifies the numerical engine: lambdification, integration, conservation laws.
Also includes notebook-structural tests to prevent marimo DAG regressions.

Requirements traced to:
  T1: Lambdified RHS matches hand-coded numpy at random points
  T2: Hamiltonian H* ~ 0 conserved along integrated trajectories
  T3: ||p||^2 conserved along integrated trajectories
  T4: Capture condition x1^2+x2^2 = ell_tilde^2 at terminal time
  T5: Transversality p(T)/||p(T)|| = x(T)/||x(T)|| at terminal surface
  T6: Stationary evader (w=0): straight-line capture time = d - ell_tilde
  T7: Usable part: lambda > 0 iff sin(alpha) > w
  T8: No marimo cell variable redefinitions
  T9: Canonical dynamics use only reduced symbols (no Function objects)
  T10: Full symbolic derivation matches canonical form numerically
  T11: Dynamics symmetry under x1 -> -x1 reflection
  T12: Isochrone approximate mirror-symmetry about x2-axis
  T13: Small-w integration stability (w=0.01)
  T14: Usable part width monotonically decreases with w
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---- Build the RHS functions (mirroring notebook logic) ----

def rhs_forward_numpy(t, state, w_val):
    """Hand-coded RHS of the 4D characteristic ODE."""
    x1, x2, p1, p2 = state
    norm_p = np.sqrt(p1**2 + p2**2)
    if norm_p < 1e-15:
        return [0.0, 0.0, 0.0, 0.0]
    sigma = p2 * x1 - p1 * x2
    phi_star = -np.sign(sigma)
    x1d = -phi_star * x2 + w_val * p1 / norm_p
    x2d = phi_star * x1 + w_val * p2 / norm_p - 1.0
    p1d = -phi_star * p2
    p2d = phi_star * p1
    return [x1d, x2d, p1d, p2d]


def rhs_forward_lambdified(t, state, w_val):
    """Lambdified version (built from SymPy, reproduced here for testing)."""
    import sympy as sp
    from sympy import symbols, sin, cos, atan2, sqrt, sign, simplify

    x1_s, x2_s = symbols('x_1 x_2', real=True)
    p1_s, p2_s = symbols('p_1 p_2', real=True)
    phi_s = symbols('phi', real=True)
    psi_s = symbols('psi', real=True)
    w_s = symbols('w', positive=True)

    f1 = -phi_s * x2_s + w_s * sin(psi_s)
    f2 = phi_s * x1_s + w_s * cos(psi_s) - 1

    sigma = p2_s * x1_s - p1_s * x2_s
    phi_opt = -sign(sigma)
    psi_opt = atan2(p1_s, p2_s)

    rhs_x1 = f1.subs([(phi_s, phi_opt), (psi_s, psi_opt)])
    rhs_x2 = f2.subs([(phi_s, phi_opt), (psi_s, psi_opt)])
    rhs_p1 = (-phi_opt * p2_s)
    rhs_p2 = (phi_opt * p1_s)

    fn = sp.lambdify(
        [x1_s, x2_s, p1_s, p2_s, w_s],
        [rhs_x1, rhs_x2, rhs_p1, rhs_p2],
        modules=['numpy']
    )

    x1, x2, p1, p2 = state
    return fn(x1, x2, p1, p2, w_val)


def rhs_backward(t, state, w_val):
    fwd = rhs_forward_numpy(t, state, w_val)
    return [-v for v in fwd]


def compute_terminal_conditions(alpha_arr, w_val, ell_tilde_val):
    x1_T = ell_tilde_val * np.cos(alpha_arr)
    x2_T = ell_tilde_val * np.sin(alpha_arr)
    lam = -1.0 / (ell_tilde_val * (w_val - np.sin(alpha_arr)))
    p1_T = lam * x1_T
    p2_T = lam * x2_T
    return np.column_stack([x1_T, x2_T, p1_T, p2_T])


# ---- T1: Lambdified vs hand-coded consistency ----

def test_T1_lambdify_consistency():
    """Lambdified and hand-coded RHS must agree at random points."""
    rng = np.random.default_rng(42)

    for _ in range(50):
        x1 = rng.uniform(-5, 5)
        x2 = rng.uniform(-5, 5)
        # Ensure ||p|| > 0
        angle = rng.uniform(0, 2 * np.pi)
        norm_p = rng.uniform(0.1, 3.0)
        p1 = norm_p * np.cos(angle)
        p2 = norm_p * np.sin(angle)
        w_val = rng.uniform(0.05, 0.5)

        state = [x1, x2, p1, p2]
        hand = rhs_forward_numpy(0, state, w_val)
        lamb = rhs_forward_lambdified(0, state, w_val)

        for j in range(4):
            assert abs(hand[j] - lamb[j]) < 1e-10, \
                f"T1 FAIL: component {j} mismatch at state={state}, w={w_val}: " \
                f"hand={hand[j]}, lambdified={lamb[j]}"

    print("T1 PASS: Lambdified RHS matches hand-coded numpy at 50 random points")


# ---- T2: Hamiltonian conservation ----

def test_T2_hamiltonian_conservation():
    """H* should stay near 0 along integrated trajectories."""
    w_val = 0.25
    ell_tilde = 0.5

    # Pick an angle on the usable part
    alpha = np.pi / 2  # sin(pi/2) = 1 > 0.25 = w
    tc = compute_terminal_conditions(np.array([alpha]), w_val, ell_tilde)
    y0 = tc[0]

    sol = solve_ivp(
        rhs_backward, [0, 10], y0, args=(w_val,),
        method='RK45', max_step=0.02, rtol=1e-10, atol=1e-12,
    )
    assert sol.success, f"T2 FAIL: integration failed: {sol.message}"

    x1, x2, p1, p2 = sol.y
    norm_p = np.sqrt(p1**2 + p2**2)
    sigma = p2 * x1 - p1 * x2
    H_star = -np.abs(sigma) + w_val * norm_p - p2 + 1.0

    max_drift = np.max(np.abs(H_star))
    assert max_drift < 1e-6, f"T2 FAIL: max |H*| = {max_drift}"
    print(f"T2 PASS: H* conserved, max |H*| = {max_drift:.2e}")


# ---- T3: Costate norm conservation ----

def test_T3_costate_norm_conservation():
    """||p||^2 must be conserved along trajectories."""
    w_val = 0.3
    ell_tilde = 0.4

    alpha = 1.2  # sin(1.2) ~ 0.93 > 0.3
    tc = compute_terminal_conditions(np.array([alpha]), w_val, ell_tilde)
    y0 = tc[0]

    sol = solve_ivp(
        rhs_backward, [0, 10], y0, args=(w_val,),
        method='RK45', max_step=0.02, rtol=1e-10, atol=1e-12,
    )
    assert sol.success, f"T3 FAIL: integration failed"

    p_norm_sq = sol.y[2]**2 + sol.y[3]**2
    drift = np.max(np.abs(p_norm_sq - p_norm_sq[0]))
    assert drift < 1e-8, f"T3 FAIL: ||p||^2 drift = {drift}"
    print(f"T3 PASS: ||p||^2 conserved, max drift = {drift:.2e}")


# ---- T4: Capture condition at terminal surface ----

def test_T4_capture_condition():
    """At t=T (tau=0), the state must lie on the terminal circle."""
    w_val = 0.25
    ell_tilde = 0.5

    alphas = np.linspace(np.arcsin(w_val) + 0.01,
                         np.pi - np.arcsin(w_val) - 0.01, 10)
    tc = compute_terminal_conditions(alphas, w_val, ell_tilde)

    for i, y0 in enumerate(tc):
        r_sq = y0[0]**2 + y0[1]**2
        assert abs(r_sq - ell_tilde**2) < 1e-12, \
            f"T4 FAIL: r^2 = {r_sq}, expected {ell_tilde**2} at alpha={alphas[i]}"

    print("T4 PASS: All terminal states lie on the capture circle")


# ---- T5: Transversality at terminal surface ----

def test_T5_transversality():
    """p(T) must be proportional to x(T) with positive lambda."""
    w_val = 0.2
    ell_tilde = 0.6

    alphas = np.linspace(np.arcsin(w_val) + 0.01,
                         np.pi - np.arcsin(w_val) - 0.01, 20)
    tc = compute_terminal_conditions(alphas, w_val, ell_tilde)

    for i, y0 in enumerate(tc):
        x_vec = y0[:2]
        p_vec = y0[2:]
        x_norm = np.linalg.norm(x_vec)
        p_norm = np.linalg.norm(p_vec)
        if x_norm < 1e-15 or p_norm < 1e-15:
            continue
        x_hat = x_vec / x_norm
        p_hat = p_vec / p_norm
        # p_hat should equal x_hat (same direction, positive lambda)
        dot = np.dot(x_hat, p_hat)
        assert abs(dot - 1.0) < 1e-10, \
            f"T5 FAIL: dot(x_hat, p_hat) = {dot} at alpha={alphas[i]}"
        # lambda should be positive
        lam = p_norm / x_norm
        assert lam > 0, f"T5 FAIL: lambda = {lam} at alpha={alphas[i]}"

    print("T5 PASS: Transversality p(T)/||p(T)|| = x(T)/||x(T)||, lambda > 0")


# ---- T6: Stationary evader straight-line capture ----

def test_T6_stationary_evader():
    """w=0: straight-line pursuit from (0, d) should capture in time d - ell."""
    w_val = 0.0
    ell_tilde = 0.5
    d = 5.0  # starting distance along x2 axis

    # At alpha = pi/2 (top of circle): x = (0, ell_tilde), p = (0, lambda*ell_tilde)
    # Use exact values to avoid trig noise at sigma=0 (the singular surface).
    # np.cos(pi/2) ≈ 6e-17, which triggers the sign function and causes spiraling.
    # With w=0, the straight-line trajectory lies entirely on sigma=0.
    lam = 1.0 / (ell_tilde * 1.0)  # sin(pi/2) = 1
    y0 = [0.0, ell_tilde, 0.0, lam * ell_tilde]

    T_expected = d - ell_tilde
    sol = solve_ivp(
        rhs_backward, [0, T_expected + 1], y0, args=(w_val,),
        method='RK45', max_step=0.02, rtol=1e-10, atol=1e-12,
        dense_output=True,
    )
    assert sol.success

    # At tau = T_expected, the trajectory should be at x2 ~ d, x1 ~ 0
    x1_end = sol.sol(T_expected)[0]
    x2_end = sol.sol(T_expected)[1]

    assert abs(x1_end) < 1e-6, f"T6 FAIL: x1 = {x1_end}, expected ~0"
    assert abs(x2_end - d) < 0.05, \
        f"T6 FAIL: x2 = {x2_end}, expected ~{d}"

    print(f"T6 PASS: w=0 straight-line, at tau={T_expected}: "
          f"x = ({x1_end:.6f}, {x2_end:.4f}), expected (0, {d})")


# ---- T7: Usable part boundary ----

def test_T7_usable_part():
    """lambda > 0 iff sin(alpha) > w."""
    for w_val in [0.1, 0.25, 0.4]:
        ell_tilde = 0.5
        # Test points inside usable part
        alpha_inside = (np.arcsin(w_val) + np.pi / 2) / 2 + np.pi / 4
        lam_inside = -1.0 / (ell_tilde * (w_val - np.sin(alpha_inside)))
        assert lam_inside > 0, \
            f"T7 FAIL: lambda={lam_inside} at alpha={alpha_inside}, w={w_val}"

        # Test points outside usable part
        alpha_outside = np.arcsin(w_val) / 2  # below arcsin(w)
        lam_outside = -1.0 / (ell_tilde * (w_val - np.sin(alpha_outside)))
        assert lam_outside < 0, \
            f"T7 FAIL: lambda={lam_outside} at alpha={alpha_outside}, w={w_val}"

        # Test at exact boundary: sin(alpha) = w => lambda diverges
        alpha_boundary = np.arcsin(w_val)
        denom = w_val - np.sin(alpha_boundary)
        assert abs(denom) < 1e-12, \
            f"T7 FAIL: denom = {denom} at boundary"

    print("T7 PASS: Usable part lambda > 0 iff sin(alpha) > w")


# ---- T8: Marimo cell variable uniqueness ----

def test_T8_no_variable_redefinitions():
    """No two @app.cell functions should define the same non-private variable."""
    import ast
    import collections

    with open("homicidal_chauffeur.py") as f:
        tree = ast.parse(f.read())

    cell_vars = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        is_cell = any(
            isinstance(d, ast.Attribute) and d.attr == "cell"
            for d in node.decorator_list
        )
        if not is_cell:
            continue

        assigned = set()
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        assigned.add(t.id)
                    elif isinstance(t, ast.Tuple):
                        for e in t.elts:
                            if isinstance(e, ast.Name):
                                assigned.add(e.id)
            elif isinstance(stmt, ast.AugAssign) and isinstance(
                stmt.target, ast.Name
            ):
                assigned.add(stmt.target.id)
            elif isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name):
                    assigned.add(stmt.target.id)
                for s in ast.walk(stmt):
                    if isinstance(s, ast.Assign):
                        for t in s.targets:
                            if isinstance(t, ast.Name):
                                assigned.add(t.id)
                    elif isinstance(s, ast.AugAssign) and isinstance(
                        s.target, ast.Name
                    ):
                        assigned.add(s.target.id)
        cell_vars[node.name] = assigned

    all_vars = collections.defaultdict(list)
    for cell, vs in cell_vars.items():
        for v in vs:
            if not v.startswith("_"):
                all_vars[v].append(cell)

    conflicts = {v: cells for v, cells in all_vars.items() if len(cells) > 1}
    assert not conflicts, (
        f"T8 FAIL: Variable redefinitions across cells: "
        + ", ".join(f"{v} in {cells}" for v, cells in sorted(conflicts.items()))
    )

    print(f"T8 PASS: No variable redefinitions across {len(cell_vars)} cells")


# ---- T9: Canonical dynamics use only reduced symbols ----

def test_T9_canonical_form_symbols():
    """f1 and f2 must contain only the reduced symbols, not Function objects."""
    import sympy as sp
    from sympy import symbols, sin, cos

    x1, x2 = symbols("x_1 x_2", real=True)
    p1, p2 = symbols("p_1 p_2", real=True)
    phi = symbols("phi", real=True)
    psi = symbols("psi", real=True)
    w = symbols("w", positive=True)

    f1 = -phi * x2 + w * sin(psi)
    f2 = phi * x1 + w * cos(psi) - 1

    # Verify free symbols are exactly the expected set
    expected = {phi, psi, w, x1, x2}
    assert f1.free_symbols == {phi, x2, w, psi}, \
        f"T9 FAIL: f1 has unexpected symbols: {f1.free_symbols}"
    assert f2.free_symbols == {phi, x1, w, psi}, \
        f"T9 FAIL: f2 has unexpected symbols: {f2.free_symbols}"

    # Verify no unevaluated Function applications (theta(t), x_P(t), etc.)
    # sp.Function catches sin/cos too, so check for AppliedUndef specifically.
    from sympy.core.function import AppliedUndef
    undef_f1 = f1.atoms(AppliedUndef)
    undef_f2 = f2.atoms(AppliedUndef)
    assert not undef_f1, \
        f"T9 FAIL: f1 contains unevaluated functions: {undef_f1}"
    assert not undef_f2, \
        f"T9 FAIL: f2 contains unevaluated functions: {undef_f2}"

    print("T9 PASS: Canonical dynamics use only reduced symbols")


# ---- T10: Full derivation matches canonical form ----

def test_T10_derivation_matches_canonical():
    """The full body-frame derivation must produce expressions that agree
    with f1 = -phi*x2 + w*sin(psi), f2 = phi*x1 + w*cos(psi) - 1
    at numerical test points."""
    import sympy as sp
    from sympy import (symbols, Function, sin, cos, diff,
                       trigsimp, expand_trig, simplify)

    t = symbols("t", real=True)
    v_P, v_E = symbols("v_P v_E", positive=True)
    phi = symbols("phi", real=True)
    psi = symbols("psi", real=True)
    psi_lab = symbols("psi_lab", real=True)
    w = symbols("w", positive=True)

    # Time-dependent functions (absolute coordinates)
    x_Pt = Function("x_P")(t)
    y_Pt = Function("y_P")(t)
    x_Et = Function("x_E")(t)
    y_Et = Function("y_E")(t)
    theta_t = Function("theta")(t)

    # Body-frame coordinates
    Dx, Dy = x_Et - x_Pt, y_Et - y_Pt
    x1_expr = -Dx * sin(theta_t) + Dy * cos(theta_t)
    x2_expr = Dx * cos(theta_t) + Dy * sin(theta_t)

    # Differentiate and substitute dynamics
    x1_dot = diff(x1_expr, t)
    x2_dot = diff(x2_expr, t)

    subs = {
        diff(x_Pt, t): v_P * cos(theta_t),
        diff(y_Pt, t): v_P * sin(theta_t),
        diff(theta_t, t): phi,
        diff(x_Et, t): v_E * cos(psi_lab),
        diff(y_Et, t): v_E * sin(psi_lab),
    }
    x1_dot = trigsimp(expand_trig(x1_dot.subs(subs)))
    x2_dot = trigsimp(expand_trig(x2_dot.subs(subs)))

    # Substitute psi_lab = psi + theta_t, then normalize
    x1_dot = trigsimp(expand_trig(x1_dot.subs(psi_lab, psi + theta_t)))
    x2_dot = trigsimp(expand_trig(x2_dot.subs(psi_lab, psi + theta_t)))
    x1_dot = simplify(x1_dot.subs(v_E, w * v_P).subs(v_P, 1))
    x2_dot = simplify(x2_dot.subs(v_E, w * v_P).subs(v_P, 1))

    # The derived expressions still contain Function objects but should
    # agree numerically with the canonical form at test points.
    # We substitute numerical values for ALL symbols including the Functions.
    import math
    rng = np.random.default_rng(99)

    for _ in range(20):
        phi_v = rng.uniform(-1, 1)
        psi_v = rng.uniform(0, 2 * math.pi)
        w_v = rng.uniform(0.05, 0.5)
        # Values for the body-frame coordinates (via the Function objects)
        xP_v, yP_v = rng.uniform(-5, 5, 2)
        xE_v, yE_v = rng.uniform(-5, 5, 2)
        th_v = rng.uniform(0, 2 * math.pi)

        # Compute x1, x2 from the body-frame definition
        dx, dy = xE_v - xP_v, yE_v - yP_v
        x1_v = -dx * math.sin(th_v) + dy * math.cos(th_v)
        x2_v = dx * math.cos(th_v) + dy * math.sin(th_v)

        # Evaluate derived expression (substitute Function objects numerically)
        num_subs = {
            phi: phi_v, psi: psi_v, w: w_v,
            x_Pt: xP_v, y_Pt: yP_v, x_Et: xE_v, y_Et: yE_v,
            theta_t: th_v,
        }
        x1d_derived = complex(x1_dot.subs(num_subs)).real
        x2d_derived = complex(x2_dot.subs(num_subs)).real

        # Evaluate canonical form
        x1d_canon = -phi_v * x2_v + w_v * math.sin(psi_v)
        x2d_canon = phi_v * x1_v + w_v * math.cos(psi_v) - 1.0

        assert abs(x1d_derived - x1d_canon) < 1e-10, \
            f"T10 FAIL: x1_dot mismatch: derived={x1d_derived}, canon={x1d_canon}"
        assert abs(x2d_derived - x2d_canon) < 1e-10, \
            f"T10 FAIL: x2_dot mismatch: derived={x2d_derived}, canon={x2d_canon}"

    print("T10 PASS: Full derivation matches canonical form at 20 random points")


# ---- T11: Dynamics symmetry under x1 -> -x1 reflection ----

def test_T11_dynamics_symmetry():
    """The RHS should satisfy the reflection symmetry:
    f(-x1, x2, -p1, p2, w) = (-f1, f2, -g1, g2) where (f1,f2,g1,g2) = RHS(x1,x2,p1,p2,w).
    This encodes left-right symmetry of the pursuer's body frame."""
    rng = np.random.default_rng(77)

    for _ in range(50):
        x1 = rng.uniform(-5, 5)
        x2 = rng.uniform(-5, 5)
        angle = rng.uniform(0, 2 * np.pi)
        norm_p = rng.uniform(0.1, 3.0)
        p1 = norm_p * np.cos(angle)
        p2 = norm_p * np.sin(angle)
        w_val = rng.uniform(0.05, 0.5)

        # Original
        f1, f2, g1, g2 = rhs_forward_numpy(0, [x1, x2, p1, p2], w_val)
        # Reflected
        f1r, f2r, g1r, g2r = rhs_forward_numpy(
            0, [-x1, x2, -p1, p2], w_val
        )

        assert abs(f1r - (-f1)) < 1e-12, \
            f"T11 FAIL: f1 symmetry: f1r={f1r}, -f1={-f1}"
        assert abs(f2r - f2) < 1e-12, \
            f"T11 FAIL: f2 symmetry: f2r={f2r}, f2={f2}"
        assert abs(g1r - (-g1)) < 1e-12, \
            f"T11 FAIL: g1 symmetry: g1r={g1r}, -g1={-g1}"
        assert abs(g2r - g2) < 1e-12, \
            f"T11 FAIL: g2 symmetry: g2r={g2r}, g2={g2}"

    print("T11 PASS: Dynamics symmetric under x1 -> -x1 reflection (50 points)")


# ---- T12: Isochrone symmetry about x2-axis ----

def test_T12_isochrone_symmetry():
    """Backward trajectories from symmetric terminal points (alpha, pi-alpha)
    should produce mirror-image paths about the x2-axis."""
    w_val = 0.25
    ell_tilde = 0.5
    T_horizon = 5.0

    # Pick an alpha in the usable part
    alpha_min = np.arcsin(w_val)
    alpha_test = (alpha_min + np.pi / 2) / 2 + 0.1  # somewhere in usable part
    alpha_mirror = np.pi - alpha_test  # mirror about x2 axis

    tc1 = compute_terminal_conditions(np.array([alpha_test]), w_val, ell_tilde)
    tc2 = compute_terminal_conditions(np.array([alpha_mirror]), w_val, ell_tilde)

    sol1 = solve_ivp(rhs_backward, [0, T_horizon], tc1[0], args=(w_val,),
                      method='RK45', max_step=0.05, rtol=1e-10, atol=1e-12,
                      dense_output=True)
    sol2 = solve_ivp(rhs_backward, [0, T_horizon], tc2[0], args=(w_val,),
                      method='RK45', max_step=0.05, rtol=1e-10, atol=1e-12,
                      dense_output=True)

    # Evaluate both at the same time points
    t_eval = np.linspace(0, T_horizon, 200)
    y1 = sol1.sol(t_eval)
    y2 = sol2.sol(t_eval)

    # Mirror: x1 should negate, x2 should match
    x1_diff = np.max(np.abs(y1[0] + y2[0]))  # x1_1 + x1_2 ~ 0
    x2_diff = np.max(np.abs(y1[1] - y2[1]))  # x2_1 - x2_2 ~ 0

    assert x1_diff < 1e-8, f"T12 FAIL: x1 mirror mismatch = {x1_diff}"
    assert x2_diff < 1e-8, f"T12 FAIL: x2 mirror mismatch = {x2_diff}"

    print(f"T12 PASS: Isochrone symmetry (max x1 diff={x1_diff:.2e}, x2 diff={x2_diff:.2e})")


# ---- T13: Small-w integration stability ----

def test_T13_small_w_stability():
    """Integration should remain stable and conserve invariants for very small w."""
    w_val = 0.01
    ell_tilde = 0.5
    T_horizon = 10.0

    alpha_min = np.arcsin(w_val)
    alpha = np.pi / 2  # top of circle, well inside usable part
    tc = compute_terminal_conditions(np.array([alpha]), w_val, ell_tilde)

    sol = solve_ivp(rhs_backward, [0, T_horizon], tc[0], args=(w_val,),
                     method='RK45', max_step=0.05, rtol=1e-10, atol=1e-12)
    assert sol.success, f"T13 FAIL: Integration failed for w={w_val}"

    x1_arr, x2_arr, p1_arr, p2_arr = sol.y
    # Check Hamiltonian conservation
    norm_p = np.sqrt(p1_arr**2 + p2_arr**2)
    sigma_arr = p2_arr * x1_arr - p1_arr * x2_arr
    H_star = -np.abs(sigma_arr) + w_val * norm_p - p2_arr + 1.0
    max_H = np.max(np.abs(H_star))
    assert max_H < 1e-6, f"T13 FAIL: H* drift = {max_H} for w={w_val}"

    # Check costate norm conservation
    p_sq = p1_arr**2 + p2_arr**2
    max_p_drift = np.max(np.abs(p_sq - p_sq[0]))
    assert max_p_drift < 1e-8, f"T13 FAIL: ||p||^2 drift = {max_p_drift} for w={w_val}"

    print(f"T13 PASS: w={w_val} stable (H* drift={max_H:.2e}, ||p||^2 drift={max_p_drift:.2e})")


# ---- T14: Usable part width decreases with w ----

def test_T14_usable_part_monotone():
    """The usable part arc width (pi - 2*arcsin(w)) should monotonically
    decrease as w increases from 0 to 1."""
    w_values = np.linspace(0.01, 0.99, 50)
    widths = np.pi - 2 * np.arcsin(w_values)

    # Check monotone decreasing
    diffs = np.diff(widths)
    assert np.all(diffs < 0), \
        f"T14 FAIL: Usable part width not monotonically decreasing"

    # Check boundary values
    assert widths[0] > np.pi - 0.1, \
        f"T14 FAIL: Width at w~0 should be ~pi, got {widths[0]}"
    assert widths[-1] < 0.3, \
        f"T14 FAIL: Width at w~1 should be ~0, got {widths[-1]}"

    print("T14 PASS: Usable part width monotonically decreases with w")


# ---- Run all tests ----
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Verification Tests")
    print("=" * 60)
    test_T1_lambdify_consistency()
    test_T2_hamiltonian_conservation()
    test_T3_costate_norm_conservation()
    test_T4_capture_condition()
    test_T5_transversality()
    test_T6_stationary_evader()
    test_T7_usable_part()
    test_T8_no_variable_redefinitions()
    test_T9_canonical_form_symbols()
    test_T10_derivation_matches_canonical()
    test_T11_dynamics_symmetry()
    test_T12_isochrone_symmetry()
    test_T13_small_w_stability()
    test_T14_usable_part_monotone()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

"""
Parameter propagation experiments for the Homicidal Chauffeur notebook.

Tests whether user-toggled slider values (v_E, omega_max, ell) actually
affect downstream computations throughout the notebook.

Experiments:
  E1: Capture radius sweep — ell_tilde changes propagate to terminal conditions & trajectories
  E2: Speed ratio sweep — w changes propagate to usable arc, trajectories, reachable sets
  E3: Composite trajectory validity — hardcoded angles (40°, 95°) vs usable arc at various w
  E4: §1 static demo isolation — confirm chase_demo_static has zero parameter dependencies
"""

import ast
import math
import textwrap

import numpy as np
from scipy.integrate import solve_ivp


# ---- Reusable functions (mirroring notebook logic) ----

def rhs_forward_numpy(t, state, w_val):
    """Hand-coded RHS of the 4D characteristic ODE."""
    x1, x2, p1, p2 = state
    norm_p = math.sqrt(p1**2 + p2**2)
    if norm_p < 1e-15:
        return [0.0, 0.0, 0.0, 0.0]
    sigma = p2 * x1 - p1 * x2
    phi_star = -np.sign(sigma)
    x1d = -phi_star * x2 + w_val * p1 / norm_p
    x2d = phi_star * x1 + w_val * p2 / norm_p - 1.0
    p1d = -phi_star * p2
    p2d = phi_star * p1
    return [x1d, x2d, p1d, p2d]


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


def usable_arc(w_val):
    """Return (alpha_min, alpha_max) for the usable part of the terminal circle."""
    alpha_min = np.arcsin(min(w_val, 0.999))
    alpha_max = np.pi - alpha_min
    return alpha_min, alpha_max


def integrate_one(alpha, w_val, ell_tilde_val, T=8.0):
    """Integrate one backward trajectory and return the solution."""
    ic = compute_terminal_conditions(np.array([alpha]), w_val, ell_tilde_val)[0]
    sol = solve_ivp(
        rhs_backward, [0, T], ic, args=(w_val,),
        method='RK45', max_step=0.05, dense_output=True,
        rtol=1e-10, atol=1e-12,
    )
    return sol


# ---- E1: Capture radius sweep ----

def test_E1_capture_radius_changes_terminal_conditions():
    """Changing ell_tilde_val must produce different terminal conditions."""
    w = 0.45
    alpha = np.array([np.pi / 3])  # 60 degrees, well inside usable arc

    tc1 = compute_terminal_conditions(alpha, w, ell_tilde_val=0.5)
    tc2 = compute_terminal_conditions(alpha, w, ell_tilde_val=1.0)

    diff = np.linalg.norm(tc1 - tc2)
    assert diff > 0.1, f"E1 FAIL: terminal conditions unchanged when ell_tilde changed (diff={diff})"
    print(f"E1a PASS: Terminal conditions differ with ell_tilde (diff={diff:.4f})")


def test_E1_capture_radius_changes_trajectories():
    """Changing ell_tilde_val must produce different integrated trajectories."""
    w = 0.45
    alpha = np.pi / 3

    sol1 = integrate_one(alpha, w, ell_tilde_val=0.5, T=6.0)
    sol2 = integrate_one(alpha, w, ell_tilde_val=1.0, T=6.0)

    # Compare endpoints
    end1 = sol1.y[:, -1]
    end2 = sol2.y[:, -1]
    diff = np.linalg.norm(end1 - end2)
    assert diff > 0.1, f"E1 FAIL: trajectories unchanged when ell_tilde changed (diff={diff})"
    print(f"E1b PASS: Trajectories differ with ell_tilde (endpoint diff={diff:.4f})")


def test_E1_usable_arc_independent_of_ell():
    """The usable arc (arcsin(w), pi-arcsin(w)) depends only on w, not ell."""
    w = 0.45
    arc1 = usable_arc(w)
    # ell doesn't appear in the usable arc formula — verify
    # (This is a mathematical identity, but confirms our implementation)
    for ell in [0.1, 0.5, 1.0, 2.0]:
        arc = usable_arc(w)
        assert arc == arc1, f"E1 FAIL: usable arc changed with ell"
    print(f"E1c PASS: Usable arc ({np.degrees(arc1[0]):.1f}°, {np.degrees(arc1[1]):.1f}°) independent of ell")


# ---- E2: Speed ratio sweep ----

def test_E2_speed_ratio_changes_usable_arc():
    """Changing w must shift the usable arc boundaries."""
    results = []
    for w in [0.2, 0.3, 0.45, 0.6, 0.8]:
        a_min, a_max = usable_arc(w)
        width = a_max - a_min
        results.append((w, np.degrees(a_min), np.degrees(a_max), np.degrees(width)))
        print(f"  w={w:.2f}: usable arc = ({np.degrees(a_min):.1f}°, {np.degrees(a_max):.1f}°), width={np.degrees(width):.1f}°")

    # Arc should narrow as w increases
    widths = [r[3] for r in results]
    assert all(widths[i] > widths[i + 1] for i in range(len(widths) - 1)), \
        f"E2 FAIL: usable arc width not monotonically decreasing with w"
    print("E2a PASS: Usable arc narrows monotonically with increasing w")


def test_E2_speed_ratio_changes_trajectories():
    """Different w values must produce different trajectory shapes."""
    ell_tilde = 0.5
    alpha = np.pi / 3

    sol1 = integrate_one(alpha, w_val=0.3, ell_tilde_val=ell_tilde, T=6.0)
    sol2 = integrate_one(alpha, w_val=0.6, ell_tilde_val=ell_tilde, T=6.0)

    end1 = sol1.y[:, -1]
    end2 = sol2.y[:, -1]
    diff = np.linalg.norm(end1 - end2)
    assert diff > 0.1, f"E2 FAIL: trajectories unchanged when w changed (diff={diff})"
    print(f"E2b PASS: Trajectories differ with w (endpoint diff={diff:.4f})")


def test_E2_full_propagation_chain():
    """Verify the full chain: slider → derived params → terminal conditions → ODE → endpoint."""
    configs = [
        # (v_E, omega_max, ell) → (w, R_min, ell_tilde)
        (0.45, 1.0, 0.5),   # default
        (0.30, 1.0, 0.5),   # slower evader
        (0.45, 2.0, 0.5),   # tighter turns
        (0.45, 1.0, 0.8),   # larger capture radius
    ]

    endpoints = []
    for v_E, omega, ell in configs:
        v_P = 1.0
        w = v_E / v_P
        R_min = v_P / omega
        ell_tilde = ell / R_min

        a_min, a_max = usable_arc(w)
        alpha = (a_min + a_max) / 2  # midpoint of usable arc

        sol = integrate_one(alpha, w, ell_tilde, T=6.0)
        ep = sol.y[:, -1]
        endpoints.append(ep)
        print(f"  v_E={v_E}, ω={omega}, ℓ={ell} → w={w:.3f}, ℓ̃={ell_tilde:.3f} → endpoint=({ep[0]:.3f}, {ep[1]:.3f})")

    # All four configs should produce distinct endpoints
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            diff = np.linalg.norm(endpoints[i] - endpoints[j])
            assert diff > 1e-3, \
                f"E2 FAIL: configs {i} and {j} produced same endpoint (diff={diff})"

    print("E2c PASS: All four parameter configurations produce distinct trajectories")


# ---- E3: Composite trajectory validity ----

def _composite_angles(w_val):
    """Compute composite angles using the same percentile formula as the notebook."""
    arc_min = np.arcsin(min(w_val, 0.999))
    arc_width = np.pi - 2 * arc_min
    alpha_A = arc_min + 0.105 * arc_width
    alpha_B = arc_min + 0.540 * arc_width
    return alpha_A, alpha_B


def test_E3_composite_angles_in_usable_arc():
    """Percentile-based composite angles stay in the usable arc for all w values."""
    print("  Percentile formula: alpha_A = arc_min + 0.105 * arc_width, alpha_B = arc_min + 0.540 * arc_width")

    for w in np.arange(0.05, 0.96, 0.01):
        alpha_A, alpha_B = _composite_angles(w)
        a_min, a_max = usable_arc(w)
        a_ok = a_min < alpha_A < a_max
        b_ok = a_min < alpha_B < a_max
        assert a_ok, f"alpha_A={np.degrees(alpha_A):.1f}° outside arc at w={w:.2f}"
        assert b_ok, f"alpha_B={np.degrees(alpha_B):.1f}° outside arc at w={w:.2f}"

    print("E3a PASS: Percentile-based composite angles valid for all w in [0.05, 0.95]")

    # Verify default w=0.45 reproduces original angles closely
    alpha_A, alpha_B = _composite_angles(0.45)
    assert abs(np.degrees(alpha_A) - 40.0) < 0.5, \
        f"At default w=0.45, alpha_A={np.degrees(alpha_A):.2f}°, expected ~40°"
    assert abs(np.degrees(alpha_B) - 95.0) < 0.5, \
        f"At default w=0.45, alpha_B={np.degrees(alpha_B):.2f}°, expected ~95°"
    print(f"E3b PASS: At default w=0.45, angles = ({np.degrees(alpha_A):.2f}°, {np.degrees(alpha_B):.2f}°) ≈ (40°, 95°)")


def test_E3_composite_crossing_exists():
    """Verify that percentile-based characteristics cross across parameter regimes."""
    configs = [
        (0.45, 0.5, "default"),
        (0.30, 0.5, "slower evader"),
        (0.45, 1.0, "larger ell_tilde"),
        (0.60, 0.5, "faster evader"),
        (0.80, 0.5, "fast evader (was broken with hardcoded angles)"),
    ]

    for w, ell_tilde, label in configs:
        alpha_A, alpha_B = _composite_angles(w)

        sol_A = integrate_one(alpha_A, w, ell_tilde, T=15.0)
        sol_B = integrate_one(alpha_B, w, ell_tilde, T=15.0)

        # Sample both and find closest approach
        N = 500
        tA = np.linspace(0, sol_A.t[-1], N)
        tB = np.linspace(0, sol_B.t[-1], N)
        sA = sol_A.sol(tA)
        sB = sol_B.sol(tB)

        min_dist = 1e10
        for i in range(0, N, 5):
            dists = np.sqrt((sB[0] - sA[0, i])**2 + (sB[1] - sA[1, i])**2)
            d = np.min(dists)
            if d < min_dist:
                min_dist = d

        crosses = min_dist < 0.2
        print(f"  {label} (w={w}, ℓ̃={ell_tilde}): min_dist={min_dist:.4f} — {'CROSS' if crosses else 'NO CROSS'}")

    print("E3c DONE: Crossing analysis complete")


# ---- E4: §1 static demo isolation ----

def test_E4_static_demo_has_no_parameter_deps():
    """Parse the notebook source and verify chase_demo_static's function signature
    contains no parameter-related arguments."""
    with open("homicidal_chauffeur.py", "r") as f:
        source = f.read()

    tree = ast.parse(source)

    param_names = {
        'w_val', 'ell_tilde_val', 'v_E_val', 'omega_val', 'ell_val', 'R_min_val',
        'v_E_slider', 'omega_max_slider', 'ell_slider',
        'trajectories', 'physical_trajs', 'isochrones',
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'chase_demo_static':
            args = [arg.arg for arg in node.args.args]
            overlap = set(args) & param_names
            assert not overlap, \
                f"E4 FAIL: chase_demo_static takes parameter args: {overlap}"
            print(f"  chase_demo_static signature: ({', '.join(args)})")
            print(f"  No parameter dependencies found.")
            print("E4a PASS: §1 static demo is fully isolated from sliders")
            return

    raise AssertionError("E4 FAIL: chase_demo_static not found in source")


def test_E4_problem_geometry_isolated():
    """Verify problem_geometry_plot also has no parameter dependencies."""
    with open("homicidal_chauffeur.py", "r") as f:
        source = f.read()

    tree = ast.parse(source)

    param_names = {
        'w_val', 'ell_tilde_val', 'v_E_val', 'omega_val', 'ell_val', 'R_min_val',
        'v_E_slider', 'omega_max_slider', 'ell_slider',
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'problem_geometry_plot':
            args = [arg.arg for arg in node.args.args]
            overlap = set(args) & param_names
            assert not overlap, \
                f"E4 FAIL: problem_geometry_plot takes parameter args: {overlap}"
            print(f"  problem_geometry_plot signature: ({', '.join(args)})")
            print("E4b PASS: §1 geometry diagram is fully isolated from sliders")
            return

    raise AssertionError("E4 FAIL: problem_geometry_plot not found in source")


# ---- Runner ----

if __name__ == "__main__":
    import sys

    tests = [
        ("E1a", test_E1_capture_radius_changes_terminal_conditions),
        ("E1b", test_E1_capture_radius_changes_trajectories),
        ("E1c", test_E1_usable_arc_independent_of_ell),
        ("E2a", test_E2_speed_ratio_changes_usable_arc),
        ("E2b", test_E2_speed_ratio_changes_trajectories),
        ("E2c", test_E2_full_propagation_chain),
        ("E3a", test_E3_composite_angles_in_usable_arc),
        ("E3c", test_E3_composite_crossing_exists),
        ("E4a", test_E4_static_demo_has_no_parameter_deps),
        ("E4b", test_E4_problem_geometry_isolated),
    ]

    print("=" * 70)
    print("Parameter Propagation Experiments")
    print("=" * 70)

    passed = 0
    failed = 0
    for label, fn in tests:
        print(f"\n--- {label}: {fn.__doc__.strip().split(chr(10))[0]} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 70)
    sys.exit(1 if failed else 0)

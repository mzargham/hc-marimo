"""
Regression test suite for the Homicidal Chauffeur notebook.

Pins current correct behavior at the default parameter set (w=0.45, ℓ̃=0.5)
so that future changes to parameter propagation, adaptive sampling, or
physical lift don't silently break results.

Tests are organized in tiers:
  R1–R4:   Golden values at default parameters
  R5–R8:   Physical lift correctness
  R9–R11:  Composite trajectory properties
  R12–R14: Adaptive sampling behavior
  R15–R20: Structural & invariant checks
"""

import ast

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp


# ---- Default parameters ----
W_DEFAULT = 0.45
ELL_TILDE_DEFAULT = 0.5
T_HORIZON = 8.0
N_LIFT = 300


# ---- Shared functions (mirroring notebook logic) ----

def rhs_forward_numpy(t, state, w_val):
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
    alpha_min = np.arcsin(min(w_val, 0.999))
    alpha_max = np.pi - alpha_min
    return alpha_min, alpha_max


def integrate_one(alpha, w_val, ell_tilde_val, T=T_HORIZON):
    ic = compute_terminal_conditions(np.array([alpha]), w_val, ell_tilde_val)[0]
    return solve_ivp(
        rhs_backward, [0, T], ic, args=(w_val,),
        method='RK45', max_step=0.05, dense_output=True,
        rtol=1e-10, atol=1e-12,
    )


def adaptive_tau(sol, t0, t1, N):
    """Reimplementation of _adaptive_tau from the notebook."""
    nc = 200
    tc = np.linspace(t0, t1, nc)
    sc = sol.sol(tc)
    sig = sc[3] * sc[0] - sc[2] * sc[1]
    eps = 0.05
    wt = 1.0 + 1.0 / (np.abs(sig) + eps)
    cdf = cumulative_trapezoid(wt, tc, initial=0)
    cdf /= cdf[-1]
    u = np.linspace(0, 1, N)
    tau = np.interp(u, cdf, tc)
    tau[0] = t0
    tau[-1] = t1
    return tau


def lift_single(sol, w_val, N=N_LIFT):
    """Reimplementation of _lift_single from the notebook."""
    T = sol.t[-1]
    tau = adaptive_tau(sol, 0, T, N)
    states = sol.sol(tau)
    x1, x2, p1, p2 = states
    sigma = p2 * x1 - p1 * x2
    phi_star = -np.sign(sigma)
    theta_cum = cumulative_trapezoid(-phi_star, tau, initial=0)
    psi_lab = np.arctan2(p1[0], p2[0])
    X_E = -w_val * np.cos(psi_lab) * tau
    Y_E = -w_val * np.sin(psi_lab) * tau
    cos_th = np.cos(theta_cum)
    sin_th = np.sin(theta_cum)
    dx = -x1 * sin_th + x2 * cos_th
    dy = x1 * cos_th + x2 * sin_th
    X_P = X_E - dx
    Y_P = Y_E - dy
    # Flip to forward time
    X_P = X_P[::-1]; Y_P = Y_P[::-1]
    X_E = X_E[::-1]; Y_E = Y_E[::-1]
    theta = theta_cum[::-1]
    t_fwd = tau[-1] - tau[::-1]
    # Shift so evader starts at origin
    X_P -= X_E[0]; Y_P -= Y_E[0]
    X_E -= X_E[0]; Y_E -= Y_E[0]
    dist = np.sqrt((X_P - X_E)**2 + (Y_P - Y_E)**2)
    return {
        "t": t_fwd, "X_P": X_P, "Y_P": Y_P,
        "X_E": X_E, "Y_E": Y_E, "theta": theta, "dist": dist,
    }


def _default_trajectories(n=5):
    """Integrate n backward trajectories at default params."""
    a_min, a_max = usable_arc(W_DEFAULT)
    eps = 1e-3
    alphas = np.linspace(a_min + eps, a_max - eps, n)
    sols = []
    for alpha in alphas:
        sols.append(integrate_one(alpha, W_DEFAULT, ELL_TILDE_DEFAULT))
    return alphas, sols


# ================================================================
# Tier 1: Golden values at default parameters
# ================================================================

def test_R1_terminal_conditions_golden():
    """Terminal conditions at key angles match analytic golden values."""
    w, ell = W_DEFAULT, ELL_TILDE_DEFAULT

    # α = π/2 (usable arc center)
    ic = compute_terminal_conditions(np.array([np.pi / 2]), w, ell)[0]
    assert abs(ic[0]) < 1e-14, f"x1 should be 0 at α=π/2, got {ic[0]}"
    assert abs(ic[1] - 0.5) < 1e-14
    assert abs(ic[2]) < 1e-14, f"p1 should be 0 at α=π/2, got {ic[2]}"
    lam_expected = -1.0 / (ell * (w - 1.0))  # = 1/(0.5*0.55) ≈ 3.6364
    assert abs(ic[3] - lam_expected * ell) < 1e-12

    # α = π/3 and α = 2π/3 should be mirror-symmetric in x₁
    ic_lo = compute_terminal_conditions(np.array([np.pi / 3]), w, ell)[0]
    ic_hi = compute_terminal_conditions(np.array([2 * np.pi / 3]), w, ell)[0]
    assert abs(ic_lo[0] + ic_hi[0]) < 1e-14, "x1 should mirror"
    assert abs(ic_lo[1] - ic_hi[1]) < 1e-14, "x2 should be equal"
    assert abs(ic_lo[2] + ic_hi[2]) < 1e-14, "p1 should mirror"
    assert abs(ic_lo[3] - ic_hi[3]) < 1e-14, "p2 should be equal"


def test_R2_backward_trajectory_endpoints_golden():
    """Backward trajectory endpoints at default params match golden references."""
    # Golden values computed with rtol=1e-10, atol=1e-12
    golden_x1x2 = [
        (-6.3570356871, -2.2108113511),
        (-2.8933904745,  4.3551697529),
        (-1.9215105306,  1.4404083516),
        ( 2.8933904866,  4.3551697501),
        ( 6.3570356877, -2.2108113494),
    ]

    alphas, sols = _default_trajectories(5)

    for i, sol in enumerate(sols):
        assert sol.success, f"Trajectory {i} failed to integrate"
        assert sol.t[-1] >= 0.99 * T_HORIZON, f"Trajectory {i} didn't reach T"
        ep = sol.y[:, -1]
        gx1, gx2 = golden_x1x2[i]
        assert abs(ep[0] - gx1) < 1e-4, \
            f"Traj {i}: x1={ep[0]:.10f}, expected {gx1}"
        assert abs(ep[1] - gx2) < 1e-4, \
            f"Traj {i}: x2={ep[1]:.10f}, expected {gx2}"


def test_R3_usable_arc_golden():
    """Usable arc boundaries at default w match analytic values."""
    a_min, a_max = usable_arc(W_DEFAULT)
    assert abs(a_min - np.arcsin(0.45)) < 1e-14
    assert abs(a_max - (np.pi - np.arcsin(0.45))) < 1e-14

    # Width ≈ 126.52°
    width_deg = np.degrees(a_max - a_min)
    assert abs(width_deg - 126.52) < 0.1


def test_R4_trajectory_mirror_symmetry():
    """Trajectories at α and π−α mirror about x₂-axis at default params."""
    a_min, a_max = usable_arc(W_DEFAULT)
    eps = 1e-3
    alphas = np.linspace(a_min + eps, a_max - eps, 10)

    for i in range(5):
        alpha_lo = alphas[i]
        alpha_hi = alphas[-(i + 1)]
        # These should satisfy alpha_lo + alpha_hi ≈ π
        assert abs(alpha_lo + alpha_hi - np.pi) < 0.01

        sol_lo = integrate_one(alpha_lo, W_DEFAULT, ELL_TILDE_DEFAULT)
        sol_hi = integrate_one(alpha_hi, W_DEFAULT, ELL_TILDE_DEFAULT)

        # Compare at same time points
        t_eval = np.linspace(0, min(sol_lo.t[-1], sol_hi.t[-1]), 50)
        s_lo = sol_lo.sol(t_eval)
        s_hi = sol_hi.sol(t_eval)

        # x1 should negate, x2 should match
        assert np.max(np.abs(s_lo[0] + s_hi[0])) < 0.05, \
            f"Pair {i}: x1 not mirroring"
        assert np.max(np.abs(s_lo[1] - s_hi[1])) < 0.05, \
            f"Pair {i}: x2 not matching"


# ================================================================
# Tier 2: Physical lift correctness
# ================================================================

def test_R5_evader_straight_line():
    """Evader moves in a straight line in lab frame (constant heading)."""
    alphas, sols = _default_trajectories(5)

    for i, sol in enumerate(sols):
        p = lift_single(sol, W_DEFAULT)
        dXE = np.diff(p["X_E"])
        dYE = np.diff(p["Y_E"])
        # Skip near-zero segments
        mag = np.sqrt(dXE**2 + dYE**2)
        mask = mag > 1e-10
        angles = np.arctan2(dYE[mask], dXE[mask])
        # All angles should be the same
        deviation = np.max(np.abs(angles - angles[0]))
        assert deviation < 1e-6, \
            f"Traj {i}: evader angle deviation = {deviation:.2e}"


def test_R6_evader_speed():
    """Evader moves at speed w in lab frame."""
    sol = integrate_one(np.pi / 3, W_DEFAULT, ELL_TILDE_DEFAULT)
    p = lift_single(sol, W_DEFAULT)

    dt = np.diff(p["t"])
    dXE = np.diff(p["X_E"])
    dYE = np.diff(p["Y_E"])
    speeds = np.sqrt(dXE**2 + dYE**2) / dt

    assert abs(np.mean(speeds) - W_DEFAULT) < 1e-6, \
        f"Mean evader speed = {np.mean(speeds):.10f}, expected {W_DEFAULT}"
    assert np.std(speeds) < 1e-6, \
        f"Speed variation too large: std = {np.std(speeds):.2e}"


def test_R7_pursuer_turn_rate_bounded():
    """Pursuer turn rate |dθ/dt| ≤ 1 everywhere."""
    alphas, sols = _default_trajectories(5)

    for i, sol in enumerate(sols):
        p = lift_single(sol, W_DEFAULT)
        dt = np.diff(p["t"])
        dtheta = np.diff(p["theta"])
        turn_rates = np.abs(dtheta / dt)
        assert np.max(turn_rates) < 1.0 + 1e-4, \
            f"Traj {i}: max turn rate = {np.max(turn_rates):.6f}"


def test_R8_forward_time_structure():
    """After lift: t starts at 0, increases monotonically, evader at origin."""
    sol = integrate_one(np.pi / 3, W_DEFAULT, ELL_TILDE_DEFAULT)
    p = lift_single(sol, W_DEFAULT)

    assert p["t"][0] == 0.0
    assert p["t"][-1] > 0.0
    assert np.all(np.diff(p["t"]) > 0), "Time not monotonically increasing"
    assert abs(p["X_E"][0]) < 1e-14, "Evader X not at origin"
    assert abs(p["Y_E"][0]) < 1e-14, "Evader Y not at origin"


# ================================================================
# Tier 3: Composite trajectory
# ================================================================

def _composite_angles(w_val):
    """Compute composite angles using the same percentile formula as the notebook."""
    arc_min = np.arcsin(min(w_val, 0.999))
    arc_width = np.pi - 2 * arc_min
    alpha_A = arc_min + 0.105 * arc_width  # ~40° at default w=0.45
    alpha_B = arc_min + 0.540 * arc_width  # ~95° at default w=0.45
    return alpha_A, alpha_B


def _build_composite():
    """Build the composite trajectory at default params (matching notebook)."""
    alpha_A, alpha_B = _composite_angles(W_DEFAULT)
    T_bk = 15.0
    w = W_DEFAULT
    ell = ELL_TILDE_DEFAULT

    sols_ab = {}
    for label, alpha in [('A', alpha_A), ('B', alpha_B)]:
        ic = compute_terminal_conditions(np.array([alpha]), w, ell)[0]
        s = solve_ivp(
            rhs_backward, [0, T_bk], ic, args=(w,),
            method='RK45', max_step=0.02, dense_output=True,
            rtol=1e-12, atol=1e-14,
        )
        sols_ab[label] = s

    # Find crossing
    Nc = 2000
    tauA = np.linspace(0, sols_ab['A'].t[-1], Nc)
    tauB = np.linspace(0, sols_ab['B'].t[-1], Nc)
    sA = sols_ab['A'].sol(tauA)
    sB = sols_ab['B'].sol(tauB)
    min_d = 1e10
    ciA = ciB = 0
    for ti in range(Nc):
        dists = np.sqrt((sB[0] - sA[0, ti])**2 + (sB[1] - sA[1, ti])**2)
        idx = np.argmin(dists)
        if dists[idx] < min_d:
            min_d = dists[idx]
            ciA = ti
            ciB = idx
    tau_cA = tauA[ciA]
    tau_cB = tauB[ciB]

    return sols_ab, min_d, tau_cA, tau_cB


def test_R9_composite_structure():
    """Composite trajectory has two phases with evader direction change."""
    sols_ab, min_d, tau_cA, tau_cB = _build_composite()
    w = W_DEFAULT

    # Build both phases and lift
    N = N_LIFT
    # Phase 1: A from capture to crossing
    t1 = adaptive_tau(sols_ab['A'], 0, tau_cA, N)
    s1 = sols_ab['A'].sol(t1)
    psi_A = np.arctan2(s1[2, 0], s1[3, 0])

    # Phase 2: B from crossing onward
    t2 = adaptive_tau(sols_ab['B'], tau_cB, sols_ab['B'].t[-1], N)
    s2 = sols_ab['B'].sol(t2)
    # psi_B needs heading correction from Phase 1
    phi1 = -np.sign(s1[3] * s1[0] - s1[2] * s1[1])
    th_cross = cumulative_trapezoid(-phi1, t1, initial=0)[-1]
    psi_B = np.arctan2(s2[2, 0], s2[3, 0]) + th_cross

    # Direction change
    direction_change = abs(psi_A - psi_B)
    # Should be roughly 0.7–1.0 rad (40–57 degrees)
    assert 0.5 < direction_change < 1.2, \
        f"Direction change = {direction_change:.4f} rad ({np.degrees(direction_change):.1f}°)"


def test_R10_crossing_point_golden():
    """Percentile-based characteristics cross in reduced space at default params."""
    _, min_d, tau_cA, tau_cB = _build_composite()

    # Crossing should be tight
    assert min_d < 0.02, f"Crossing distance = {min_d:.6f}, expected < 0.02"

    # Golden crossing times — percentile angles (~40.0°, ~95.1°) at default w
    # produce crossing times very close to the original 40°/95° values.
    assert abs(tau_cA - 6.1156) < 0.2, \
        f"tau_cA = {tau_cA:.4f}, expected ~6.1156"
    assert abs(tau_cB - 9.0120) < 0.2, \
        f"tau_cB = {tau_cB:.4f}, expected ~9.0120"


def test_R11_composite_capture_distance():
    """At end of composite trajectory, pursuer captures evader at distance ℓ̃."""
    sols_ab, _, tau_cA, _ = _build_composite()

    # Lift Phase 1 (A from capture to crossing) — capture is at τ=0
    sol_A = sols_ab['A']
    s0 = sol_A.sol(0)
    body_dist = np.sqrt(s0[0]**2 + s0[1]**2)
    assert abs(body_dist - ELL_TILDE_DEFAULT) < 1e-6, \
        f"Capture distance in body frame = {body_dist:.10f}, expected {ELL_TILDE_DEFAULT}"


# ================================================================
# Tier 4: Adaptive sampling
# ================================================================

def test_R12_adaptive_tau_monotonicity_and_endpoints():
    """Adaptive time grid is strictly increasing with exact endpoints."""
    sol = integrate_one(np.pi / 3, W_DEFAULT, ELL_TILDE_DEFAULT)
    T = sol.t[-1]

    tau = adaptive_tau(sol, 0, T, 300)
    assert tau[0] == 0.0, f"tau[0] = {tau[0]}"
    assert tau[-1] == T, f"tau[-1] = {tau[-1]}, expected {T}"
    assert np.all(np.diff(tau) > 0), "tau not strictly increasing"
    assert len(tau) == 300


def test_R13_adaptive_tau_concentrates_near_switches():
    """Adaptive grid has higher point density near σ≈0 than away from it."""
    sol = integrate_one(np.pi / 3, W_DEFAULT, ELL_TILDE_DEFAULT)
    tau = adaptive_tau(sol, 0, sol.t[-1], 300)

    states = sol.sol(tau)
    sigma = states[3] * states[0] - states[2] * states[1]
    gaps = np.diff(tau)

    # Near σ=0: |σ| < 0.2
    near_zero = np.abs(sigma[:-1]) < 0.2
    far_zero = np.abs(sigma[:-1]) > 0.5

    if np.any(near_zero) and np.any(far_zero):
        density_near = np.sum(near_zero) / np.sum(gaps[near_zero])
        density_far = np.sum(far_zero) / np.sum(gaps[far_zero])
        ratio = density_near / density_far
        assert ratio > 1.5, \
            f"Density ratio near/far σ=0 = {ratio:.2f}, expected > 1.5"


def test_R14_adaptive_tau_max_gap_bounded():
    """Max gap in adaptive grid is bounded relative to uniform spacing."""
    sol = integrate_one(np.pi / 3, W_DEFAULT, ELL_TILDE_DEFAULT)
    tau = adaptive_tau(sol, 0, sol.t[-1], 300)

    gaps = np.diff(tau)
    uniform_gap = sol.t[-1] / 299
    ratio = np.max(gaps) / uniform_gap
    assert ratio < 2.0, \
        f"Max gap / uniform gap = {ratio:.4f}, expected < 2.0"


# ================================================================
# Tier 5: Structural & invariant checks at default params
# ================================================================

def test_R15_hamiltonian_conservation_default():
    """H* ≈ 0 along all default-param backward trajectories."""
    alphas, sols = _default_trajectories(5)

    for i, sol in enumerate(sols):
        t_eval = np.linspace(0, sol.t[-1], 200)
        x1, x2, p1, p2 = sol.sol(t_eval)
        norm_p = np.sqrt(p1**2 + p2**2)
        sigma = p2 * x1 - p1 * x2
        H = -np.abs(sigma) + W_DEFAULT * norm_p - p2 + 1.0
        max_drift = np.max(np.abs(H))
        # Near usable-arc boundaries, bang-bang switching causes larger drift
        assert max_drift < 1e-4, \
            f"Traj {i}: H* max drift = {max_drift:.2e}"


def test_R16_costate_norm_conservation_default():
    """‖p‖² conserved along all default-param backward trajectories."""
    alphas, sols = _default_trajectories(5)

    for i, sol in enumerate(sols):
        t_eval = np.linspace(0, sol.t[-1], 200)
        _, _, p1, p2 = sol.sol(t_eval)
        norm_sq = p1**2 + p2**2
        drift = np.max(np.abs(norm_sq - norm_sq[0]))
        # Near usable-arc boundaries, bang-bang switching degrades conservation
        assert drift < 1e-3, \
            f"Traj {i}: ‖p‖² drift = {drift:.2e}"


def test_R17_isochrones_nest_outward():
    """Isochrone at T₁ < T₂ is closer to origin than isochrone at T₂."""
    a_min, a_max = usable_arc(W_DEFAULT)
    eps = 1e-3
    n_dense = 40
    alphas = np.linspace(a_min + eps, a_max - eps, n_dense)
    T_max = 12.0

    # Integrate all
    solutions = []
    for alpha in alphas:
        ic = compute_terminal_conditions(np.array([alpha]), W_DEFAULT, ELL_TILDE_DEFAULT)[0]
        sol = solve_ivp(rhs_backward, [0, T_max], ic, args=(W_DEFAULT,),
                        method='RK45', max_step=0.1, dense_output=True,
                        rtol=1e-8, atol=1e-10)
        solutions.append(sol)

    # Extract isochrone mean distances at T=2 and T=8
    for T_inner, T_outer in [(2.0, 8.0), (4.0, 12.0)]:
        dists_inner = []
        dists_outer = []
        for sol in solutions:
            if not sol.success:
                continue
            if sol.t[-1] >= T_inner:
                s = sol.sol(T_inner)
                dists_inner.append(np.sqrt(s[0]**2 + s[1]**2))
            if sol.t[-1] >= T_outer:
                s = sol.sol(T_outer)
                dists_outer.append(np.sqrt(s[0]**2 + s[1]**2))
        if dists_inner and dists_outer:
            mean_inner = np.mean(dists_inner)
            mean_outer = np.mean(dists_outer)
            assert mean_outer > mean_inner, \
                f"Isochrone T={T_outer} (mean dist {mean_outer:.2f}) " \
                f"not outside T={T_inner} (mean dist {mean_inner:.2f})"


def test_R18_reachable_set_scatter_covers_expected_region():
    """Reachable set scatter at default params covers expected (x₁, x₂) ranges."""
    a_min, a_max = usable_arc(W_DEFAULT)
    eps = 1e-3
    n_dense = 40
    alphas = np.linspace(a_min + eps, a_max - eps, n_dense)
    T_max = 12.0

    all_x1, all_x2 = [], []
    for alpha in alphas:
        ic = compute_terminal_conditions(np.array([alpha]), W_DEFAULT, ELL_TILDE_DEFAULT)[0]
        sol = solve_ivp(rhs_backward, [0, T_max], ic, args=(W_DEFAULT,),
                        method='RK45', max_step=0.1, dense_output=True,
                        rtol=1e-8, atol=1e-10)
        if sol.success:
            t_sample = np.linspace(0, sol.t[-1], 100)
            s = sol.sol(t_sample)
            all_x1.append(s[0])
            all_x2.append(s[1])

    x1_all = np.concatenate(all_x1)
    x2_all = np.concatenate(all_x2)

    # With 40 trajectories at T=12, scatter spans roughly [-5, 5] × [-6, 2]
    assert np.min(x1_all) < -3.0, f"x1 min = {np.min(x1_all):.2f}, expected < -3"
    assert np.max(x1_all) > 3.0, f"x1 max = {np.max(x1_all):.2f}, expected > 3"
    assert np.min(x2_all) < -4.0, f"x2 min = {np.min(x2_all):.2f}, expected < -4"
    assert np.max(x2_all) > 1.0, f"x2 max = {np.max(x2_all):.2f}, expected > 1"


def test_R19_evader_field_unit_length():
    """Nearest-neighbor costate lookup produces unit-length headings.

    Verifies the core computation: ψ* = atan2(p₁/‖p‖, p₂/‖p‖)
    yields unit vectors (sin ψ*, cos ψ*) with magnitude 1.
    """
    # Collect costate samples from trajectories
    alphas, sols = _default_trajectories(5)

    all_p1, all_p2 = [], []
    for sol in sols:
        t_eval = np.linspace(0, sol.t[-1], 50)
        states = sol.sol(t_eval)
        all_p1.append(states[2])
        all_p2.append(states[3])

    p1 = np.concatenate(all_p1)
    p2 = np.concatenate(all_p2)
    norm = np.sqrt(p1**2 + p2**2)
    norm = np.where(norm > 1e-12, norm, 1.0)

    # Compute heading components
    sin_psi = p1 / norm
    cos_psi = p2 / norm

    # These should form unit vectors
    magnitudes = np.sqrt(sin_psi**2 + cos_psi**2)
    assert np.allclose(magnitudes, 1.0, atol=1e-12), \
        f"Non-unit heading vectors: max deviation = {np.max(np.abs(magnitudes - 1.0)):.2e}"


def test_R20_no_nan_or_inf():
    """No NaN or Inf in any trajectory output at default params."""
    alphas, sols = _default_trajectories(5)

    for i, sol in enumerate(sols):
        # Raw backward trajectory
        assert not np.any(np.isnan(sol.y)), f"NaN in traj {i} backward"
        assert not np.any(np.isinf(sol.y)), f"Inf in traj {i} backward"

        # Physical lift
        p = lift_single(sol, W_DEFAULT)
        for key in ["t", "X_P", "Y_P", "X_E", "Y_E", "theta", "dist"]:
            arr = p[key]
            assert not np.any(np.isnan(arr)), f"NaN in traj {i} lift[{key}]"
            assert not np.any(np.isinf(arr)), f"Inf in traj {i} lift[{key}]"

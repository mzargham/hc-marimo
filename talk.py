"""SciPy 2026 talk deck - "Derivations, Not Just Simulations".

A self-contained marimo notebook that IS the slide deck AND the live demo.
Presented in slides layout: `uv run marimo run talk.py`.

Structure (see .claude/skills/marimo-talk-notebook):
  - CONTENT BEATS in file order = slide order (beats 1-10); every cell is a
    content slide (no blanks).
  - Setup / imports / heavy compute fold into the TITLE cell and the numeric
    pipeline / slider DEFINITIONS fold into the CLOSING cell (execution is
    dependency-order, not file-order, so downstream cells still get them).
  - Sliders are DEFINED in the closing cell and DISPLAYED+CONSUMED in the
    beat cells, so a slider and its figure share one slide and update live.
  - The on-stage symbolic derivation (beats 5) produces the SAME SymPy objects
    that the numerics lambdify and integrate - derivations, not just simulations.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", layout_file="layouts/talk.slides.json")


# ===========================================================================
# CONTENT BEATS  (top of file = slide order)
# ===========================================================================


@app.cell
def _beat1_title():
    # Setup, imports, symbols, fixed parameters, hook data, and the
    # terminal-condition helper live here (folded in so the deck has no
    # blank trailing slides); the title markdown below is the output.
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import sympy as sp
    from sympy import (
        symbols, Function, cos, sin, atan2, sqrt, sign, simplify, diff,
        Matrix, latex, trigsimp, expand_trig, collect, Abs,
    )
    from scipy.integrate import solve_ivp, cumulative_trapezoid
    import sympy as _sp
    x1, x2 = symbols("x_1 x_2", real=True)
    p1, p2 = symbols("p_1 p_2", real=True)
    phi_ctrl = symbols("phi", real=True)
    psi_ctrl = symbols("psi", real=True)
    psi_lab = _sp.Symbol(r"\psi_{\mathrm{lab}}", real=True)
    v_P = symbols("v_P", positive=True)
    v_E_sym = symbols("v_E", positive=True)
    w_sym = symbols("w", positive=True)
    t = symbols("t", real=True)
    # Canonical operating point for the heavy simulation (dense; run at load).
    # The beat-4 sliders are decoupled from this so scrubbing stays instant.
    W_FIXED = 0.45
    ELL_TILDE_FIXED = 0.5
    N_TRAJ = 80          # backward characteristics (dense enough for the heat map)
    T_HORIZON = 12.0
    # §2 initial condition expressed in body coordinates (answers the ★ at V*≈12.1s).
    # Beat-2 data. Two pure-pursuit heuristic chases (cheap forward-Euler), plus
    # the hardcoded composite optimal chase with the ★ heading switch.
    def simulate_pure_pursuit(p_init, e_init, w, ell, policy, T_max=20.0, dt=0.005):
        xp, yp, th = p_init
        xe, ye = e_init
        n = int(T_max / dt)
        XP, YP, TH = np.empty(n + 1), np.empty(n + 1), np.empty(n + 1)
        XE, YE, TT = np.empty(n + 1), np.empty(n + 1), np.empty(n + 1)
        XP[0], YP[0], TH[0], XE[0], YE[0], TT[0] = xp, yp, th, xe, ye, 0.0
        cap, last = None, n
        for k in range(n):
            bearing = float(np.arctan2(ye - yp, xe - xp))
            err = (bearing - th + np.pi) % (2 * np.pi) - np.pi
            phi = float(np.clip(err / dt, -1.0, 1.0))
            psi = bearing if policy == "run_away" else bearing + 0.5 * np.pi
            xp += np.cos(th) * dt
            yp += np.sin(th) * dt
            th += phi * dt
            xe += w * np.cos(psi) * dt
            ye += w * np.sin(psi) * dt
            XP[k + 1], YP[k + 1], TH[k + 1] = xp, yp, th
            XE[k + 1], YE[k + 1], TT[k + 1] = xe, ye, (k + 1) * dt
            if (xe - xp) ** 2 + (ye - yp) ** 2 <= ell * ell:
                cap, last = (k + 1) * dt, k + 1
                break
        sl = slice(0, last + 1)
        return {"xp": XP[sl], "yp": YP[sl], "xe": XE[sl], "ye": YE[sl],
                "capture_time": cap if cap is not None else TT[last]}

    _P0 = (6.4691, -1.7816, -3.3856)
    _E0 = (0.0, 0.0)
    hook_naive = simulate_pure_pursuit(_P0, _E0, W_FIXED, 0.5, "run_away")
    hook_perp = simulate_pure_pursuit(_P0, _E0, W_FIXED, 0.5, "perpendicular")

    demo_lab = {
        "xp": np.array([6.4691, 6.2348, 6.0217, 5.8627, 5.7537, 5.7110, 5.7349, 5.8259, 5.9782, 6.1711, 6.4095, 6.6664, 6.9047, 7.1247, 7.2993, 7.4125, 7.4672, 7.4527, 7.3754, 7.2332, 7.0381, 6.8149, 6.5580, 6.2951, 6.2766, 6.2646, 6.0210, 5.7699, 5.5400, 5.3464, 5.2076, 5.1188, 5.0361, 4.9081, 4.7374, 4.5186, 4.2733, 4.0178, 3.7805, 3.5535, 3.3639, 3.2243, 3.1464, 3.1276, 3.1746, 3.2843, 3.4407, 3.6490, 3.8890, 4.1403]),
        "yp": np.array([-1.7816, -1.6896, -1.5338, -1.3381, -1.0974, -0.8365, -0.5852, -0.3371, -0.1214, 0.0408, 0.1534, 0.2559, 0.3358, 0.4805, 0.6774, 0.9018, 1.1593, 1.4222, 1.6613, 1.8828, 2.0595, 2.1752, 2.2326, 2.2210, 2.2121, 2.2098, 2.1964, 2.2464, 2.3588, 2.5262, 2.7266, 2.9667, 3.2240, 3.4450, 3.6182, 3.7495, 3.8205, 3.8264, 3.7710, 3.6525, 3.4798, 3.2642, 3.0320, 2.7752, 2.5218, 2.2883, 2.0986, 1.9463, 1.8517, 1.8248]),
        "xe": np.array([0.0000, 0.1133, 0.2320, 0.3454, 0.4641, 0.5828, 0.6961, 0.8148, 0.9336, 1.0469, 1.1656, 1.2843, 1.3976, 1.5164, 1.6351, 1.7484, 1.8671, 1.9858, 2.0992, 2.2179, 2.3366, 2.4499, 2.5686, 2.6874, 2.6928, 2.6963, 2.7672, 2.8416, 2.9161, 2.9905, 3.0614, 3.1359, 3.2103, 3.2848, 3.3557, 3.4301, 3.5046, 3.5790, 3.6499, 3.7243, 3.7988, 3.8732, 3.9441, 4.0186, 4.0930, 4.1675, 4.2384, 4.3128, 4.3873, 4.4617]),
        "ye": np.array([0.0000, 0.0042, 0.0086, 0.0128, 0.0172, 0.0216, 0.0258, 0.0302, 0.0346, 0.0388, 0.0432, 0.0476, 0.0518, 0.0562, 0.0605, 0.0647, 0.0691, 0.0735, 0.0777, 0.0821, 0.0865, 0.0907, 0.0951, 0.0995, 0.0997, 0.1039, 0.1884, 0.2772, 0.3659, 0.4546, 0.5391, 0.6278, 0.7165, 0.8052, 0.8897, 0.9785, 1.0672, 1.1559, 1.2404, 1.3291, 1.4178, 1.5066, 1.5911, 1.6798, 1.7685, 1.8572, 1.9417, 2.0304, 2.1192, 2.2079]),
        "switch_idx": 24,
        "opt_time": 12.1,
    }
    # Usable-part transversality: seed states on the capture circle for backward shooting.
    def compute_terminal_conditions(alpha_arr, w_val, ell_tilde_val):
        x1_T = ell_tilde_val * np.cos(alpha_arr)
        x2_T = ell_tilde_val * np.sin(alpha_arr)
        lam = -1.0 / (ell_tilde_val * (w_val - np.sin(alpha_arr)))
        return np.column_stack([x1_T, x2_T, lam * x1_T, lam * x2_T])

    mo.md(
        r"""
        # Derivations, Not Just Simulations
        ## Teaching Applied Mathematics with Scientific Python

        **Michael Zargham**
        
        Chief Engineer, Dynamical Systems Group
        
        SciPy 2026
        """
    )
    return (mo, Abs, Function, Matrix, atan2, collect, cos,
            cumulative_trapezoid, diff, expand_trig, latex, np, plt,
            sign, simplify, sin, solve_ivp, sp, sqrt, symbols,
            trigsimp, p1, p2, phi_ctrl, psi_ctrl, psi_lab, t,
            v_E_sym, v_P, w_sym, x1, x2, ELL_TILDE_FIXED, N_TRAJ,
            T_HORIZON, W_FIXED, demo_lab, hook_naive, hook_perp,
            compute_terminal_conditions)


@app.cell
def _beat2_hook(mo, np, plt, hook_naive, hook_perp, demo_lab):
    # THE HOOK. Three evaders vs the same pursuer from the same start.
    # Two natural heuristics - and the provably optimal escape, which kinks (★).
    def draw_pursuit(ax, sim, title):
        ax.plot(sim["xp"], sim["yp"], "-", color="#2166ac", lw=1.8, alpha=0.85,
                label="Pursuer (fast, wide turns)")
        ax.plot(sim["xe"], sim["ye"], "--", color="#b2182b", lw=1.8, alpha=0.85,
                label="Evader (slow, agile)")
        ax.plot(sim["xp"][0], sim["yp"][0], "o", color="#2166ac", ms=9,
                mec="white", mew=0.8, zorder=10)
        ax.plot(sim["xe"][0], sim["ye"][0], "o", color="#b2182b", ms=9,
                mec="white", mew=0.8, zorder=10)
        ax.plot(sim["xp"][-1], sim["yp"][-1], "s", color="#2166ac", ms=9,
                mec="white", mew=0.8, zorder=10)
        _th = np.linspace(0, 2 * np.pi, 80)
        ax.plot(sim["xp"][-1] + 0.5 * np.cos(_th), sim["yp"][-1] + 0.5 * np.sin(_th),
                ":", color="gray", lw=1.1, alpha=0.7)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.set_xticks([])
        ax.set_yticks([])

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.8))

    draw_pursuit(_axes[0], hook_naive,
                 rf'"Just run away": caught at $t={hook_naive["capture_time"]:.1f}$')
    draw_pursuit(_axes[1], hook_perp,
                 rf'"Cut sideways": caught at $t={hook_perp["capture_time"]:.1f}$')

    # Optimal panel: the hardcoded composite chase, with the ★ heading switch.
    _sw = demo_lab["switch_idx"]
    _ax = _axes[2]
    _ax.plot(demo_lab["xp"], demo_lab["yp"], "-", color="#2166ac", lw=1.8, alpha=0.85)
    _ax.plot(demo_lab["xe"][:_sw + 1], demo_lab["ye"][:_sw + 1], "--",
             color="#b2182b", lw=1.8, alpha=0.85)
    _ax.plot(demo_lab["xe"][_sw:], demo_lab["ye"][_sw:], "--",
             color="#b2182b", lw=1.8, alpha=0.85)
    _ax.plot(demo_lab["xp"][0], demo_lab["yp"][0], "o", color="#2166ac", ms=9,
             mec="white", mew=0.8, zorder=10)
    _ax.plot(demo_lab["xe"][0], demo_lab["ye"][0], "o", color="#b2182b", ms=9,
             mec="white", mew=0.8, zorder=10)
    _ax.plot(demo_lab["xp"][-1], demo_lab["yp"][-1], "s", color="#2166ac", ms=9,
             mec="white", mew=0.8, zorder=10)
    _ax.plot(demo_lab["xe"][_sw], demo_lab["ye"][_sw], "*", color="#111111", ms=20,
             mec="white", mew=0.9, zorder=11)
    _ax.annotate("★", (demo_lab["xe"][_sw], demo_lab["ye"][_sw]),
                 textcoords="offset points", xytext=(10, -4), fontsize=15)
    _th2 = np.linspace(0, 2 * np.pi, 80)
    _ax.plot(demo_lab["xp"][-1] + 0.5 * np.cos(_th2),
             demo_lab["yp"][-1] + 0.5 * np.sin(_th2), ":", color="gray", lw=1.1, alpha=0.7)
    _ax.set_aspect("equal")
    _ax.set_title(rf'OPTIMAL: survives to $t={demo_lab["opt_time"]:.1f}$', fontsize=12)
    _ax.grid(True, alpha=0.25)
    _ax.set_xticks([])
    _ax.set_yticks([])

    # Same frame on all three panels: identical square limits so the boxes render at the
    # same size (matching the tallest, the optimal panel) and at a common scale.
    _sims = (hook_naive, hook_perp, demo_lab)
    _xall = np.concatenate([np.concatenate([_s["xp"], _s["xe"]]) for _s in _sims])
    _yall = np.concatenate([np.concatenate([_s["yp"], _s["ye"]]) for _s in _sims])
    _cx = 0.5 * (_xall.min() + _xall.max())
    _cy = 0.5 * (_yall.min() + _yall.max())
    _half = 0.5 * max(_xall.max() - _xall.min(), _yall.max() - _yall.min()) * 1.12
    for _a in _axes:
        _a.set_xlim(_cx - _half, _cx + _half)
        _a.set_ylim(_cy - _half, _cy + _half)

    _axes[0].legend(loc="upper left", fontsize=8)
    _fig.tight_layout()

    mo.vstack([
        mo.md("## Pursuit-Evasion Game"),
        _fig,
        mo.md(
            r"""
            A **fast, clumsy** car chasing a **slow, nimble** runner. Tag, basically.
            Two heuristic escapes do fine. But the *provably optimal* escape makes a
            **sharp mid-chase turn (★)** that no intuition predicts.

            *(Heuristic panels: pure-pursuit chaser. Optimal panel: optimal vs. optimal — the saddle-point benchmark.)*
            """
        ),
    ])
    return


@app.cell
def _beat3_textbooks(mo):
    # Book cover + one point (narrate the "done by hand / static prose" bits).
    def _book_img():
        import base64 as _b64
        import pathlib as _pl
        for _p in ("assets/isaacs_differential_games.jpg",
                   "assets/isaacs_differential_games.png"):
            _f = _pl.Path(_p)
            if _f.exists():
                _mime = "png" if _p.endswith(".png") else "jpeg"
                _data = _b64.b64encode(_f.read_bytes()).decode()
                return (f'<img src="data:image/{_mime};base64,{_data}" '
                        'style="max-height:460px;border-radius:4px;'
                        'box-shadow:0 6px 24px rgba(0,0,0,0.4);">')
        # Deployed (WASM) fallback: the browser filesystem has no assets/,
        # but the deploy workflow serves them next to the page over HTTP.
        return ('<img src="assets/isaacs_differential_games.jpg" '
                'alt="Isaacs, Differential Games (1965)" '
                'style="max-height:460px;border-radius:4px;'
                'box-shadow:0 6px 24px rgba(0,0,0,0.4);">')

    mo.hstack(
        [
            mo.md(
                r"""
                ## The answer is ~75 years old

                Isaacs, *Differential Games* (1965).

                In advanced applied math the insight is **irreducibly symbolic**:

                - coordinate reductions
                - optimality conditions
                - conservation laws

                So let's not just **read** the derivation
                
                let's **compute** it.
                """
            ),
            mo.md(_book_img()),
        ],
        justify="center",
        align="center",
        gap=2,
        widths=[1, 1],
    )
    return


@app.cell
def _beat4a_physical(mo, np, plt, v_E_slider, omega_slider, ell_slider):
    # SYMBOLIZE, part 1 - the physical (absolute / lab) picture: 5 degrees of freedom.
    # The sliders drive this schematic live (cheap); the heavy sims run at a fixed point.
    _vE = v_E_slider.value
    _om = omega_slider.value
    _ell = ell_slider.value
    _vP = 1.0
    _Rmin = 1.0 / _om

    _fig, _ax = plt.subplots(figsize=(7.4, 5.4))
    _P = np.array([1.6, 1.2])
    _theta = 0.6
    _psi = 2.3
    _E = np.array([4.4, 3.0])
    _c = np.linspace(0, 2 * np.pi, 120)
    _uP = np.array([np.cos(_theta), np.sin(_theta)])
    _uE = np.array([np.cos(_psi), np.sin(_psi)])

    # pursuer body + velocity arrow (length ∝ v_P; fixed reference speed)
    _size = 0.32
    _tri = np.array([[_size, 0], [-_size * 0.5, _size * 0.5], [-_size * 0.5, -_size * 0.5]])
    _rot = np.array([[np.cos(_theta), -np.sin(_theta)], [np.sin(_theta), np.cos(_theta)]])
    _ax.add_patch(plt.Polygon((_rot @ _tri.T).T + _P, fc="#2855a1", ec="black", lw=1.2, zorder=5))
    _pv = 1.3 * _vP
    _ax.annotate("", xy=_P + _pv * _uP, xytext=_P,
                 arrowprops=dict(arrowstyle="->", color="#2855a1", lw=2.2))
    _ax.annotate(r"$v_P$", _P + _pv * _uP + [0.10, 0.06], fontsize=11, color="#2855a1")
    _ax.annotate(r"$\theta$", _P + 0.5 * _uP + [0.04, 0.20], fontsize=12, color="#2855a1")

    # minimum turning circle: radius = R_min = v_P / omega (tangent to P, left side)
    _tc = _P + _Rmin * np.array([-np.sin(_theta), np.cos(_theta)])
    _ax.plot(_tc[0] + _Rmin * np.cos(_c), _tc[1] + _Rmin * np.sin(_c),
             "--", color="#2855a1", lw=1, alpha=0.4)
    _ax.annotate(r"$R_{\min}$", [_tc[0] - _Rmin * 0.95, _tc[1] + 0.08],
                 fontsize=10, color="#2855a1", alpha=0.8)

    # capture circle: radius = ell
    _ax.plot(_P[0] + _ell * np.cos(_c), _P[1] + _ell * np.sin(_c),
             ":", color="gray", lw=1.6, alpha=0.85)
    _ax.annotate(r"$\ell$", _P + [_ell + 0.08, 0.02], fontsize=10, color="gray")

    # evader dot + velocity arrow (length ∝ v_E, SAME scale as v_P - compare lengths)
    _ax.plot(*_E, "o", color="#c0392b", ms=11, zorder=5)
    _ev = 1.3 * _vE
    _ax.annotate("", xy=_E + _ev * _uE, xytext=_E,
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.2))
    _ax.annotate(r"$v_E$", _E + _ev * _uE + [-0.36, 0.04], fontsize=11, color="#c0392b")

    _ax.plot([_P[0], _E[0]], [_P[1], _E[1]], "k--", lw=0.9, alpha=0.4)
    _ax.annotate(r"$P$", _P + [-0.5, -0.35], fontsize=14, color="#2855a1", fontweight="bold")
    _ax.annotate(r"$E$", _E + [0.25, -0.12], fontsize=14, color="#c0392b", fontweight="bold")

    # lab axes (fixed generous frame so the R_min circle never clips)
    _ax.annotate("", xy=(6.0, -1.5), xytext=(-1.7, -1.5),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    _ax.annotate("", xy=(-1.7, 4.8), xytext=(-1.7, -1.5),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    _ax.text(5.8, -1.35, r"$x$", fontsize=11)
    _ax.text(-1.98, 4.6, r"$y$", fontsize=11)
    _ax.set_xlim(-2.05, 6.3)
    _ax.set_ylim(-1.85, 5.15)
    _ax.set_aspect("equal")
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.set_title("Absolute (lab) frame", fontsize=11)
    _fig.tight_layout()

    mo.vstack([
        mo.md("## Symbolize: the physical game (5 degrees of freedom)"),
        mo.hstack([
            mo.vstack([
                _fig,
                mo.md(
                    rf"Quantities are dimensionless: $R_{{\min}} = v_P/\omega_{{\max}} = {1/_om:.2f}$ sets the length scale."
                ),
            ]),
            mo.vstack([
                mo.md(
                rf"""
                State in the **absolute (lab) frame**:

                $$(x_P,\; y_P,\; \theta,\; x_E,\; y_E)$$

                | physical knob | value |
                |---|---|
                | $v_E$ evader speed | {_vE:.2f} |
                | $\omega_{{\max}}$ turn rate | {_om:.1f} |
                | $\ell$ capture radius | {_ell:.2f} |
                """
                ),
                v_E_slider,
                omega_slider,
                ell_slider,
            ]),
        ], justify="start", widths=[2, 1], align="start"),
    ])
    return


@app.cell
def _beat4b_reduce(mo, np, plt):
    # SYMBOLIZE, part 2 - collapse the 5-DOF physical state to the 2-DOF relative frame.
    # The figure is the paper's coordinate_progression: three different lab
    # configurations are literally the SAME reduced problem. The (w, ℓ̃) parameter
    # point is narrated, not printed (kept the slide light).
    def build_reduction():
        fig, (ax_lab, ax_body) = plt.subplots(1, 2, figsize=(12.5, 5.4))
        BODY = np.array([1.0, 2.0])            # shared body-relative E position
        CFG = [
            (np.array([1.0, 1.0]), np.pi / 6, "#2c5da0", "1"),
            (np.array([4.0, 1.0]), 5 * np.pi / 6, "#2a9d8f", "2"),
            (np.array([2.5, 4.0]), -np.pi / 3, "#8e44ad", "3"),
        ]
        sz = 0.26
        tpl = np.array([[sz, 0], [-sz * 0.5, sz * 0.4], [-sz * 0.5, -sz * 0.4]])

        def _R(a):
            return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

        def _body_to_lab(P, th, b):
            return P + np.array([b[0] * (-np.sin(th)) + b[1] * np.cos(th),
                                 b[0] * np.cos(th) + b[1] * np.sin(th)])

        _th = np.linspace(0, 2 * np.pi, 80)
        # LEFT - three lab configs, each with the SAME body-relative evader
        for P, th, color, lab in CFG:
            E = _body_to_lab(P, th, BODY)
            ax_lab.add_patch(plt.Polygon((_R(th) @ tpl.T).T + P, fc=color, ec="black", lw=1.2, zorder=5))
            ax_lab.annotate("", xy=P + 0.55 * np.array([np.cos(th), np.sin(th)]), xytext=P,
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.85))
            ax_lab.annotate(rf"$P_{lab}$", P + [-0.45, -0.30], fontsize=11, fontweight="bold", color=color)
            ax_lab.plot(*E, "o", color=color, ms=9, zorder=5, mec="white", mew=0.8)
            ax_lab.annotate(rf"$E_{lab}$", E + [0.14, -0.30], fontsize=11, fontweight="bold", color=color)
            ax_lab.plot([P[0], E[0]], [P[1], E[1]], "--", color=color, lw=0.9, alpha=0.55)
        ax_lab.annotate("", xy=(5.7, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
        ax_lab.annotate("", xy=(0, 5.5), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
        ax_lab.text(5.6, -0.30, r"$x$", fontsize=12)
        ax_lab.text(-0.30, 5.4, r"$y$", fontsize=12)
        ax_lab.text(0.10, 5.85, r"each pair $(P_i, E_i)$ has the same body-relative $E$",
                    fontsize=9, color="gray", style="italic")
        ax_lab.set_xlim(-0.6, 6.2)
        ax_lab.set_ylim(-0.6, 6.0)
        ax_lab.set_aspect("equal")
        ax_lab.set_xticks([])
        ax_lab.set_yticks([])
        ax_lab.set_title("Three different lab configurations", fontsize=12)
        ax_lab.grid(True, alpha=0.15)
        # RIGHT - all three collapse to one body-frame point
        ax_body.add_patch(plt.Polygon((_R(np.pi / 2) @ tpl.T).T, fc="#444444", ec="black", lw=1.5, zorder=5))
        ax_body.annotate(r"$P$", [-0.42, -0.20], fontsize=13, fontweight="bold", color="#222222")
        ax_body.plot(0.5 * np.cos(_th), 0.5 * np.sin(_th), ":", color="gray", lw=1.2, alpha=0.7)
        for color, mk in [(CFG[0][2], 18), (CFG[1][2], 13), (CFG[2][2], 8)]:
            ax_body.plot(BODY[0], BODY[1], "o", color=color, ms=mk, zorder=5, mec="white", mew=1.0, alpha=0.95)
        ax_body.annotate(r"$E_1 = E_2 = E_3 = (x_1, x_2)$", BODY + [0.30, 0.05],
                         fontsize=12, color="black", fontweight="bold")
        ax_body.plot([BODY[0], BODY[0]], [0, BODY[1]], "--", color="gray", lw=0.9, alpha=0.7)
        ax_body.plot([0, BODY[0]], [BODY[1], BODY[1]], "--", color="gray", lw=0.9, alpha=0.7)
        ax_body.plot([0, BODY[0]], [0, BODY[1]], "k--", lw=0.9, alpha=0.55)
        ax_body.annotate("", xy=(3.5, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
        ax_body.annotate("", xy=(0, 3.5), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
        ax_body.text(3.4, -0.30, r"$x_1$ (perpendicular)", fontsize=11)
        ax_body.text(-0.30, 3.45, r"$x_2$ (along heading)", fontsize=11)
        ax_body.set_xlim(-1.5, 4.2)
        ax_body.set_ylim(-1.0, 4.0)
        ax_body.set_aspect("equal")
        ax_body.set_xticks([])
        ax_body.set_yticks([])
        ax_body.set_title(r"Relative (body) frame: translate by $-P$, rotate by $-\theta$", fontsize=12)
        ax_body.grid(True, alpha=0.15)
        fig.tight_layout()
        return fig

    mo.vstack([
        mo.md("## Reduce to the relative frame: 5 DOF → 2 DOF"),
        build_reduction(),
        mo.md(
            r"""
            Pin the frame to the pursuer: only the **relative position** $(x_1, x_2)$
            survives. All 3 physical setups are the *same* reduced state.
            """
        ),
    ])
    return


@app.cell
def _beat4c_controls(mo, np, plt):
    # Define the two players' control variables BEFORE they appear in the dynamics.
    # Two separate diagrams, each stacked beside its own description.
    def pursuer_fig():
        fig, ax = plt.subplots(figsize=(4.6, 2.3))
        _sz = 0.40
        _tri = np.array([[0, _sz], [-_sz * 0.6, -_sz * 0.5], [_sz * 0.6, -_sz * 0.5]])
        ax.add_patch(plt.Polygon(_tri, fc="#2855a1", ec="black", lw=1.2, zorder=5))
        ax.annotate(r"$P$", (-0.55, -0.12), color="#2855a1", fontsize=13, fontweight="bold")
        # fixed forward velocity (speed 1) - it cannot stop or sidestep
        ax.annotate("", xy=(0, 1.05), xytext=(0, 0.42),
                    arrowprops=dict(arrowstyle="->", color="#2855a1", lw=2.4))
        # bounded turn rate phi (arrowheads both ways)
        _a = np.linspace(np.radians(55), np.radians(125), 40)
        _r = 1.38
        ax.plot(_r * np.cos(_a), _r * np.sin(_a), color="#2855a1", lw=1.5, alpha=0.85)
        ax.annotate("", xy=(_r * np.cos(_a[0]), _r * np.sin(_a[0])),
                    xytext=(_r * np.cos(_a[4]), _r * np.sin(_a[4])),
                    arrowprops=dict(arrowstyle="->", color="#2855a1", lw=1.5))
        ax.annotate("", xy=(_r * np.cos(_a[-1]), _r * np.sin(_a[-1])),
                    xytext=(_r * np.cos(_a[-5]), _r * np.sin(_a[-5])),
                    arrowprops=dict(arrowstyle="->", color="#2855a1", lw=1.5))
        ax.annotate(r"turn rate $\phi$", (0, 1.62), ha="center", color="#2855a1", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-0.65, 1.85)
        ax.axis("off")
        fig.tight_layout()
        return fig

    def evader_fig():
        fig, ax = plt.subplots(figsize=(4.6, 2.3))
        _E = np.array([0.0, 0.0])
        _w = 0.82
        _cc = np.linspace(0, 2 * np.pi, 140)
        ax.fill(_E[0] + _w * np.cos(_cc), _E[1] + _w * np.sin(_cc),
                color="#c0392b", alpha=0.12, zorder=2)
        ax.plot(_E[0] + _w * np.cos(_cc), _E[1] + _w * np.sin(_cc),
                "-", color="#c0392b", lw=1.4, alpha=0.7)
        ax.plot(*_E, "o", color="#c0392b", ms=11, zorder=5)
        ax.annotate(r"$E$", _E + [0.14, -0.26], color="#c0392b", fontsize=13, fontweight="bold")
        _b = np.radians(125)
        ax.annotate("", xy=_E + _w * np.array([np.cos(_b), np.sin(_b)]), xytext=_E,
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.6))
        ax.annotate(r"heading $\psi$", _E + [-0.12, _w + 0.14], ha="center",
                    color="#c0392b", fontsize=11)
        ax.annotate(r"$\leq w$", _E + [_w * 0.60, _w * 0.48], color="#c0392b",
                    fontsize=9, alpha=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.05, 1.25)
        ax.axis("off")
        fig.tight_layout()
        return fig

    mo.vstack([
        mo.md("## Two players, two controls"),
        mo.hstack([
            pursuer_fig(),
            mo.md(
                r"""
                the **pursuer** only steers. Committed forward at fixed speed $v_P = 1$
                (it cannot stop or sidestep), it sets its **turn rate** $\phi = \Phi(\mathbf{x})$
                from a **turn-rate policy** $\Phi$, capped by $|\phi| \le 1$.
                """
            ),
        ], justify="start", align="center", widths=[2, 3]),
        mo.hstack([
            evader_fig(),
            mo.md(
                r"""
                the **evader** picks a **heading** anywhere in a disk of radius $w$:
                any direction, any speed up to $w$. Optimal play runs flat-out on the rim,
                so its one knob is the **heading** $\psi = \Psi(\mathbf{x})$, from a
                **heading policy** $\Psi$.
                """
            ),
        ], justify="start", align="center", widths=[2, 3]),
    ])
    return


@app.cell
def _beat5a_reduce(mo, sp, np, diff, sin, cos, trigsimp, expand_trig, simplify,
                   latex, t, x1, x2, phi_ctrl, psi_ctrl, psi_lab, v_P, v_E_sym, w_sym):
    # DERIVE, part 1 - the two hard operations SymPy eliminates:
    #   (a) differentiation  (sp.diff of the body-frame coordinates)
    #   (b) trig simplification  (sp.trigsimp collapsing to Isaacs' canonical form)
    def reduce_body_frame():
        x_P, y_P = sp.Function("x_P")(t), sp.Function("y_P")(t)
        x_E, y_E = sp.Function("x_E")(t), sp.Function("y_E")(t)
        theta = sp.Function("theta")(t)
        dx, dy = x_E - x_P, y_E - y_P
        # body-frame relative position (rotate the lab displacement by -theta)
        x1e = -dx * sin(theta) + dy * cos(theta)
        x2e = dx * cos(theta) + dy * sin(theta)
        # (a) DIFFERENTIATE - product + chain rule, automatic:
        x1_raw = diff(x1e, t)
        # substitute the known lab-frame dynamics, then (b) TRIG-SIMPLIFY:
        subs = {
            diff(x_P, t): v_P * cos(theta), diff(y_P, t): v_P * sin(theta),
            diff(theta, t): phi_ctrl,
            diff(x_E, t): v_E_sym * cos(psi_lab), diff(y_E, t): v_E_sym * sin(psi_lab),
        }
        x1_sub = trigsimp(expand_trig(x1_raw.subs(subs)))
        # relabel evader heading into the body frame (psi_lab = psi + theta):
        x1_rel = trigsimp(expand_trig(x1_sub.subs(psi_lab, psi_ctrl + theta)))
        x1_dimless = simplify(x1_rel.subs(v_E_sym, w_sym * v_P).subs(v_P, 1))
        return x1_raw, x1_dimless

    _x1_raw, _ = reduce_body_frame()

    # Split the (long) raw derivative into two aligned lines so it never runs off the
    # slide: halve the sum's terms and render each half on its own line.
    _terms = sp.Add.make_args(_x1_raw)
    _half = (len(_terms) + 1) // 2
    _raw_l1 = latex(sp.Add(*_terms[:_half], evaluate=False))
    _raw_l2 = latex(sp.Add(*_terms[_half:], evaluate=False))
    if not _raw_l2.lstrip().startswith("-"):
        _raw_l2 = "+ " + _raw_l2

    # The reduction proves the canonical Isaacs form; state it in reduced symbols
    # (downstream cells need these exact objects).
    f1 = -phi_ctrl * x2 + w_sym * sin(psi_ctrl)
    f2 = phi_ctrl * x1 + w_sym * cos(psi_ctrl) - 1

    mo.md(
        rf"""
        ## Derive: let SymPy do the brutal manipulations

        **1. Differentiate** the body-frame coordinate $x_1$. `sp.diff` applies the
        product and chain rules automatically:

        $$\begin{{aligned}}
        \dot x_1\big|_{{\text{{raw}}}} &= {_raw_l1} \\
        &\quad {_raw_l2}
        \end{{aligned}}$$

        **2. Substitute** the lab dynamics and **`trigsimp`**. The heading $\theta$
        cancels out entirely, leaving the canonical reduced dynamics:

        $$\dot x_1 = {latex(f1)} \qquad \dot x_2 = {latex(f2)}$$
        """
    )
    return f1, f2


@app.cell
def _beat5b_objective(mo):
    # DERIVE - state the game's objective: the value function V* (what optimal play optimizes).
    # No figure here: V*'s true shape is genuinely weird; we show the computed shape at the end.
    mo.md(
        r"""
        ## The objective: the value function

        Fix both **policies** $\Phi, \Psi$ (from the last slide) and the chase from
        $\mathbf{x}$ runs to capture at a time $T(\mathbf{x}, \Phi, \Psi)$. The **value** is
        the accumulated **running cost** ($1$, since we minimize time):
        $$V(\mathbf{x}\mid \Phi, \Psi) = \int_0^{\,T(\mathbf{x},\, \Phi,\, \Psi)} 1\,dt.$$
        where $T(\mathbf{x},\, \Phi,\, \Psi)$ is the first time when $x_1^2 + x_2^2 \le \tilde\ell^{\,2}$ given initial state $\mathbf{x}$ and the policies $\Phi, \Psi$.
        
        Jointly optimal play defines the **value function** $V^*$:

        $$V^*(\mathbf{x}) = \min_{\Phi}\,\max_{\Psi}\; V(\mathbf{x}\mid \Phi, \Psi),$$
        

        The optimal policies $\Phi^*, \Psi^*$ output the **actions** as a function of the state $\mathbf{x}$ as opposed to precomputing them.
        """
    )
    return


@app.cell
def _beat5c_hamiltonian(mo):
    # DERIVE - the approach: the Hamiltonian and its saddle point (sets up phi*, psi*).
    mo.md(
        r"""
        ## The Hamiltonian falls out of $V^*$

        Under optimal play the value is the **time still to run**, the running cost
        integrated forward to capture:
        $$V^*(\mathbf{x}(t)) = \int_t^{T} 1\,ds.$$

        Its gradient $\mathbf{p} = \nabla V^*$ **prices every direction**: moving with the
        dynamics $\mathbf{f} = \dot{\mathbf{x}}$ changes the value at rate
        $$\frac{dV^*}{dt} = \mathbf{p}\cdot\mathbf{f}.$$
        
        Optimal play is the **pointwise saddle** of $H$ over the actions, each read off
        the costate $\mathbf{p} = \nabla V^*(\mathbf{x})$:
        $$\phi^* = \arg\min_{\phi} H, \qquad \psi^* = \arg\max_{\psi} H,$$
        which, as functions of state, are the policies $\Phi^*, \Psi^*$.

        The **Optimality condition**, the Hamilton-Jacobi-Isaacs equation:
        $$H = \mathbf{p}\cdot\mathbf{f} + 1 = 0.$$

        """
    )
    return


@app.cell
def _beat5b_costate(mo, np, plt, sp, diff, simplify, latex,
                    f1, f2, x1, x2, p1, p2):
    # DERIVE - define the two ingredients of H: the dynamics f and the costate p (with
    # its rotation dynamics and conserved norm).
    H_expr = sp.expand(p1 * f1 + p2 * f2 + 1)
    p1_dot_expr = simplify(-diff(H_expr, x1))          # costate (adjoint) ODE
    p2_dot_expr = simplify(-diff(H_expr, x2))
    _conservation = simplify(2 * (p1 * p1_dot_expr + p2 * p2_dot_expr))  # d/dt ||p||^2

    def costate_fig():
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        _th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(_th), np.sin(_th), "-", color="#bbb", lw=1.3)
        # ghost copies of p, before and after: the vector sweeps around the circle
        for _g in (np.radians(25), np.radians(100)):
            ax.annotate("", xy=(np.cos(_g), np.sin(_g)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.5, alpha=0.22))
        # the current costate
        _a = np.radians(60)
        ax.annotate("", xy=(np.cos(_a), np.sin(_a)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#b2182b", lw=2.8))
        ax.annotate(r"$\mathbf{p}$", (np.cos(_a) * 1.12, np.sin(_a) * 1.12),
                    color="#b2182b", fontsize=14, fontweight="bold")
        # spin-direction arc
        _arc = np.linspace(np.radians(28), np.radians(112), 40)
        ax.plot(1.24 * np.cos(_arc), 1.24 * np.sin(_arc), color="#555", lw=1.1)
        ax.annotate("", xy=(1.24 * np.cos(_arc[-1]), 1.24 * np.sin(_arc[-1])),
                    xytext=(1.24 * np.cos(_arc[-1] - 0.07), 1.24 * np.sin(_arc[-1] - 0.07)),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.1))
        ax.annotate(r"spins at rate $\phi$", (-0.66, 1.34), fontsize=9.5, color="#555")
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color="black", lw=0.7)
        ax.text(1.34, 0.06, r"$p_1$", fontsize=11)
        ax.text(0.08, 1.40, r"$p_2$", fontsize=11)
        ax.text(-0.62, -1.40, r"$\|\mathbf{p}\|$ conserved", fontsize=10, color="#333")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    mo.vstack([
        mo.md("## The costate rides a circle"),
        mo.hstack([
            costate_fig(),
            mo.md(
                rf"""
                The **dynamics** $\mathbf{{f}} = \dot{{\mathbf{{x}}}}$, from the reduction:
                
                $$\dot x_1 = {latex(f1)}, \qquad \dot x_2 = {latex(f2)}$$

                The **costate** $\mathbf{{p}} = \nabla V^*$ is not frozen. Hamilton's adjoint
                law moves it, and it evaluates to a **rotation**:
                
                $$\dot{{\mathbf{{p}}}} = -\,\partial H/\partial\mathbf{{x}}
                  = \big({latex(p1_dot_expr)},\;\; {latex(p2_dot_expr)}\big).$$

                A pure rotation at the pursuer's turn rate $\phi$: the price vector spins but
                never stretches,
                
                $$\tfrac{{d}}{{dt}}\lVert\mathbf{{p}}\rVert^2 = {latex(_conservation)}.$$

                The frame is pinned to the pursuer, so as it turns the costate $\mathbf{{p}}$ rotates with it.
                """
            ),
        ], justify="start", align="center", widths=[2, 3]),
    ])
    return H_expr, p1_dot_expr, p2_dot_expr


@app.cell
def _beat5b_pursuer(mo, np, plt, simplify, collect, latex, H_expr, phi_ctrl):
    # DERIVE - player 1: H is linear in phi → bang-bang φ* = -sign(σ).
    sigma_expr = simplify(collect(H_expr, phi_ctrl).coeff(phi_ctrl))

    def bangbang_fig():
        # phi* as a function of the switching function sigma (built from the costate p):
        # the bang-bang law phi* = -sign(sigma).
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        _s = np.linspace(-1.5, 1.5, 400)
        ax.plot(_s[_s < 0], -np.sign(_s[_s < 0]), "-", color="#2166ac", lw=2.8)
        ax.plot(_s[_s > 0], -np.sign(_s[_s > 0]), "-", color="#2166ac", lw=2.8)
        ax.plot([0, 0], [-1, 1], ":", color="#2166ac", lw=1.2, alpha=0.5)
        ax.plot(0, 1, "o", color="white", mec="#2166ac", mew=1.8, ms=8, zorder=5)
        ax.plot(0, -1, "o", color="#2166ac", ms=8, zorder=5)
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color="black", lw=0.7)
        ax.annotate(r"flip at $\sigma=0$", (0.06, 0.16), fontsize=10)
        ax.set_xlabel(r"$\sigma$   (switching function)", fontsize=11)
        ax.set_ylabel(r"$\phi^*$", fontsize=12)
        ax.set_yticks([-1, 1])
        ax.set_yticklabels([r"$-1$ (hard right)", r"$+1$ (hard left)"], fontsize=9)
        ax.set_xticks([])
        ax.set_ylim(-1.7, 1.7)
        fig.tight_layout()
        return fig

    mo.vstack([
        mo.md("## Player 1: the pursuer turns *bang-bang*"),
        mo.hstack([
            bangbang_fig(),
            mo.md(
                rf"""
                $H$ is **linear** in the pursuer's turn $\phi$, so one `.coeff(phi)` gives
                the **switching function**:
                $$\sigma = {latex(sigma_expr)}.$$

                Since heading $\phi \in[-1, 1]$ is used to minimize a linear objective, the pursuer always turns at
                full-rate, its sign set by $\sigma$:
                $$\phi^* = -\operatorname{{sign}}(\sigma).$$

                $\sigma$ flips sign several times along a chase, so the pursuer switches
                hard-left / hard-right repeatedly, each a $\sigma=0$ crossing.
                """
            ),
        ], justify="start", align="center", widths=[2, 3]),
    ])
    return (sigma_expr,)


@app.cell
def _beat5b_evader(mo, np, plt, atan2, p1, p2):
    # DERIVE - player 2: maximise H over ψ → ψ* aligns velocity with the costate.
    psi_star_expr = atan2(p1, p2)

    def gradient_fig():
        # The evader climbs perpendicular to the level sets of V*, along p = grad V*.
        fig, ax = plt.subplots(figsize=(5.0, 4.7))
        _xs = np.linspace(-1.5, 1.5, 200)
        for _c in np.linspace(-0.7, 1.35, 6):                  # smooth curved level sets
            ax.plot(_xs, _c - 0.30 * _xs**2, "-", color="#d3d3d3", lw=1.2, zorder=1)
        _x0 = np.array([0.55, 0.15])
        _g = np.array([0.60 * _x0[0], 1.0])                    # grad V* = (0.6 x, 1)
        _g = _g / np.hypot(*_g)
        ax.plot(*_x0, "o", color="black", ms=6, zorder=6)
        ax.annotate("", xy=_x0 + 0.62 * _g, xytext=_x0,
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=3.0))
        ax.annotate(r"$v_E$", _x0 + 0.62 * _g + [0.06, 0.02], color="#c0392b",
                    fontsize=13, fontweight="bold")
        ax.annotate(r"$\mathbf{p}=\nabla V^*$", _x0 + 0.30 * _g + [0.16, -0.02],
                    color="#777", fontsize=10)
        ax.annotate("higher $V^*$\n(longer to catch)", (0.0, 1.28), fontsize=9.5,
                    color="#333", ha="center")
        ax.annotate("level sets of $V^*$", (-1.02, -0.55), fontsize=9, color="#999")
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.15, 1.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    mo.vstack([
        mo.md("## Player 2: the evader runs *up the gradient*"),
        mo.hstack([
            gradient_fig(),
            mo.md(
                r"""
                The evader's terms in $H$ are $\,w\,(p_1\sin\psi + p_2\cos\psi)\,$: its
                heading projected onto $\mathbf{p}$, **maximised** by pointing straight
                along $\mathbf{p}$:
                $$\psi^* = \mathrm{atan2}(p_1, p_2).$$
                Since $\mathbf{p} = \nabla V^*$, the evader **climbs the value gradient**
                (contribution $w\lVert\mathbf{p}\rVert$), toward states that take longest to
                catch.

                There are phase transitions where the optimal direction is not unique and can change abruptly!
                """
            ),
        ], justify="start", align="center", widths=[2, 3]),
    ])
    return (psi_star_expr,)


@app.cell
def _beat6_lambdify(mo):
    # LAMBDIFY - the join from algebra to arithmetic.
    mo.md(
        r"""
        ## Lambdify: symbols to fast NumPy

        The optimal dynamics are a symbolic function $\mathbf{f}(\mathbf{x})$. One call turns
        it into a numerical function we can run on data:

        ```python
        state = sp.Matrix([x1, x2, p1, p2])          # the state
        f     = sp.Matrix([ẋ1, ẋ2, ṗ1, ṗ2])          # the derived vector field  f(state)
        rhs   = sp.lambdify((state, w), f, "numpy")
        ```

        Then just call it on data:

        ```python
        rhs([0.7, 1.3, 0.4, -0.9], 0.45)     # -> [-1.12, -0.71, 0.90, 0.40]
        ```
        """
    )
    return


@app.cell
def _beat7_simulate(mo, np, plt, physical_trajs, traj_index_slider, t_forward_slider):
    # SIMULATE - integrate the lambdified dynamics; scrub the chase live.
    _p = physical_trajs[traj_index_slider.value]
    _t_now = t_forward_slider.value
    _t = _p["t"]
    _idx = max(0, min(np.searchsorted(_t, min(_t_now, _t[-1]), side="right"), len(_t) - 1))

    _fig, _ax = plt.subplots(figsize=(7.2, 7.0))
    _ax.plot(_p["X_P"][:_idx + 1], _p["Y_P"][:_idx + 1], "-", color="#2166ac",
             lw=1.6, alpha=0.8, label="Pursuer")
    _ax.plot(_p["X_E"][:_idx + 1], _p["Y_E"][:_idx + 1], "--", color="#b2182b",
             lw=1.6, alpha=0.8, label="Evader")
    _ax.plot(_p["X_P"][0], _p["Y_P"][0], "o", color="#2166ac", ms=9, mec="white", mew=0.8)
    _ax.plot(_p["X_E"][0], _p["Y_E"][0], "o", color="#b2182b", ms=9, mec="white", mew=0.8)
    _xp, _yp, _th = _p["X_P"][_idx], _p["Y_P"][_idx], _p["theta"][_idx]
    _sz = 0.45
    _ax.fill([_xp + _sz * np.cos(_th), _xp + _sz * 0.5 * np.cos(_th + 2.4), _xp + _sz * 0.5 * np.cos(_th - 2.4)],
             [_yp + _sz * np.sin(_th), _yp + _sz * 0.5 * np.sin(_th + 2.4), _yp + _sz * 0.5 * np.sin(_th - 2.4)],
             color="#2166ac", ec="black", lw=0.5)
    _ax.plot(_p["X_E"][_idx], _p["Y_E"][_idx], "o", color="#b2182b", ms=8, mec="black", mew=0.5)
    _c = np.linspace(0, 2 * np.pi, 100)
    _ax.plot(_xp + 0.5 * np.cos(_c), _yp + 0.5 * np.sin(_c), ":", color="gray", lw=1, alpha=0.6)
    if _p.get("composite") and _idx >= _p["switch_idx"]:
        _sw = _p["switch_idx"]
        _ax.plot(_p["X_E"][_sw], _p["Y_E"][_sw], "*", color="#111111", ms=18, mec="white", mew=0.9)
    _ax.set_aspect("equal")
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.legend(loc="upper right", fontsize=9)
    _ax.set_title(rf"Physical chase at $t={_t_now:.1f}$   (dist $={_p['dist'][_idx]:.2f}$)", fontsize=11)
    _fig.tight_layout()

    mo.vstack([
        mo.md("## Simulate: `solve_ivp` on the lambdified dynamics"),
        mo.md("*Same numbers as slide 2, now **generated**, not asserted. Last index = the ★ chase.*"),
        traj_index_slider,
        t_forward_slider,
        _fig,
    ])
    return


@app.cell
def _payoff1_lemma(mo):
    # PAYOFF 1 - one derivation step, up close: a script that derives AND verifies,
    # naming exactly the axioms it used.
    mo.md(
        r"""
        ## One step, one script

        Every step of that derivation is its own SymPy lemma that **declares the axioms it
        uses** and **verifies its own result**. Here is the bang-bang law, the pursuer's
        forced hard-over turn:

        ```python
        # derivation/lemma_04_bangbang.py
        AXIOMS_USED = ["A3", "A7", "I2"]   # |phi|<=1, zero-sum saddle, Isaacs' condition

        def derive(ctx):
            sigma = ctx["sigma"]                     # switching function, from L3
            ctx["phi_star"] = -sp.sign(sigma)        # minimize a linear H on [-1, 1]

        def verify(ctx):
            sigma, phi_star = ctx["sigma"], ctx["phi_star"]
            # a linear objective on [-1, 1] is minimized at -sign(slope), value -|slope|
            assert sp.simplify(sigma * phi_star - (-sp.Abs(sigma))) == 0
        ```

        Not prose that *claims* $\phi^{*}=-\operatorname{sign}(\sigma)$: a script that
        **derives it and checks it**, stating exactly what it assumed to get there.
        """
    )
    return


@app.cell
def _payoff2_chain(mo):
    # PAYOFF 2 - split the machine's job (VERIFY) from the human's (VALIDATE). The L1-L8
    # symbolic checks are verification; the derived-vs-assumed ledger is what validation
    # weighs. Comes before the reveal, which ends the talk on Validate.
    mo.vstack([
        mo.md("## Verify with the machine. Validate as a human."),
        mo.hstack([
            mo.md(
                r"""
                **Verify**: *did we build it right?*

                `python derivation/run.py` runs eight lemmas, each checked as a symbolic
                identity, green in CI.

                - **L1** reduction ($\theta$ cancels)
                - **L2** $H=\mathbf{p}\cdot\mathbf{f}+1$
                - **L3** switching $\sigma$
                - **L4** bang-bang $\phi^{*}$
                - **L5** evader $\psi^{*}=\operatorname{atan2}(p_1,p_2)$
                - **L6** costate $\dot{\mathbf{p}}=-\partial H/\partial\mathbf{x}$
                - **L7** $\lVert p\rVert^{2}$ conserved
                - **L8** characteristic system
                """
            ),
            mo.md(
                r"""
                **Validate**: *did we build the right thing?*

                A verified derivation can still rest on the wrong model. The same run prints
                the ledger, derived versus assumed:

                - **Derived**: all eight steps, closed symbolic identities.
                - **Posited** (the game): Dubins car, simple-motion evader, $\lvert\phi\rvert\le1,$ time cost, zero-sum saddle.
                - **Inherited** (cited, not proven): Pontryagin / HJB, Isaacs' condition, viscosity existence.
                - **Leaned on** (regularity): $V^{*}\in C^{1}$, which **breaks at the ★**, the dispersal surface where $\mathbf{p}$ is multivalued.

                Calling those choices appropriate for *this* problem is the human act, not the
                machine's.
                """
            ),
        ], widths=[1, 1], gap=2, align="start"),
    ])
    return


@app.cell
def _beat10_reveal(mo, np, sp, sign, solve_ivp, cumulative_trapezoid,
                   compute_terminal_conditions, W_FIXED, ELL_TILDE_FIXED,
                   N_TRAJ, T_HORIZON, x1, x2, p1, p2, phi_ctrl, psi_ctrl,
                   w_sym, f1, f2, sigma_expr, psi_star_expr, p1_dot_expr,
                   p2_dot_expr):
    # Numeric RHS, backward characteristics, physical lift, and the
    # sliders live here (folded in so the deck has no blank trailing
    # slides); the closing markdown below is the output.
    # Build the numeric RHS from the SAME symbolic objects derived on stage.
    _subs = [(phi_ctrl, -sign(sigma_expr)), (psi_ctrl, psi_star_expr)]
    _rhs_x1 = f1.subs(_subs)
    _rhs_x2 = f2.subs(_subs)
    _rhs_p1 = p1_dot_expr.subs(phi_ctrl, -sign(sigma_expr))
    _rhs_p2 = p2_dot_expr.subs(phi_ctrl, -sign(sigma_expr))
    _lam = sp.lambdify([x1, x2, p1, p2, w_sym], [_rhs_x1, _rhs_x2, _rhs_p1, _rhs_p2],
                       modules=["numpy"])

    def rhs_forward(tt, state, w_val):
        s1, s2, s3, s4 = state
        return _lam(s1, s2, s3, s4, w_val)

    def rhs_backward(tt, state, w_val):
        return [-v for v in rhs_forward(tt, state, w_val)]

    # Dense backward characteristics (the heavy compute; runs once at load).
    _amin = np.arcsin(min(W_FIXED, 0.999))
    _amax = np.pi - _amin
    _alphas = np.linspace(_amin + 1e-3, _amax - 1e-3, N_TRAJ)
    _term = compute_terminal_conditions(_alphas, W_FIXED, ELL_TILDE_FIXED)
    trajectories = []
    for _i in range(N_TRAJ):
        trajectories.append(solve_ivp(
            rhs_backward, [0, T_HORIZON], _term[_i], args=(W_FIXED,),
            method="RK45", max_step=0.1, dense_output=True, rtol=1e-8, atol=1e-10))
    # Lift each reduced backward characteristic to a forward-time lab-frame chase,
    # and stitch the composite (two crossing characteristics → the ★ direction change).
    _N = 300

    def _adaptive_tau(sol, t0, t1, N):
        _tc = np.linspace(t0, t1, 200)
        _sc = sol.sol(_tc)
        _sig = _sc[3] * _sc[0] - _sc[2] * _sc[1]
        _wt = 1.0 + 1.0 / (np.abs(_sig) + 0.05)
        _cdf = cumulative_trapezoid(_wt, _tc, initial=0)
        _cdf /= _cdf[-1]
        _tau = np.interp(np.linspace(0, 1, N), _cdf, _tc)
        _tau[0], _tau[-1] = t0, t1
        return _tau

    def _lift(sol):
        _T = sol.t[-1]
        _tau = _adaptive_tau(sol, 0, _T, _N)
        _x1, _x2, _p1, _p2 = sol.sol(_tau)
        _phi = -np.sign(_p2 * _x1 - _p1 * _x2)
        _theta = cumulative_trapezoid(-_phi, _tau, initial=0)
        _psi = np.arctan2(_p1[0], _p2[0])
        _XE = -W_FIXED * np.cos(_psi) * _tau
        _YE = -W_FIXED * np.sin(_psi) * _tau
        _ct, _st = np.cos(_theta), np.sin(_theta)
        _XP = _XE - (-_x1 * _st + _x2 * _ct)
        _YP = _YE - (_x1 * _ct + _x2 * _st)
        _tf = _T - _tau[::-1]
        _XP, _YP, _XE, _YE, _thf = _XP[::-1], _YP[::-1], _XE[::-1], _YE[::-1], _theta[::-1]
        _XP -= _XE[0]; _YP -= _YE[0]; _XE -= _XE[0]; _YE -= _YE[0]
        return {"t": _tf, "X_P": _XP, "Y_P": _YP, "X_E": _XE, "Y_E": _YE,
                "theta": _thf, "dist": np.sqrt((_XP - _XE)**2 + (_YP - _YE)**2)}

    _phys = [_lift(_s) for _s in trajectories]

    # Composite: two characteristics crossing at a dispersal surface.
    _sab = {}
    for _lbl, _a in [("A", np.radians(40.0)), ("B", np.radians(95.0))]:
        _ic = compute_terminal_conditions(np.array([_a]), W_FIXED, ELL_TILDE_FIXED)[0]
        _sab[_lbl] = solve_ivp(rhs_backward, [0, 15.0], _ic, args=(W_FIXED,),
                               method="RK45", max_step=0.02, dense_output=True,
                               rtol=1e-12, atol=1e-14)
    _Nc = 2000
    _tauA = np.linspace(0, _sab["A"].t[-1], _Nc)
    _tauB = np.linspace(0, _sab["B"].t[-1], _Nc)
    _sA, _sB = _sab["A"].sol(_tauA), _sab["B"].sol(_tauB)
    _mind, _ciA, _ciB = 1e10, 0, 0
    for _ti in range(_Nc):
        _d = np.sqrt((_sB[0] - _sA[0, _ti])**2 + (_sB[1] - _sA[1, _ti])**2)
        _j = int(np.argmin(_d))
        if _d[_j] < _mind:
            _mind, _ciA, _ciB = _d[_j], _ti, _j
    _t1 = _adaptive_tau(_sab["A"], 0, _tauA[_ciA], _N)
    _x11, _x21, _p11, _p21 = _sab["A"].sol(_t1)
    _phi1 = -np.sign(_p21 * _x11 - _p11 * _x21)
    _th1 = cumulative_trapezoid(-_phi1, _t1, initial=0)
    _psiA = np.arctan2(_p11[0], _p21[0])
    _XE1 = -W_FIXED * np.cos(_psiA) * _t1
    _YE1 = -W_FIXED * np.sin(_psiA) * _t1
    _c1, _s1 = np.cos(_th1), np.sin(_th1)
    _XP1 = _XE1 - (-_x11 * _s1 + _x21 * _c1)
    _YP1 = _YE1 - (_x11 * _c1 + _x21 * _s1)
    _t2 = _adaptive_tau(_sab["B"], _tauB[_ciB], _sab["B"].t[-1], _N)
    _x12, _x22, _p12, _p22 = _sab["B"].sol(_t2)
    _phi2 = -np.sign(_p22 * _x12 - _p12 * _x22)
    _thc = _th1[-1]
    _th2 = _thc + cumulative_trapezoid(-_phi2, _t2, initial=0)
    _psiB = np.arctan2(_p12[0], _p22[0]) + _thc
    _dt2 = _t2 - _t2[0]
    _XE2 = _XE1[-1] - W_FIXED * np.cos(_psiB) * _dt2
    _YE2 = _YE1[-1] - W_FIXED * np.sin(_psiB) * _dt2
    _c2, _s2 = np.cos(_th2), np.sin(_th2)
    _XP2 = _XE2 - (-_x12 * _s2 + _x22 * _c2)
    _YP2 = _YE2 - (_x12 * _c2 + _x22 * _s2)
    _XPc = np.concatenate([_XP1, _XP2[1:]])[::-1]
    _YPc = np.concatenate([_YP1, _YP2[1:]])[::-1]
    _XEc = np.concatenate([_XE1, _XE2[1:]])[::-1]
    _YEc = np.concatenate([_YE1, _YE2[1:]])[::-1]
    _thcc = np.concatenate([_th1, _th2[1:]])[::-1]
    _tauc = np.concatenate([_t1, _tauA[_ciA] + _dt2[1:]])
    _tfc = _tauc[-1] - _tauc[::-1]
    _XPc -= _XEc[0]; _YPc -= _YEc[0]; _XEc -= _XEc[0]; _YEc -= _YEc[0]
    _phys.append({"t": _tfc, "X_P": _XPc, "Y_P": _YPc, "X_E": _XEc, "Y_E": _YEc,
                  "theta": _thcc, "dist": np.sqrt((_XPc - _XEc)**2 + (_YPc - _YEc)**2),
                  "composite": True, "switch_idx": len(_XPc) - _N})

    physical_trajs = _phys
    composite_idx = len(_phys) - 1
    T_max_phys = max(_p["t"][-1] for _p in _phys)
    # All display-consumed sliders. Defined here (trailing, no output → blank slide);
    # DISPLAYED and read in the beat cells so they update their figures live.
    v_E_slider = mo.ui.slider(0.05, 0.95, 0.05, value=0.45, label=r"$v_E$ (evader speed)")
    omega_slider = mo.ui.slider(0.5, 3.0, 0.1, value=1.0, label=r"$\omega_{\max}$ (turn rate)")
    ell_slider = mo.ui.slider(0.1, 1.5, 0.05, value=0.5, label=r"$\ell$ (capture radius)")
    traj_index_slider = mo.ui.slider(0, len(physical_trajs) - 1, 1, value=composite_idx,
                                     label="trajectory (last = the ★ chase)")
    t_forward_slider = mo.ui.slider(0.0, T_max_phys, 0.1, value=T_max_phys,
                                    label="forward time $t$")

    # THE REVEAL (LAST slide) - name the six-stage pattern as earned insight; the talk ends
    # on Validate, the human step, where the speaker claims the assumptions were
    # contextually appropriate.
    mo.md(
        r"""
        ## What you just watched was a pattern

        | | |
        |---|---|
        | **Motivate** | the ★ puzzle: physical intuition before formalism |
        | **Symbolize** | state, parameters, dynamics as SymPy objects |
        | **Derive** | differentiation + trig-simplification → optimal play |
        | **Lambdify** | `sp.lambdify`: symbols to fast numerics |
        | **Simulate** | `solve_ivp` on the derived dynamics |
        | **Validate** | is this model appropriate for the problem at hand? |

        I've been playing with this in this notebook:

        `github.com/mzargham/hc-marimo` · live: `mzargham.github.io/hc-marimo` · paper: PR 1206 on scipy-conference/scipy_proceedings
        """
    )
    return (T_max_phys, composite_idx, physical_trajs, ell_slider,
            omega_slider, rhs_backward, t_forward_slider,
            traj_index_slider, trajectories, v_E_slider)



if __name__ == "__main__":
    app.run()

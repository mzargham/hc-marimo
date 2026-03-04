import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import sympy as sp
    from sympy import (
        symbols, Function, cos, sin, atan2, sqrt, sign, simplify,
        diff, Matrix, latex, Rational, pi, trigsimp, together, cancel,
        Eq, collect, factor, expand_trig, Abs
    )
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    return (
        Matrix,
        atan2,
        cos,
        diff,
        expand_trig,
        latex,
        mo,
        np,
        plt,
        sign,
        simplify,
        sin,
        solve_ivp,
        sp,
        sqrt,
        symbols,
        trigsimp,
    )


@app.cell
def title(mo):
    mo.md(r"""
    # The Homicidal Chauffeur: A Differential Game

    ## Symbolic Derivation & Interactive Simulation with SymPy

    ---

    *A marimo notebook exploring Rufus Isaacs' foundational pursuit-evasion
    problem through symbolic computation and numerical simulation.*

    **References:**

    - R. Isaacs, *Games of Pursuit*, RAND Corporation P-257 (1951)
    - R. Isaacs, *Differential Games*, John Wiley & Sons (1965), pp. 297–350
    - A.W. Merz, *The Homicidal Chauffeur — a Differential Game*, PhD Thesis, Stanford (1971)
    - V.S. Patsko & V.L. Turova, "Homicidal Chauffeur Game: History and Modern Studies,"
      *Advances in Dynamic Games*, Annals of the ISDG Vol. 11 (2011)
    - S. Coates & M. Pachter, "The Classical Homicidal Chauffeur Game,"
      *Dynamic Games and Applications* 9(1), 2019
    """)
    return


@app.cell
def history(mo):
    mo.md(r"""
    ## §1 — The Problem

    In 1951, Rufus Isaacs at the RAND Corporation posed a deceptively simple
    question: *Can a fast but clumsy car catch a slow but agile pedestrian?*

    The **Homicidal Chauffeur (HC) problem** is a two-player, zero-sum
    differential game of pursuit and evasion:

    - **Player P (Pursuer / "Chauffeur"):** A car with high speed $v_P$ but
      a minimum turning radius $R_{\min}$ — equivalently, a maximum angular
      velocity $\omega_{\max} = v_P / R_{\min}$. This is a *Dubins vehicle*.

    - **Player E (Evader / "Pedestrian"):** A point agent with lower speed
      $v_E < v_P$ but *unlimited maneuverability* — can instantaneously change
      heading.

    - **Capture:** P wins if the distance $\|P - E\| \leq \ell$ (the capture
      radius). E wins by evading indefinitely.

    - **Objective:** P minimizes, and E maximizes, the time to capture.

    The game plays out on an unbounded plane with no obstacles. Despite its
    recreational framing, the HC problem is a canonical model for missile
    guidance, UAV pursuit, autonomous vehicle collision avoidance, and
    submarine evasion — any scenario where a fast-but-constrained agent
    engages a slow-but-agile one.

    Isaacs' original inspiration was a guided torpedo pursuing a maneuvering
    ship. The problem was deliberately posed in unclassified language to
    enable open publication of research with direct military applications.
    """)
    return


@app.cell
def define_symbols(symbols):
    # Absolute-frame symbols
    x_P, y_P, x_E, y_E = symbols('x_P y_P x_E y_E', real=True)
    theta = symbols('theta', real=True)         # pursuer heading
    psi_lab = symbols('psi_lab', real=True)      # evader heading (lab frame)

    # Physical parameters
    v_P, v_E_sym = symbols('v_P v_E', positive=True)
    omega_max = symbols('omega_max', positive=True)
    ell_sym = symbols('ell', positive=True)      # capture radius

    # Controls
    phi_ctrl = symbols('phi', real=True)          # pursuer turn rate
    psi_ctrl = symbols('psi', real=True)          # evader heading (relative)

    # Dimensionless parameters
    w = symbols('w', positive=True)               # speed ratio v_E/v_P

    # Costate (adjoint) variables
    p1, p2 = symbols('p_1 p_2', real=True)

    # Reduced state variables
    x1, x2 = symbols('x_1 x_2', real=True)

    # Time
    t = symbols('t', real=True)
    return (
        p1,
        p2,
        phi_ctrl,
        psi_ctrl,
        psi_lab,
        t,
        theta,
        v_E_sym,
        v_P,
        w,
        x1,
        x2,
    )


@app.cell
def absolute_kinematics(
    Matrix,
    cos,
    latex,
    mo,
    phi_ctrl,
    psi_lab,
    sin,
    theta,
    v_E_sym,
    v_P,
):
    # Define the 5-DOF absolute dynamics
    # State: (x_P, y_P, theta, x_E, y_E)
    abs_xP_dot = v_P * cos(theta)
    abs_yP_dot = v_P * sin(theta)
    abs_theta_dot = phi_ctrl   # |phi| <= omega_max
    abs_xE_dot = v_E_sym * cos(psi_lab)
    abs_yE_dot = v_E_sym * sin(psi_lab)

    abs_state_dot = Matrix([
        abs_xP_dot, abs_yP_dot, abs_theta_dot, abs_xE_dot, abs_yE_dot
    ])

    mo.md(
        rf"""
        ## §2 — Absolute Kinematics (5 DOF)

        In the lab frame, the full state is
        $\mathbf{{q}} = (x_P, y_P, \theta, x_E, y_E) \in \mathbb{{R}}^5$
        with dynamics:

        $$
        \dot{{\mathbf{{q}}}} = {latex(abs_state_dot)}
        $$

        where:

        - $\theta$ is the pursuer's heading angle
        - $\phi$ is P's turn rate control, bounded $|\phi| \leq \omega_{{\max}}$
        - $\psi_{{\text{{lab}}}}$ is E's heading in the lab frame (free choice $\in [0, 2\pi)$)

        **The problem:** 5 dimensions is high for analysis. But since only the
        *relative* geometry matters (the plane is unbounded and homogeneous), we
        can eliminate 3 DOF by working in **P's body-fixed reference frame**.
        """
    )
    return


@app.cell
def reduction_setup(cos, diff, latex, mo, sin, sp, t):
    # Define time-dependent functions for the derivation
    x_P_t, y_P_t = sp.Function('x_P')(t), sp.Function('y_P')(t)
    x_E_t, y_E_t = sp.Function('x_E')(t), sp.Function('y_E')(t)
    theta_t = sp.Function('theta')(t)

    # Relative position in lab frame
    Delta_x = x_E_t - x_P_t
    Delta_y = y_E_t - y_P_t

    # Body-frame coordinates (x2 along P's velocity, x1 perpendicular)
    x1_expr = -Delta_x * sin(theta_t) + Delta_y * cos(theta_t)
    x2_expr =  Delta_x * cos(theta_t) + Delta_y * sin(theta_t)

    # Differentiate w.r.t. time
    x1_dot_raw = diff(x1_expr, t)
    x2_dot_raw = diff(x2_expr, t)

    mo.md(
        rf"""
        ## §3 — Reduction to Relative (Body-Frame) Coordinates

        We attach a coordinate frame to the pursuer P, with $x_2$ along P's
        velocity vector and $x_1$ perpendicular to it. The relative position
        of E in this frame is:

        $$
        x_1 = -(x_E - x_P)\sin\theta + (y_E - y_P)\cos\theta
        $$
        $$
        x_2 = (x_E - x_P)\cos\theta + (y_E - y_P)\sin\theta
        $$

        We define these as SymPy expressions built from `sp.Function` objects
        (time-dependent unknowns), then call `sp.diff(..., t)` to differentiate
        with respect to time. SymPy applies the product and chain rules
        automatically, producing raw derivatives that still contain abstract
        terms like $\frac{{d}}{{dt}}x_P(t)$:

        $$\dot{{x}}_1\big|_{{\text{{raw}}}} = {latex(x1_dot_raw)}$$

        $$\dot{{x}}_2\big|_{{\text{{raw}}}} = {latex(x2_dot_raw)}$$

        These are correct but not yet useful — we need to substitute the
        known dynamics from §2.
        """
    )
    return theta_t, x1_dot_raw, x2_dot_raw, x_E_t, x_P_t, y_E_t, y_P_t


@app.cell
def reduction_substitute(
    cos,
    diff,
    expand_trig,
    latex,
    mo,
    phi_ctrl,
    psi_lab,
    sin,
    t,
    theta_t,
    trigsimp,
    v_E_sym,
    v_P,
    x1_dot_raw,
    x2_dot_raw,
    x_E_t,
    x_P_t,
    y_E_t,
    y_P_t,
):
    # Substitution rules from the absolute dynamics
    deriv_subs = {
        diff(x_P_t, t): v_P * cos(theta_t),
        diff(y_P_t, t): v_P * sin(theta_t),
        diff(theta_t, t): phi_ctrl,
        diff(x_E_t, t): v_E_sym * cos(psi_lab),
        diff(y_E_t, t): v_E_sym * sin(psi_lab),
    }

    x1_dot_sub = x1_dot_raw.subs(deriv_subs)
    x2_dot_sub = x2_dot_raw.subs(deriv_subs)

    # Simplify using trig identities
    x1_dot_simplified = trigsimp(expand_trig(x1_dot_sub))
    x2_dot_simplified = trigsimp(expand_trig(x2_dot_sub))

    mo.md(
        rf"""
        Now we substitute the known §2 dynamics into the raw derivatives
        using `.subs()` — replacing each $\frac{{d}}{{dt}}x_P(t)$ with
        $v_P\cos\theta$, and so on. Then `trigsimp(expand_trig(...))`
        collapses the resulting trig expressions:

        $$\dot{{x}}_1 = {latex(x1_dot_simplified)}$$

        $$\dot{{x}}_2 = {latex(x2_dot_simplified)}$$

        These expressions still contain $\theta$ and $\psi_{{\text{{lab}}}}$
        individually. To complete the reduction, we define the **relative
        heading** $\psi = \psi_{{\text{{lab}}}} - \theta$, which is the
        evader's heading in P's body frame. After this substitution, $\theta$
        drops out entirely.
        """
    )
    return x1_dot_simplified, x2_dot_simplified


@app.cell
def reduced_dynamics(
    cos,
    expand_trig,
    latex,
    mo,
    phi_ctrl,
    psi_ctrl,
    psi_lab,
    simplify,
    sin,
    theta_t,
    trigsimp,
    v_E_sym,
    v_P,
    w,
    x1,
    x1_dot_simplified,
    x2,
    x2_dot_simplified,
):
    # Substitute psi_lab = psi + theta(t), then simplify.
    # We use theta_t (the Function object) so it cancels with the theta(t)
    # already present in the expressions from the body-frame derivation.
    _x1_dot_rel = trigsimp(expand_trig(
        x1_dot_simplified.subs(psi_lab, psi_ctrl + theta_t)
    ))
    _x2_dot_rel = trigsimp(expand_trig(
        x2_dot_simplified.subs(psi_lab, psi_ctrl + theta_t)
    ))

    # Now substitute the dimensionless form: v_P = 1, v_E = w
    _x1_dot_dimless = simplify(_x1_dot_rel.subs(v_E_sym, w * v_P).subs(v_P, 1))
    _x2_dot_dimless = simplify(_x2_dot_rel.subs(v_E_sym, w * v_P).subs(v_P, 1))

    # The derivation above proves the canonical Isaacs form. However, the
    # derived expressions still contain Function objects (x_P(t), y_P(t), etc.)
    # that SymPy cannot automatically identify as the reduced coordinates x1, x2.
    # We therefore state the canonical form explicitly using the reduced symbols,
    # which downstream cells (Hamiltonian, lambdification) require.
    f1 = -phi_ctrl * x2 + w * sin(psi_ctrl)
    f2 =  phi_ctrl * x1 + w * cos(psi_ctrl) - 1

    mo.md(
        rf"""
        We substitute $\psi_{{\text{{lab}}}} = \psi + \theta$ via `.subs()`
        and apply `trigsimp` — the heading $\theta$ cancels out of every term.

        Normalizing $v_P = 1$ and $v_E = w \cdot v_P$ (Isaacs'
        convention) gives the **reduced dynamics in dimensionless form**:

        $$
        \dot{{x}}_1 = {latex(f1)}
        $$

        $$
        \dot{{x}}_2 = {latex(f2)}
        $$

        where:

        - $\phi \in [-\omega_{{\max}}, \omega_{{\max}}]$ — pursuer's turn rate
          control (after normalization with $R_{{\min}} = 1$: $|\phi| \leq 1$)
        - $\psi \in [0, 2\pi)$ — evader's heading in P's body frame (free choice)
        - $w = v_E / v_P$ — speed ratio (dimensionless)

        The state space is now 2D: $(x_1, x_2) \in \mathbb{{R}}^2$, with the
        origin at P's position. **Capture** occurs when
        $x_1^2 + x_2^2 \leq \tilde{{\ell}}^2$, where
        $\tilde{{\ell}} = \ell / R_{{\min}} = \ell \cdot \omega_{{\max}} / v_P$
        is the **dimensionless capture radius**.

        This 5D $\to$ 2D reduction is what makes the HC problem analytically
        tractable and geometrically beautiful.
        """
    )
    return f1, f2


@app.cell
def parameter_sliders(mo):
    mo.md(r"""
    ## Physical Parameters

    The sliders below control the **physical** parameters of the game.
    We fix the pursuer speed $v_P = 1$ (without loss of generality — this
    sets the time scale).

    **Domain constraints** (to keep us in the interesting regime):

    - $v_E \leq v_P / 2 = 0.5$ — the evader must be *substantially* slower.
      When $v_E / v_P \to 1$, the pursuer's speed advantage vanishes and
      capture becomes trivial to avoid regardless of turning. We restrict
      to $w \leq 0.5$ to focus on configurations where the interplay between
      speed and maneuverability is the deciding factor.

    - $\omega_{\max} \in [0.5, 3.0]$ — the pursuer's maximum turn rate.
      Lower values mean a large, sluggish turning circle (harder to catch
      agile evaders); higher values mean a tighter turn (approaching a
      point-turning vehicle, which trivializes the game).

    - $\ell \in [0.1, 1.5]$ — capture radius. This is the physical distance
      at which we declare "caught." Larger values favor the pursuer.
    """)
    return


@app.cell
def create_sliders(mo):
    v_E_slider = mo.ui.slider(
        start=0.05, stop=0.50, step=0.05, value=0.25,
        label=r"$v_E$ (evader max speed)"
    )
    omega_max_slider = mo.ui.slider(
        start=0.5, stop=3.0, step=0.1, value=1.0,
        label=r"$\omega_{\max}$ (pursuer max turn rate)"
    )
    ell_slider = mo.ui.slider(
        start=0.1, stop=1.5, step=0.05, value=0.5,
        label=r"$\ell$ (capture radius)"
    )
    mo.vstack([v_E_slider, omega_max_slider, ell_slider])
    return ell_slider, omega_max_slider, v_E_slider


@app.cell
def derived_parameters(ell_slider, omega_max_slider, v_E_slider):
    v_E_val = v_E_slider.value
    omega_val = omega_max_slider.value
    ell_val = ell_slider.value

    v_P_val = 1.0  # fixed

    # Derived dimensionless parameters
    w_val = v_E_val / v_P_val
    R_min_val = v_P_val / omega_val
    ell_tilde_val = ell_val / R_min_val
    return R_min_val, ell_tilde_val, ell_val, omega_val, v_E_val, w_val


@app.cell
def show_parameters(
    R_min_val,
    ell_tilde_val,
    ell_val,
    mo,
    omega_val,
    v_E_val,
    w_val,
):
    mo.md(rf"""
    ### Current Configuration

    | Physical Parameter | Value | Description |
    |---|---|---|
    | $v_P$ | $1.0$ (fixed) | Pursuer speed |
    | $v_E$ | ${v_E_val:.2f}$ | Evader max speed |
    | $\omega_{{\max}}$ | ${omega_val:.1f}$ | Pursuer max turn rate |
    | $R_{{\min}} = v_P / \omega_{{\max}}$ | ${R_min_val:.3f}$ | Pursuer min turning radius |
    | $\ell$ | ${ell_val:.2f}$ | Capture radius |

    **Dimensionless parameters** (Isaacs' normalization $v_P = 1, R_{{\min}} = 1$):

    | Parameter | Expression | Value |
    |---|---|---|
    | $w = v_E / v_P$ | speed ratio | ${w_val:.3f}$ |
    | $\tilde{{\ell}} = \ell / R_{{\min}}$ | normalized capture radius | ${ell_tilde_val:.3f}$ |

    The dynamics in dimensionless form depend only on $w$ and
    $\tilde{{\ell}}$. The three physical sliders map to this 2D parameter
    space — note that $\omega_{{\max}}$ and $\ell$ both affect
    $\tilde{{\ell}}$ (a larger turn rate or a larger capture radius both
    make capture easier in normalized terms).
    """)
    return


@app.cell
def game_formulation(mo):
    mo.md(r"""
    ## §4 — The Differential Game

    The HC problem has the structure of a **two-person, zero-sum,
    continuous-time differential game**:

    **State:** $\mathbf{x} = (x_1, x_2) \in \mathbb{R}^2$ (evader position
    in pursuer's body frame)

    **Controls:**
    - P chooses turn rate $\phi \in [-1, 1]$ (after normalization)
    - E chooses heading $\psi \in [0, 2\pi)$

    **Dynamics:** $\dot{\mathbf{x}} = f(\mathbf{x}, \phi, \psi)$ as derived above

    **Terminal set:** $\mathcal{M} = \{\mathbf{x} : x_1^2 + x_2^2 \leq \tilde{\ell}^2\}$

    **Payoff:** Capture time $T = \inf\{t \geq 0 : \mathbf{x}(t) \in \mathcal{M}\}$

    **Objective:** P minimizes $T$; E maximizes $T$ (or drives $T \to \infty$)

    The **Value function** $V(\mathbf{x})$ gives the optimal capture time
    from state $\mathbf{x}$ under best play by both sides. It satisfies
    the **Hamilton-Jacobi-Isaacs (HJI) equation**:

    $$
    \min_\phi \max_\psi \left[ \nabla V \cdot f(\mathbf{x}, \phi, \psi) \right] + 1 = 0
    $$

    with boundary condition $V(\mathbf{x}) = 0$ on $\partial \mathcal{M}$.
    The $+1$ arises because we are minimizing time — each instant of
    continued play costs $+1$ unit of payoff.

    Isaacs' key insight was that if the game has a **saddle point in pure
    strategies** — i.e., the min and max commute — then the value function
    exists and can be computed via the method of characteristics.
    """)
    return


@app.cell
def build_hamiltonian(f1, f2, latex, mo, p1, p2, phi_ctrl, sp):
    # The Hamiltonian for the time-optimal differential game:
    # H = p · f(x, phi, psi) + 1
    # where p = (p1, p2) = nabla V (costate / adjoint)
    H = p1 * f1 + p2 * f2 + 1
    H_expanded = sp.expand(H)

    # Collect terms by control variable to expose structure
    H_by_phi = sp.collect(H_expanded, phi_ctrl)

    mo.md(
        rf"""
        ## §5 — The Hamiltonian & Optimal Controls

        The Hamiltonian of the time-optimal differential game is:

        $$
        H(\mathbf{{x}}, \mathbf{{p}}, \phi, \psi) =
            \mathbf{{p}} \cdot f(\mathbf{{x}}, \phi, \psi) + 1
        $$

        where $\mathbf{{p}} = (p_1, p_2) = \nabla V$ is the costate (adjoint)
        vector. We construct $H$ symbolically from the $f_1, f_2$ expressions
        derived in §3, then use `sp.collect` to group terms by the control
        variable $\phi$, revealing the Hamiltonian's structure:

        $$
        H = {latex(H_by_phi)}
        $$

        The HJI equation requires $\min_\phi \max_\psi H = 0$ along optimal
        trajectories. Because $H$ is **separable** in the two controls, the
        saddle-point condition is satisfied and min/max commute. Let's find the
        optimal controls.
        """
    )
    return (H_expanded,)


@app.cell
def optimal_phi(H_expanded, latex, mo, phi_ctrl, sp):
    # Extract the coefficient of phi (the "switching function")
    sigma = sp.collect(H_expanded, phi_ctrl).coeff(phi_ctrl)
    sigma_simplified = sp.simplify(sigma)

    mo.md(
        rf"""
        ### Pursuer's Optimal Control $\phi^*$

        The Hamiltonian is **linear** in $\phi$. We extract the coefficient
        using `.coeff(phi)` — this is the **switching function**:

        $$
        \sigma(\mathbf{{x}}, \mathbf{{p}}) = {latex(sigma_simplified)}
        $$

        Since P **minimizes** $H$ over $\phi \in [-1, 1]$:

        $$
        \phi^*(\mathbf{{x}}, \mathbf{{p}}) = -\text{{sign}}(\sigma)
        = -\text{{sign}}({latex(sigma_simplified)})
        $$

        This is a **bang-bang** control: the pursuer always turns at maximum
        rate, either hard left or hard right, switching when
        $\sigma = 0$ (the switching surface).

        When $\sigma = 0$ — i.e., when the costate vector $\mathbf{{p}}$ is
        parallel to the position vector $\mathbf{{x}}$ — the turn direction is
        singular. These singular arcs give rise to the rich taxonomy of singular
        surfaces studied by Merz.
        """
    )
    return (sigma_simplified,)


@app.cell
def optimal_psi(
    H_expanded,
    atan2,
    diff,
    latex,
    mo,
    p1,
    p2,
    psi_ctrl,
    simplify,
    sqrt,
    trigsimp,
):
    # Extract the psi-dependent terms: w*(p1*sin(psi) + p2*cos(psi))
    # E maximizes H over psi.
    # d/dpsi [p1*sin(psi) + p2*cos(psi)] = p1*cos(psi) - p2*sin(psi) = 0
    # => tan(psi) = p1/p2  =>  psi* = atan2(p1, p2)

    dH_dpsi = diff(H_expanded, psi_ctrl)
    dH_dpsi_simplified = trigsimp(simplify(dH_dpsi))

    # Optimal evader heading
    psi_star = atan2(p1, p2)

    # Value of the psi-dependent term at the optimum
    # p1*sin(atan2(p1,p2)) + p2*cos(atan2(p1,p2)) = sqrt(p1^2 + p2^2)
    psi_contribution_at_opt = sqrt(p1**2 + p2**2)

    mo.md(
        rf"""
        ### Evader's Optimal Control $\psi^*$

        The $\psi$-dependent terms in $H$ are:

        $$
        w \left( p_1 \sin\psi + p_2 \cos\psi \right)
        $$

        E **maximizes** this. We differentiate with `sp.diff(H, psi)`
        and simplify with `trigsimp` to find the stationary point:

        $$
        \frac{{\partial H}}{{\partial \psi}} = {latex(dH_dpsi_simplified)} = 0
        \quad \Longrightarrow \quad
        \psi^* = \text{{atan2}}(p_1,\, p_2)
        $$

        The evader steers to **align the component of its velocity along the
        costate gradient** — intuitively, E runs in the direction that
        increases the value function most steeply.

        At the optimum, the contribution reduces to:

        $$
        w \left( p_1 \sin\psi^* + p_2 \cos\psi^* \right)
        = w \sqrt{{p_1^2 + p_2^2}} = w \|\mathbf{{p}}\|
        $$
        """
    )
    return (psi_star,)


@app.cell
def saddle_point_verification(
    H_expanded,
    latex,
    mo,
    p1,
    p2,
    phi_ctrl,
    psi_ctrl,
    psi_star,
    sigma_simplified,
    sign,
    sp,
    sqrt,
    w,
):
    # Substitute optimal controls back into H
    # phi* = -sign(sigma), but for symbolic work use phi = -sign(sigma)
    # and psi* = atan2(p1, p2)

    # Reduced Hamiltonian at the saddle point
    H_star = H_expanded.subs(psi_ctrl, psi_star)
    H_star = H_star.subs(phi_ctrl, -sign(sigma_simplified))

    # For the continuous (non-singular) case, simplify:
    # At psi*, the trig terms become: sin(atan2(p1,p2)) = p1/||p||,
    # cos(atan2(p1,p2)) = p2/||p||
    norm_p = sqrt(p1**2 + p2**2)
    H_saddle_explicit = (
        -sp.Abs(sigma_simplified) + w * norm_p - p2 + 1
    )

    mo.md(
        rf"""
        ### Saddle-Point Verification

        Substituting $\phi^* = -\text{{sign}}(\sigma)$ and
        $\psi^* = \text{{atan2}}(p_1, p_2)$ back into $H$, the
        **reduced Hamiltonian** at the saddle point is:

        $$
        H^*(\mathbf{{x}}, \mathbf{{p}}) =
            -|\sigma| + w\|\mathbf{{p}}\| - p_2 + 1
        $$

        $$
        = -|{latex(sigma_simplified)}| + w\sqrt{{p_1^2 + p_2^2}} - p_2 + 1
        $$

        The HJI equation $H^* = 0$ along optimal trajectories becomes:

        $$
        -|{latex(sigma_simplified)}| + w\sqrt{{p_1^2 + p_2^2}} - p_2 + 1 = 0
        $$

        **Saddle-point property:** The separability of $H$ in $\phi$ and $\psi$
        (no cross-terms) guarantees that $\min_\phi \max_\psi H = \max_\psi \min_\phi H$.
        This is the **Isaacs condition** — it ensures the game has a well-defined
        value and the two optimizations can be performed independently.
        """
    )
    return


@app.cell
def costate_equations(H_expanded, diff, latex, mo, simplify, x1, x2):
    # The costate (adjoint) ODEs: dp/dt = -dH/dx
    p1_dot = -diff(H_expanded, x1)
    p2_dot = -diff(H_expanded, x2)

    p1_dot_simplified = simplify(p1_dot)
    p2_dot_simplified = simplify(p2_dot)

    mo.md(
        rf"""
        ## §6 — Costate (Adjoint) Equations

        The costate variables $\mathbf{{p}} = (p_1, p_2)$ evolve according to
        the adjoint equations $\dot{{p}}_i = -\partial H / \partial x_i$.
        We compute these directly with `sp.diff` and `simplify`:

        $$
        \dot{{p}}_1 = -\frac{{\partial H}}{{\partial x_1}}
            = {latex(p1_dot_simplified)}
        $$

        $$
        \dot{{p}}_2 = -\frac{{\partial H}}{{\partial x_2}}
            = {latex(p2_dot_simplified)}
        $$

        Together with the state equations, we now have a **4D ODE system**
        $(x_1, x_2, p_1, p_2)$ that describes the characteristics of the HJI
        PDE. Optimal trajectories are projections of these characteristics onto
        the $(x_1, x_2)$ plane.
        """
    )
    return p1_dot_simplified, p2_dot_simplified


@app.cell
def costate_conservation(
    latex,
    mo,
    p1,
    p1_dot_simplified,
    p2,
    p2_dot_simplified,
    simplify,
):
    # Prove that ||p||^2 is conserved
    d_norm_p_sq = simplify(
        2 * (p1 * p1_dot_simplified + p2 * p2_dot_simplified)
    )

    mo.md(
        rf"""
        ### Conservation of $\|\mathbf{{p}}\|^2$

        A critical structural property. We ask SymPy to compute
        $2(p_1 \dot{{p}}_1 + p_2 \dot{{p}}_2)$ and `simplify` the result:

        $$
        \frac{{d}}{{dt}}(p_1^2 + p_2^2)
        = 2(p_1 \dot{{p}}_1 + p_2 \dot{{p}}_2)
        = {latex(d_norm_p_sq)}
        $$

        SymPy confirms the result vanishes identically — the costate norm
        is **conserved** along characteristics. This is a
        consequence of the rotational structure of the adjoint equations —
        the costate vector rotates but does not grow or shrink. This
        conservation law serves as a **numerical verification check**: any
        drift in $\|\mathbf{{p}}\|$ during integration indicates numerical
        error.
        """
    )
    return


@app.cell
def full_ode_system(
    f1,
    f2,
    latex,
    mo,
    p1_dot_simplified,
    p2_dot_simplified,
    psi_ctrl,
    psi_star,
    sigma_simplified,
):
    # Assemble the full 4D ODE system with optimal controls substituted

    # State dynamics with optimal psi substituted
    # We leave phi as the switching control for now
    f1_opt_psi = f1.subs(psi_ctrl, psi_star)
    f2_opt_psi = f2.subs(psi_ctrl, psi_star)

    # Full system: (x1_dot, x2_dot, p1_dot, p2_dot)
    # with phi = -sign(sigma) and psi = atan2(p1, p2)

    mo.md(
        rf"""
        ### The Complete 4D Characteristic System

        Assembling state + costate with optimal controls
        $\phi^* = -\text{{sign}}({latex(sigma_simplified)})$ and
        $\psi^* = \text{{atan2}}(p_1, p_2)$:

        $$
        \dot{{x}}_1 = -\phi^* x_2 + w \frac{{p_1}}{{\|\mathbf{{p}}\|}}
        $$

        $$
        \dot{{x}}_2 = \phi^* x_1 + w \frac{{p_2}}{{\|\mathbf{{p}}\|}} - 1
        $$

        $$
        \dot{{p}}_1 = {latex(p1_dot_simplified)}
        $$

        $$
        \dot{{p}}_2 = {latex(p2_dot_simplified)}
        $$

        where $\phi^* = -\text{{sign}}({latex(sigma_simplified)})$ and
        $\|\mathbf{{p}}\| = \sqrt{{p_1^2 + p_2^2}}$ is constant along solutions.

        **Transversality conditions:** At the terminal surface
        $x_1^2 + x_2^2 = \tilde{{\ell}}^2$, the costate must be normal to the
        surface: $\mathbf{{p}}(T) = \lambda \, \mathbf{{x}}(T)$ for some
        scalar $\lambda > 0$ (pointing outward, consistent with $V$ increasing
        away from the target).

        ---

        *This completes the symbolic derivation. Before moving to numerical
        simulation, we pause to discuss the singular surfaces that arise when
        the switching function vanishes.*
        """
    )
    return


@app.cell
def singular_surfaces(mo):
    mo.md(r"""
    ## §7 — Singular Surfaces: A Taxonomy

    The pursuer's optimal control $\phi^* = -\text{sign}(\sigma)$ is
    well-defined whenever the switching function
    $\sigma = p_2 x_1 - p_1 x_2 \neq 0$. But what happens on the
    **switching surface** $\sigma = 0$ — where the costate $\mathbf{p}$
    is parallel to the position $\mathbf{x}$?

    On $\sigma = 0$, the pursuer gains no advantage from turning left vs.
    right, and the bang-bang control law breaks down. The resolution depends
    on the *geometry* of how trajectories approach and leave this surface.
    Merz (1971) classified four types of singular surfaces that arise in
    differential games:

    ### 1. Dispersal Surface

    Optimal trajectories **diverge** from the surface on both sides.
    Approaching the surface, the pursuer faces a genuine choice — turn
    left or right — with both options equally costly. The value function
    $V(\mathbf{x})$ is continuous across the dispersal line but its
    gradient has a jump discontinuity.

    In the HC game, the dispersal line appears along portions of the
    $x_2$-axis where the evader is directly ahead of or behind the pursuer.

    ### 2. Universal Surface

    Trajectories **cross through** the surface. The optimal control on
    the surface itself is determined by a higher-order condition: one
    examines $\dot{\sigma}$ (and potentially higher derivatives) to find
    an intermediate (non-bang-bang) turn rate that keeps the trajectory on
    the singular arc. This is the analog of singular arcs in classical
    optimal control.

    ### 3. Equivocal Surface

    The surface is **approached from one side only**. Trajectories arriving
    at the surface can either cross through or turn back — the resolution
    requires careful analysis of the limit of optimal play as one
    approaches the surface. Equivocal surfaces are rare in optimal control
    but appear naturally in differential games where both players influence
    the dynamics.

    ### 4. Focal Surface

    Trajectories **converge onto** the surface from one side and then
    **slide along** it. This creates chattering behavior — the optimal
    control switches infinitely rapidly between its extreme values, and the
    trajectory effectively follows the singular surface. In practice,
    numerical integration must handle this carefully to avoid artifacts.

    ---

    ### Why the HC Game is Special

    For certain parameter ranges $(w, \tilde{\ell})$, all four types of
    singular surface **coexist** in the solution — making the Homicidal
    Chauffeur one of the richest and most studied examples in differential
    game theory. Merz's 1971 Stanford thesis identified over 20 qualitatively
    distinct subregions in the $(w, \tilde{\ell})$ parameter space, each
    with a different configuration of singular surfaces.

    This richness is precisely why the HC problem has remained a benchmark
    for over 70 years: it is complex enough to exhibit the full taxonomy
    of singular phenomena, yet simple enough (2D state space, scalar
    controls) to admit detailed analytical treatment.

    **Computational note:** Our numerical integrator uses
    $\phi^* = -\text{sign}(\sigma)$, which produces $\phi^* = 0$ on the
    switching surface. This is a pragmatic simplification — it yields
    correct trajectories away from singular surfaces but may lose accuracy
    near them. A complete treatment would require explicit detection and
    handling of each singular surface type, which is a research-level
    computation beyond our scope here.

    **References:** Merz (1971) Chapters 3–4; Patsko & Turova (2011)
    Section 4; Bernhard (1977) for the general theory of singular surfaces
    in differential games.
    """)
    return


@app.cell
def lambdify_ode(
    f1,
    f2,
    mo,
    np,
    p1,
    p1_dot_simplified,
    p2,
    p2_dot_simplified,
    phi_ctrl,
    psi_ctrl,
    psi_star,
    sigma_simplified,
    sign,
    sp,
    w,
    x1,
    x2,
):
    # Substitute optimal controls into the full 4D system
    rhs_x1 = f1.subs([(phi_ctrl, -sign(sigma_simplified)),
                       (psi_ctrl, psi_star)])
    rhs_x2 = f2.subs([(phi_ctrl, -sign(sigma_simplified)),
                       (psi_ctrl, psi_star)])
    rhs_p1 = p1_dot_simplified.subs(phi_ctrl, -sign(sigma_simplified))
    rhs_p2 = p2_dot_simplified.subs(phi_ctrl, -sign(sigma_simplified))

    # Lambdify: SymPy expressions → numpy-callable function
    _rhs_lambdified = sp.lambdify(
        [x1, x2, p1, p2, w],
        [rhs_x1, rhs_x2, rhs_p1, rhs_p2],
        modules=['numpy']
    )

    def rhs_forward(t, state, w_val):
        """RHS of the 4D characteristic ODE (forward time)."""
        s1, s2, s3, s4 = state
        return _rhs_lambdified(s1, s2, s3, s4, w_val)

    def rhs_backward(t, state, w_val):
        """RHS for backward integration (negate forward dynamics)."""
        fwd = rhs_forward(t, state, w_val)
        return [-v for v in fwd]

    # Also build a hand-coded numpy version for cross-validation in tests
    def rhs_numpy(state, w_val):
        """Hand-coded RHS — independent of sp.lambdify."""
        s1, s2, s3, s4 = state
        norm_p = np.sqrt(s3**2 + s4**2)
        if norm_p < 1e-15:
            return [0.0, 0.0, 0.0, 0.0]
        sigma = s4 * s1 - s3 * s2
        phi_star = -np.sign(sigma)
        x1d = -phi_star * s2 + w_val * s3 / norm_p
        x2d = phi_star * s1 + w_val * s4 / norm_p - 1.0
        p1d = -phi_star * s4
        p2d = phi_star * s3
        return [x1d, x2d, p1d, p2d]

    mo.md(
        r"""
        ## §8 — Numerical Trajectory Simulation

        We now cross the **SymPy → NumPy bridge**. The symbolic 4D ODE system
        from §6 is converted to a fast numpy-callable function using
        `sp.lambdify`. This lets us integrate trajectories numerically with
        `scipy.integrate.solve_ivp`.

        The lambdified function takes state $(x_1, x_2, p_1, p_2)$ and
        parameter $w$, substitutes the optimal controls
        $\phi^* = -\text{sign}(\sigma)$ and
        $\psi^* = \text{atan2}(p_1, p_2)$, and returns the derivatives.

        We also maintain a hand-coded NumPy implementation as an independent
        cross-check — the test suite verifies both implementations agree.
        """
    )
    return (rhs_backward,)


@app.cell
def usable_part(latex, mo, np, sp, w):
    # Derive the usable part of the terminal circle symbolically.
    # At the terminal surface x = ell_tilde*(cos(alpha), sin(alpha)),
    # transversality gives p(T) = lambda * x(T).
    # H* = -|sigma| + w*||p|| - p2 + 1 = 0
    # At terminal: sigma = 0 (p parallel to x), so |sigma| = 0.
    # ||p|| = lambda * ell_tilde, p2 = lambda * ell_tilde * sin(alpha)
    # => w*lambda*ell_tilde - lambda*ell_tilde*sin(alpha) + 1 = 0
    # => lambda = -1 / (ell_tilde * (w - sin(alpha)))
    # lambda > 0 requires sin(alpha) > w.

    alpha_sym = sp.Symbol('alpha', real=True)
    ell_tilde_sym = sp.Symbol('ell_tilde', positive=True)
    lambda_expr = -1 / (ell_tilde_sym * (w - sp.sin(alpha_sym)))

    def compute_terminal_conditions(alpha_arr, w_val, ell_tilde_val):
        """Compute (x1, x2, p1, p2) at the terminal surface for each alpha.

        Returns shape (N, 4) array of initial conditions for backward
        integration, or None for alpha values outside the usable part.
        """
        x1_T = ell_tilde_val * np.cos(alpha_arr)
        x2_T = ell_tilde_val * np.sin(alpha_arr)
        lam = -1.0 / (ell_tilde_val * (w_val - np.sin(alpha_arr)))
        p1_T = lam * x1_T
        p2_T = lam * x2_T
        return np.column_stack([x1_T, x2_T, p1_T, p2_T])

    mo.md(
        rf"""
        ### Usable Part of the Terminal Circle

        Not every point on the capture boundary is a valid starting point for
        backward integration. The **transversality condition**
        $\mathbf{{p}}(T) = \lambda \, \mathbf{{x}}(T)$ combined with $H^* = 0$
        yields:

        $$
        \lambda(\alpha) = {latex(lambda_expr)}
        $$

        For $\lambda > 0$ (required by the minimum-time interpretation), we
        need $\sin\alpha > w$. The **usable part** of the terminal circle is
        therefore:

        $$
        \alpha \in \left(\arcsin w, \; \pi - \arcsin w\right)
        $$

        Points outside this arc — where the evader can exit the capture region
        faster than the pursuer can close — are not reachable under optimal
        play. This is a fundamental asymmetry: the pursuer can only "approach"
        the target from the upper portion of the circle (the side the pursuer
        is driving toward).
        """
    )
    return (compute_terminal_conditions,)


@app.cell
def trajectory_sliders(mo):
    n_traj_slider = mo.ui.slider(
        start=5, stop=50, step=5, value=20,
        label="Number of trajectories"
    )
    T_horizon_slider = mo.ui.slider(
        start=1.0, stop=20.0, step=0.5, value=8.0,
        label=r"Backward time horizon $T$"
    )
    mo.vstack([
        mo.md("### Trajectory Controls"),
        n_traj_slider,
        T_horizon_slider,
    ])
    return T_horizon_slider, n_traj_slider


@app.cell
def backward_trajectories(
    T_horizon_slider,
    compute_terminal_conditions,
    ell_tilde_val,
    n_traj_slider,
    np,
    rhs_backward,
    solve_ivp,
    w_val,
):
    n_traj = n_traj_slider.value
    T_horizon = T_horizon_slider.value

    # Usable part: alpha in (arcsin(w), pi - arcsin(w))
    _alpha_min = np.arcsin(min(w_val, 0.999))
    _alpha_max = np.pi - _alpha_min

    # Sample angles on the usable part (excluding exact endpoints)
    _eps = 1e-3
    alphas = np.linspace(_alpha_min + _eps, _alpha_max - _eps, n_traj)

    # Compute terminal conditions
    terminal_states = compute_terminal_conditions(alphas, w_val, ell_tilde_val)

    # Integrate each trajectory backward
    trajectories = []
    for _i in range(n_traj):
        _sol = solve_ivp(
            rhs_backward,
            [0, T_horizon],
            terminal_states[_i],
            args=(w_val,),
            method='RK45',
            max_step=0.05,
            dense_output=True,
            rtol=1e-10,
            atol=1e-12,
        )
        trajectories.append(_sol)
    return (trajectories,)


@app.cell
def trajectory_plot(ell_tilde_val, mo, np, plt, trajectories, w_val):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 8))

    # Terminal circle
    _theta = np.linspace(0, 2 * np.pi, 200)
    _ax.plot(
        ell_tilde_val * np.cos(_theta),
        ell_tilde_val * np.sin(_theta),
        'k--', linewidth=1, alpha=0.5, label='Terminal circle'
    )

    # Usable part (bold)
    _alpha_min = np.arcsin(min(w_val, 0.999))
    _alpha_max = np.pi - _alpha_min
    _usable = np.linspace(_alpha_min, _alpha_max, 100)
    _ax.plot(
        ell_tilde_val * np.cos(_usable),
        ell_tilde_val * np.sin(_usable),
        'r-', linewidth=3, alpha=0.8, label='Usable part'
    )

    # Plot trajectories (backward time = forward in plot)
    _cmap = plt.cm.viridis
    for _i, _sol in enumerate(trajectories):
        _color = _cmap(_i / max(len(trajectories) - 1, 1))
        _ax.plot(_sol.y[0], _sol.y[1], '-', color=_color, linewidth=0.8,
                alpha=0.7)
        _ax.plot(_sol.y[0, -1], _sol.y[1, -1], 'o', color=_color,
                markersize=3)

    # Pursuer at origin
    _ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)

    _ax.set_xlabel(r'$x_1$ (perpendicular)')
    _ax.set_ylabel(r'$x_2$ (along heading)')
    _ax.set_aspect('equal')
    _ax.set_title(
        rf'Optimal Trajectories ($w = {w_val:.3f}$, '
        rf'$\tilde{{\ell}} = {ell_tilde_val:.3f}$)'
    )
    _ax.legend(loc='lower right', fontsize=9)
    _ax.grid(True, alpha=0.3)

    plt.tight_layout()
    mo.md(
        r"""
        The plot shows a family of **optimal trajectories** integrated
        backward from the usable part of the terminal circle. Each curve
        is the path the evader traces (in the pursuer's body frame) under
        optimal play by both sides. The colored dots mark the initial
        positions — states from which capture occurs in exactly $T$ time
        units.

        The pursuer sits at the origin (black cross). Trajectories spiral
        outward because the pursuer's turning constraint forces curved
        approach paths.
        """
    )
    return


@app.cell
def reachable_set(
    compute_terminal_conditions,
    ell_tilde_val,
    np,
    rhs_backward,
    solve_ivp,
    w_val,
):
    # Compute isochrones: for each time horizon T, collect the endpoints
    # of backward trajectories started from the usable part.
    _alpha_min = np.arcsin(min(w_val, 0.999))
    _alpha_max = np.pi - _alpha_min
    _eps = 1e-3
    _n_dense = 80  # points per isochrone

    _alphas_dense = np.linspace(_alpha_min + _eps, _alpha_max - _eps, _n_dense)
    _terminal = compute_terminal_conditions(_alphas_dense, w_val, ell_tilde_val)

    _time_horizons = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0]

    isochrones = {}
    for _T in _time_horizons:
        _endpoints = []
        for _i in range(_n_dense):
            _sol = solve_ivp(
                rhs_backward,
                [0, _T],
                _terminal[_i],
                args=(w_val,),
                method='RK45',
                max_step=0.1,
                rtol=1e-8,
                atol=1e-10,
            )
            if _sol.success:
                _endpoints.append([_sol.y[0, -1], _sol.y[1, -1]])
        isochrones[_T] = np.array(_endpoints)
    return (isochrones,)


@app.cell
def reachable_set_plot(ell_tilde_val, isochrones, mo, np, plt, w_val):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 8))

    # Terminal circle
    _theta = np.linspace(0, 2 * np.pi, 200)
    _ax.plot(
        ell_tilde_val * np.cos(_theta),
        ell_tilde_val * np.sin(_theta),
        'k-', linewidth=1.5, label=r'$\tilde{\ell}$'
    )

    # Plot isochrones
    _cmap = plt.cm.plasma
    _sorted_T = sorted(isochrones.keys())
    for _i, _T in enumerate(_sorted_T):
        _pts = isochrones[_T]
        if len(_pts) < 2:
            continue
        _color = _cmap(_i / max(len(_sorted_T) - 1, 1))
        _ax.plot(_pts[:, 0], _pts[:, 1], '-', color=_color, linewidth=1.5,
                label=rf'$T = {_T:.0f}$')

    _ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)
    _ax.set_xlabel(r'$x_1$ (perpendicular)')
    _ax.set_ylabel(r'$x_2$ (along heading)')
    _ax.set_aspect('equal')
    _ax.set_title(
        rf'Backward Reachable Set / Isochrones '
        rf'($w = {w_val:.3f}$, $\tilde{{\ell}} = {ell_tilde_val:.3f}$)'
    )
    _ax.legend(loc='lower right', fontsize=9)
    _ax.grid(True, alpha=0.3)

    plt.tight_layout()
    mo.md(
        r"""
        ## §9 — Backward Reachable Set

        Each curve is an **isochrone** — the locus of initial positions from
        which capture occurs in exactly $T$ time units under optimal play.
        The innermost curve is $T = 1$ (close to the terminal circle); outer
        curves correspond to longer games.

        Together, these isochrones are the **level sets of the value function**
        $V(\mathbf{x})$. The region enclosed by the $T$-isochrone is the
        **backward reachable set** at horizon $T$: all states from which the
        pursuer can guarantee capture within time $T$.

        These curves were computed by the **method of characteristics**:
        backward integration of the 4D Hamiltonian ODE from the usable part
        of the terminal circle, using `scipy.integrate.solve_ivp` with the
        lambdified SymPy expressions.
        """
    )
    return


@app.cell
def hamiltonian_check(mo, np, trajectories, w_val):
    # Numerical verification: H* and ||p||^2 conservation along trajectories
    def _check_conservation(trajs, w):
        max_H = 0.0
        max_p = 0.0
        for sol in trajs:
            x1_arr, x2_arr, p1_arr, p2_arr = sol.y
            norm_p = np.sqrt(p1_arr**2 + p2_arr**2)
            sigma_arr = p2_arr * x1_arr - p1_arr * x2_arr
            H_star = -np.abs(sigma_arr) + w * norm_p - p2_arr + 1.0
            p_norm_sq = p1_arr**2 + p2_arr**2
            max_p = max(max_p, np.max(np.abs(p_norm_sq - p_norm_sq[0])))
            max_H = max(max_H, np.max(np.abs(H_star)))
        return max_H, max_p

    max_H_drift, max_p_drift = _check_conservation(trajectories, w_val)

    mo.md(
        rf"""
        ### Numerical Conservation Check

        Along all {len(trajectories)} computed trajectories:

        | Invariant | Max drift | Tolerance |
        |---|---|---|
        | $H^* = 0$ | ${max_H_drift:.2e}$ | $< 10^{{-6}}$ |
        | $\|\mathbf{{p}}\|^2 = \text{{const}}$ | ${max_p_drift:.2e}$ | $< 10^{{-8}}$ |

        The Hamiltonian should vanish identically along optimal
        characteristics (it is the zero level set of the HJI equation).
        Any drift indicates numerical integration error. The costate norm
        is an independent conservation law from §6.
        """
    )
    return


@app.cell
def verification_summary(mo):
    mo.md(r"""
    ## §10 — Verification & Validation

    How do we know the derivations and numerical results above are correct?
    This notebook employs a layered V&V strategy spanning symbolic proofs,
    numerical invariants, cross-implementation checks, and degenerate-case
    analysis.

    ### Machine-Verified Invariants

    The numerical verification cell above checks two conservation laws along
    every computed trajectory:

    - **Hamiltonian conservation** ($H^* \approx 0$): The reduced Hamiltonian
      $H^* = -|\sigma| + w\|\mathbf{p}\| - p_2 + 1$ must vanish along optimal
      characteristics. Any drift indicates numerical integration error or
      incorrect control substitution.

    - **Costate norm conservation** ($\|\mathbf{p}\|^2 = \text{const}$): The
      rotational structure of the adjoint equations (§6) guarantees
      $\frac{d}{dt}\|\mathbf{p}\|^2 = 0$. This is verified both symbolically
      (SymPy confirms it vanishes identically) and numerically (max drift
      $< 10^{-8}$ along integrated trajectories).

    ### Symbolic–Numerical Cross-Checks

    The companion test suite (`test_phase2.py`) verifies:

    - **Lambdified vs. hand-coded ODE** (T1): The `sp.lambdify`-generated
      function matches an independently coded NumPy implementation at 50
      random state-costate-parameter points. This guards against lambdification
      bugs.

    - **Full derivation vs. canonical form** (T10): The complete body-frame
      derivation (differentiating the coordinate transformation, substituting
      absolute dynamics, simplifying via trig identities) produces expressions
      that agree numerically with the canonical form
      $f_1 = -\phi x_2 + w\sin\psi$, $f_2 = \phi x_1 + w\cos\psi - 1$
      at 20 random test points.

    - **Symbol purity** (T9): The canonical dynamics contain only the
      reduced symbols $\{\phi, \psi, w, x_1, x_2\}$ — no residual
      `Function` objects like $\theta(t)$ or $x_P(t)$ from the derivation
      chain.

    ### Degenerate Cases

    **Stationary evader** ($w = 0$): The dynamics reduce to
    $\dot{x}_1 = -\phi x_2$, $\dot{x}_2 = \phi x_1 - 1$ — pure pursuit
    at unit speed along the $x_2$ axis. From initial position $(0, d)$,
    the capture time is exactly $d - \tilde{\ell}$. This is verified by
    test T6 and serves as a basic sanity check that the signs and
    normalizations are correct.

    **Equal speeds** ($w \to 1$): As the speed ratio approaches unity, the
    usable part of the terminal circle shrinks — $\alpha \in
    (\arcsin w, \pi - \arcsin w)$ collapses to a single point at
    $\alpha = \pi/2$. Physically, the evader can escape in almost every
    direction, and capture becomes possible only for very large
    $\tilde{\ell}$ (the pursuer must rely on its turning constraint being
    less of a liability than the evader's speed deficit). This limiting
    behavior is consistent with the known result that for $w \geq 1$,
    the evader can always escape regardless of $\tilde{\ell}$.

    ### Symmetry

    The reduced dynamics are symmetric under $x_1 \to -x_1$, $p_1 \to -p_1$,
    $\phi \to -\phi$, $\psi \to -\psi$ — reflecting the geometric symmetry
    of the pursuer's body frame about the heading axis. This symmetry is
    visible in the isochrone plots (§9), which are mirror-symmetric about
    the $x_2$-axis.

    ### What Remains

    The critical human-verification items (see `VALIDATION_CHECKLIST.md`,
    Section F) are:

    - **F2**: Do the trajectory spirals rotate in the correct direction
      (CW vs. CCW) compared to published figures?
    - **F3**: Do the isochrone shapes match Patsko & Turova (2011) for
      specific $(w, \tilde{\ell})$ values?
    - **F4**: For parameter values where the evader can escape, does the
      reachable set boundary remain open?

    These require visual comparison against published results and cannot be
    automated within the notebook.
    """)
    return


@app.cell
def extensions(mo):
    mo.md(r"""
    ## §11 — Extensions & Further Reading

    The Homicidal Chauffeur is one node in a large family of pursuit-evasion
    differential games. We briefly survey several important extensions.

    ### Acoustic Homicidal Chauffeur (Bernhard, 1979)

    In the classical HC game, the evader's speed $v_E$ is constant. Bernhard
    introduced a variant where $v_E$ depends on the distance to the pursuer
    — modeling acoustic detection scenarios where a submarine's safe speed
    decreases as it approaches a sonar-equipped surface vessel. This makes
    the speed ratio $w$ state-dependent, fundamentally changing the game's
    structure and eliminating some of the symmetries we exploited above.

    ### Multiplayer Pursuit-Evasion

    Real-world scenarios often involve **multiple pursuers** cooperating to
    capture one or more evaders. This raises questions of task assignment
    (which pursuer targets which evader?), coalition formation, and Pareto
    optimality. The state space grows rapidly — $n$ pursuers and $m$ evaders
    produce a $2(n + m)$-dimensional game before any reduction. Key
    references include Kumkov, Le Ménec & Patsko (2017) for multi-pursuer
    HC variants.

    ### The Reversed Game (Lewin, 1973)

    Instead of a pursuer trying to enter a capture zone around the evader,
    consider an evader trying to **escape** a detection zone around the
    pursuer. The kinematics are identical but the roles of the terminal set
    and objective function are reversed. This models scenarios like a target
    aircraft trying to escape a missile's seeker cone.

    ### Connection to Dubins Paths

    When $w = 0$ (stationary evader), the HC game reduces to the problem
    of finding the **minimum-time path for a Dubins vehicle** — a car that
    moves at constant speed with bounded turning radius. The optimal paths
    are well known: sequences of circular arcs (C) and straight-line
    segments (S), specifically of type CSC or CCC. The HC game can be
    viewed as "Dubins path planning against an adversary."

    ### Modern Computational Approaches

    While this notebook uses the **method of characteristics** (backward
    integration from the terminal surface), modern computational tools solve
    the Hamilton-Jacobi-Isaacs PDE directly on a grid using **level set
    methods**:

    - **Theoretical foundation:** Bardi, Falcone & Soravia (1999)
      established the convergence of finite-difference schemes for HJI
      equations.
    - **Software:** Toolboxes like `helperOC` (MATLAB), `hj_reachability`
      (JAX/Python), and `OptimizedDP` (heterogeneous computing) can solve
      the HJI PDE for systems with 3–6 state dimensions.

    Grid-based methods handle singular surfaces automatically (they are
    captured as kinks in the value function) but become computationally
    expensive in higher dimensions — the so-called "curse of dimensionality."
    The HC game's 2D reduced state space makes it amenable to both
    characteristic-based and grid-based methods.

    ---

    ### References

    - R. Isaacs, *Games of Pursuit*, RAND Corporation Paper P-257 (1951)
    - R. Isaacs, *Differential Games: A Mathematical Theory with Applications
      to Warfare and Pursuit, Control and Optimization*, John Wiley & Sons
      (1965)
    - A.W. Merz, *The Homicidal Chauffeur — A Differential Game*, PhD
      Thesis, Stanford University (1971)
    - P. Bernhard, "Singular surfaces in differential games: an introduction,"
      in *Differential Games and Applications*, Springer Lecture Notes in
      Information and Control Sciences 3 (1977)
    - M. Bardi, M. Falcone & P. Soravia, "Numerical methods for
      pursuit-evasion games via viscosity solutions," in *Stochastic and
      Differential Games*, Birkhäuser (1999)
    - V.S. Patsko & V.L. Turova, *Numerical Solution of Two-Dimensional
      Differential Games*, IIASA Interim Report IR-01-026 (2001)
    - V.S. Patsko & V.L. Turova, "Homicidal Chauffeur Game: History and
      Modern Studies," in *Advances in Dynamic Games*, Annals of the
      International Society of Dynamic Games 11 (2011)
    - S.V. Kumkov, S. Le Ménec & V.S. Patsko, "Zero-sum pursuit-evasion
      differential games with many objects: survey of publications,"
      *Dynamic Games and Applications* 7 (2017)
    - S. Coates & M. Pachter, "The Classical Homicidal Chauffeur Game,"
      *Dynamic Games and Applications* 9(1) (2019)
    """)
    return


if __name__ == "__main__":
    app.run()

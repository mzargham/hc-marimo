"""Render the composite (dispersal ★) chase from talk.py's beat-7 slider cell
as a looping hero GIF for the landing page. The trajectory data is computed
exactly as talk.py's _beat10_reveal cell computes it (same terminal conditions,
same RHS, same solver settings, same lift/stitch); only the rendering differs
(wide hero frame, animated over forward time instead of slider-scrubbed).
"""
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

W = 0.45          # W_FIXED
ELL = 0.5         # ELL_TILDE_FIXED
N = 300           # samples per segment (talk.py _N)

# --- dynamics: phi = -sign(sigma), psi* = atan2(p1, p2) ---------------------
def rhs_forward(tt, s, w):
    x1, x2, p1, p2 = s
    sigma = p2 * x1 - p1 * x2
    phi = -np.sign(sigma)
    n = np.hypot(p1, p2)
    sinpsi, cospsi = p1 / n, p2 / n
    return [-phi * x2 + w * sinpsi,
            phi * x1 + w * cospsi - 1.0,
            -phi * p2,
            phi * p1]

def rhs_backward(tt, s, w):
    return [-v for v in rhs_forward(tt, s, w)]

def compute_terminal_conditions(alpha_arr, w_val, ell_val):
    x1_T = ell_val * np.cos(alpha_arr)
    x2_T = ell_val * np.sin(alpha_arr)
    lam = -1.0 / (ell_val * (w_val - np.sin(alpha_arr)))
    return np.column_stack([x1_T, x2_T, lam * x1_T, lam * x2_T])

def adaptive_tau(sol, t0, t1, n):
    tc = np.linspace(t0, t1, 200)
    sc = sol.sol(tc)
    sig = sc[3] * sc[0] - sc[2] * sc[1]
    wt = 1.0 + 1.0 / (np.abs(sig) + 0.05)
    cdf = cumulative_trapezoid(wt, tc, initial=0)
    cdf /= cdf[-1]
    tau = np.interp(np.linspace(0, 1, n), cdf, tc)
    tau[0], tau[-1] = t0, t1
    return tau

# --- the two crossing characteristics (A at 40 deg, B at 95 deg) ------------
sab = {}
for lbl, a in [("A", np.radians(40.0)), ("B", np.radians(95.0))]:
    ic = compute_terminal_conditions(np.array([a]), W, ELL)[0]
    sab[lbl] = solve_ivp(rhs_backward, [0, 15.0], ic, args=(W,),
                         method="RK45", max_step=0.02, dense_output=True,
                         rtol=1e-12, atol=1e-14)
Nc = 2000
tauA = np.linspace(0, sab["A"].t[-1], Nc)
tauB = np.linspace(0, sab["B"].t[-1], Nc)
sA, sB = sab["A"].sol(tauA), sab["B"].sol(tauB)
mind, ciA, ciB = 1e10, 0, 0
for ti in range(Nc):
    d = np.sqrt((sB[0] - sA[0, ti]) ** 2 + (sB[1] - sA[1, ti]) ** 2)
    j = int(np.argmin(d))
    if d[j] < mind:
        mind, ciA, ciB = d[j], ti, j

# --- lift to lab frame and stitch the composite (verbatim logic) ------------
t1 = adaptive_tau(sab["A"], 0, tauA[ciA], N)
x11, x21, p11, p21 = sab["A"].sol(t1)
phi1 = -np.sign(p21 * x11 - p11 * x21)
th1 = cumulative_trapezoid(-phi1, t1, initial=0)
psiA = np.arctan2(p11[0], p21[0])
XE1 = -W * np.cos(psiA) * t1
YE1 = -W * np.sin(psiA) * t1
c1, s1 = np.cos(th1), np.sin(th1)
XP1 = XE1 - (-x11 * s1 + x21 * c1)
YP1 = YE1 - (x11 * c1 + x21 * s1)
t2 = adaptive_tau(sab["B"], tauB[ciB], sab["B"].t[-1], N)
x12, x22, p12, p22 = sab["B"].sol(t2)
phi2 = -np.sign(p22 * x12 - p12 * x22)
thc = th1[-1]
th2 = thc + cumulative_trapezoid(-phi2, t2, initial=0)
psiB = np.arctan2(p12[0], p22[0]) + thc
dt2 = t2 - t2[0]
XE2 = XE1[-1] - W * np.cos(psiB) * dt2
YE2 = YE1[-1] - W * np.sin(psiB) * dt2
c2, s2 = np.cos(th2), np.sin(th2)
XP2 = XE2 - (-x12 * s2 + x22 * c2)
YP2 = YE2 - (x12 * c2 + x22 * s2)
XP = np.concatenate([XP1, XP2[1:]])[::-1]
YP = np.concatenate([YP1, YP2[1:]])[::-1]
XE = np.concatenate([XE1, XE2[1:]])[::-1]
YE = np.concatenate([YE1, YE2[1:]])[::-1]
TH = np.concatenate([th1, th2[1:]])[::-1]
tauc = np.concatenate([t1, tauA[ciA] + dt2[1:]])
TF = tauc[-1] - tauc[::-1]
XP -= XE[0]; YP -= YE[0]; XE -= XE[0]; YE -= YE[0]
switch_idx = len(XP) - N
print(f"chase: T = {TF[-1]:.2f}, switch at t = {TF[switch_idx]:.2f}, "
      f"final dist = {np.hypot(XP[-1]-XE[-1], YP[-1]-YE[-1]):.3f}")

# --- render -----------------------------------------------------------------
x_all = np.concatenate([XP, XE]); y_all = np.concatenate([YP, YE])
xmid = (x_all.min() + x_all.max()) / 2
ymid = (y_all.min() + y_all.max()) / 2
half_w = (x_all.max() - x_all.min()) / 2 + 0.9
half_h = (y_all.max() - y_all.min()) / 2 + 0.9
ASPECT = 2.5  # width / height of the hero frame
if half_w / half_h < ASPECT:
    half_w = half_h * ASPECT
else:
    half_h = half_w / ASPECT

fig, ax = plt.subplots(figsize=(9.0, 9.0 / ASPECT), dpi=110)
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(xmid - half_w, xmid + half_w)
ax.set_ylim(ymid - half_h, ymid + half_h)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

line_p, = ax.plot([], [], "-", color="#2166ac", lw=1.8, alpha=0.85)
line_e, = ax.plot([], [], "--", color="#b2182b", lw=1.8, alpha=0.85)
start_p, = ax.plot(XP[0], YP[0], "o", color="#2166ac", ms=8, mec="white", mew=0.8)
start_e, = ax.plot(XE[0], YE[0], "o", color="#b2182b", ms=8, mec="white", mew=0.8)
tri = ax.fill([], [], color="#2166ac", ec="black", lw=0.5)[0]
dot_e, = ax.plot([], [], "o", color="#b2182b", ms=7, mec="black", mew=0.5)
circ = np.linspace(0, 2 * np.pi, 100)
cap, = ax.plot([], [], ":", color="gray", lw=1.1, alpha=0.65)
star, = ax.plot([], [], "*", color="#111111", ms=16, mec="white", mew=0.9)
star.set_visible(False)
clock = ax.text(0.985, 0.05, "", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, color="#5a5a5a", family="serif")

FRAMES = 84   # animated frames
HOLD = 18     # extra frames holding the final state
times = np.concatenate([np.linspace(0, TF[-1], FRAMES),
                        np.full(HOLD, TF[-1])])
SZ = 0.45

def draw(k):
    t_now = times[k]
    i = max(0, min(np.searchsorted(TF, t_now, side="right"), len(TF) - 1))
    line_p.set_data(XP[:i + 1], YP[:i + 1])
    line_e.set_data(XE[:i + 1], YE[:i + 1])
    xp, yp, th = XP[i], YP[i], TH[i]
    tri.set_xy(np.column_stack([
        [xp + SZ * np.cos(th), xp + SZ * 0.5 * np.cos(th + 2.4), xp + SZ * 0.5 * np.cos(th - 2.4)],
        [yp + SZ * np.sin(th), yp + SZ * 0.5 * np.sin(th + 2.4), yp + SZ * 0.5 * np.sin(th - 2.4)]]))
    dot_e.set_data([XE[i]], [YE[i]])
    cap.set_data(xp + ELL * np.cos(circ), yp + ELL * np.sin(circ))
    if i >= switch_idx:
        star.set_data([XE[switch_idx]], [YE[switch_idx]])
        star.set_visible(True)
    else:
        star.set_visible(False)
    clock.set_text(f"t = {t_now:4.1f}")
    return line_p, line_e, tri, dot_e, cap, star, clock

anim = FuncAnimation(fig, draw, frames=len(times), blit=True)
OUT = "chase.gif"  # run from site/
anim.save(OUT, writer=PillowWriter(fps=12))
import os
print(f"wrote {OUT}: {os.path.getsize(OUT)/1e6:.2f} MB, "
      f"{len(times)} frames @ 12 fps")

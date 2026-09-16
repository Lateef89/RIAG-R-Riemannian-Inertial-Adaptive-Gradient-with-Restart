import sys, json, os
sys.path.insert(0, "/home/claude/riagr/code")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from riag import Sphere, Stiefel, Grassmann, SPD, run_algorithm
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, load_diabetes

FIGDIR = "/home/claude/riagr/figures"
os.makedirs(FIGDIR, exist_ok=True)

COLORS = {"rgd": "#7a7a7a", "irgm": "#2e5090", "riagr": "#be640a"}
LABELS = {"rgd": "RGD", "irgm": "IRGM (no restart)", "riagr": "RIAG-R"}
STYLES = {"rgd": "--", "irgm": "-.", "riagr": "-"}

R = json.load(open("/home/claude/riagr/results/results.json"))

def plot_convergence_pair(problem, cfg_labels, titles, fname, ylim=None):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for ax, cfg, title in zip(axes, cfg_labels, titles):
        runs = R[problem][cfg]["runs"]
        for variant in ["rgd", "irgm", "riagr"]:
            g = np.array(runs[variant]["gnorm"])
            it = np.arange(1, len(g) + 1)
            ax.plot(it, np.maximum(g, 1e-16), STYLES[variant], color=COLORS[variant],
                    label=LABELS[variant], linewidth=1.4)
        ax.set_yscale("log")
        ax.set_xlabel("iteration $n$")
        ax.set_ylabel(r"$\|\mathrm{grad}\,f(w_n)\|$")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)

plot_convergence_pair("sphere", ["small_wine", "large_digits"],
                       [r"Sphere: Rayleigh quotient (Wine, $n=13$)",
                        r"Sphere: Rayleigh quotient (Digits, $n=64$)"],
                       "fig_sphere_convergence.pdf")

plot_convergence_pair("stiefel", ["small_breastcancer", "large_digits"],
                       [r"Stiefel: PCA (Breast Cancer, $n{=}30,p{=}5$)",
                        r"Stiefel: PCA (Digits, $n{=}64,p{=}10$)"],
                       "fig_stiefel_convergence.pdf")

plot_convergence_pair("spd", ["small_diabetes", "large_breastcancer"],
                       [r"SPD: covariance fitting (Diabetes, $n{=}10$)",
                        r"SPD: covariance fitting (Breast Cancer, $n{=}30$)"],
                       "fig_spd_convergence.pdf")

plot_convergence_pair("grassmann", ["small_wine", "large_digits"],
                       [r"Grassmann: subspace est. (Wine, $n{=}13,p{=}3$)",
                        r"Grassmann: subspace est. (Digits, $n{=}64,p{=}10$)"],
                       "fig_grassmann_convergence.pdf")

# =====================================================================
# Application figures (real-data qualitative results)
# =====================================================================

def standardized_cov(X):
    sd = X.std(axis=0)
    X = X[:, sd > 1e-8]
    X = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    Xs = X / sd
    C = (Xs.T @ Xs) / (Xs.shape[0] - 1)
    return 0.5 * (C + C.T), Xs

# --- Stiefel application: real PCA scatter on the digits dataset ---
digits = load_digits()
Xd, yd = digits.data, digits.target
C, Xs = standardized_cov(Xd)
n = C.shape[0]
p = 10
man = Stiefel(n, p)
rng = np.random.default_rng(11)
x0 = man.rand(rng)
f = lambda X: -float(np.trace(X.T @ C @ X))
eg = lambda X: -2 * C @ X
res = run_algorithm(man, f, eg, x0, variant="riagr", mu=0.9, eta=0.3, eps=1e-8,
                     delta=1e-8, n_iter=800)
Xlearned = res["x_final"]
proj = Xs @ Xlearned[:, :2]
w, V = np.linalg.eigh(C)
order = np.argsort(w)[::-1]
Vtrue = V[:, order[:2]]
proj_true = Xs @ Vtrue

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
sc0 = axes[0].scatter(proj_true[:, 0], proj_true[:, 1], c=yd, cmap="tab10", s=6, alpha=0.75)
axes[0].set_title("Classical eigen-decomposition\n(top-2 eigenvectors of $C$)", fontsize=8.5)
sc1 = axes[1].scatter(proj[:, 0], proj[:, 1], c=yd, cmap="tab10", s=6, alpha=0.75)
axes[1].set_title(f"RIAG-R on $\\mathrm{{St}}(64,10)$\n(first 2 of {p} learned directions)", fontsize=8.5)
for ax in axes:
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(alpha=0.25, linewidth=0.5)
cbar = fig.colorbar(sc1, ax=axes, fraction=0.035, pad=0.02, ticks=range(10))
cbar.set_label("digit class", fontsize=8)
fig.suptitle("Application: low-rank PCA of the UCI handwritten-digits data via Stiefel optimization",
              fontsize=9.5, y=1.06)
fig.savefig(os.path.join(FIGDIR, "fig_stiefel_application.pdf"), bbox_inches="tight")
plt.close(fig)
print("wrote fig_stiefel_application.pdf; principal angle to truth (deg) =",
      np.degrees(man.dist(Xlearned[:, :2] if False else Xlearned, np.hstack([Vtrue, np.zeros((n,p-2))]) ) ) if False else "n/a")

# --- SPD application: correlation-matrix recovery heatmap (breast cancer) ---
bc = load_breast_cancer()
Cbc, _ = standardized_cov(bc.data)
Starget = Cbc + 0.5 * np.eye(Cbc.shape[0])
n = Starget.shape[0]
man_spd = SPD(n)
class SPDIdentityProj(SPD):
    def proj(self, X, G):
        return G
man2 = SPDIdentityProj(n)
X0 = np.eye(n)
f_spd = lambda X: 0.5 * man_spd.dist(X, Starget) ** 2
eg_spd = lambda X: -man_spd.log(X, Starget)
res_spd = run_algorithm(man2, f_spd, eg_spd, X0, variant="riagr", mu=0.9, eta=0.5,
                         eps=1e-8, delta=1e-8, n_iter=60)
# snapshot at a handful of early iterations by re-running with different budgets
snaps = {}
for budget in [5, 15, 60]:
    r = run_algorithm(man2, f_spd, eg_spd, X0, variant="riagr", mu=0.9, eta=0.5,
                       eps=1e-8, delta=1e-8, n_iter=budget)
    snaps[budget] = r["x_final"]

def to_corr(M):
    d = np.sqrt(np.diag(M))
    return M / np.outer(d, d)

fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.6))
vmax = 1.0
im = axes[0].imshow(to_corr(Starget), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes[0].set_title("target $S$\n(real correlation)", fontsize=8)
for ax, b in zip(axes[1:], [5, 15, 60]):
    ax.imshow(to_corr(snaps[b]), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    err = np.linalg.norm(snaps[b] - Starget, 'fro') / np.linalg.norm(Starget, 'fro')
    ax.set_title(f"RIAG-R, $n={b}$\nrel.err={err:.2e}", fontsize=8)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
fig.suptitle("Application: geodesic covariance/correlation fitting on SPD(30) (Breast Cancer data)", fontsize=9.5)
fig.savefig(os.path.join(FIGDIR, "fig_spd_application.pdf"), bbox_inches="tight")
plt.close(fig)
print("wrote fig_spd_application.pdf")

# --- Grassmann application: principal-angle recovery curve + subspace scatter ---
C, Xs = standardized_cov(digits.data)
n, p = C.shape[0], 10
man_g = Grassmann(n, p)
rng = np.random.default_rng(13)
x0 = man_g.rand(rng)
f = lambda X: -float(np.trace(X.T @ C @ X))
eg = lambda X: -2 * C @ X
res_g = run_algorithm(man_g, f, eg, x0, variant="riagr", mu=0.9, eta=0.3, eps=1e-8,
                       delta=1e-8, n_iter=800)
w, V = np.linalg.eigh(C)
order = np.argsort(w)[::-1]
Vtrue_p = V[:, order[:p]]

# recompute principal angle to the TRUE top-p subspace at a few checkpoints
budgets = [10, 50, 150, 400, 800]
angles = []
for b in budgets:
    r = run_algorithm(man_g, f, eg, x0, variant="riagr", mu=0.9, eta=0.3, eps=1e-8,
                       delta=1e-8, n_iter=b)
    Xb = r["x_final"]
    s = np.linalg.svd(Xb.T @ Vtrue_p, compute_uv=False)
    s = np.clip(s, -1, 1)
    ang = np.degrees(np.max(np.arccos(s)))
    angles.append(ang)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
axes[0].plot(budgets, angles, "o-", color=COLORS["riagr"])
axes[0].set_xlabel("iteration budget")
axes[0].set_ylabel("largest principal angle (deg)\nto the true top-10 eigenspace")
axes[0].set_title("Subspace recovery accuracy", fontsize=8.5)
axes[0].grid(alpha=0.25, linewidth=0.5)

Xlearned_g = res_g["x_final"]
proj = Xs @ Xlearned_g[:, :2]
sc = axes[1].scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap="tab10", s=6, alpha=0.75)
axes[1].set_title("Digits projected onto the\nlearned Grassmann subspace", fontsize=8.5)
axes[1].set_xlabel("component 1"); axes[1].set_ylabel("component 2")
axes[1].grid(alpha=0.25, linewidth=0.5)
fig.colorbar(sc, ax=axes[1], fraction=0.045, pad=0.02, ticks=range(10)).set_label("digit class", fontsize=8)
fig.suptitle("Application: principal-subspace estimation on Gr(64,10) (Digits data)", fontsize=9.5, y=1.06)
fig.savefig(os.path.join(FIGDIR, "fig_grassmann_application.pdf"), bbox_inches="tight")
plt.close(fig)
print("wrote fig_grassmann_application.pdf; angles=", angles)

# --- Sphere application: feature-loading bar chart of the trailing eigenvector ---
wine = load_wine()
Cw, _ = standardized_cov(wine.data)
n = Cw.shape[0]
man_s = Sphere()
rng = np.random.default_rng(7)
x0 = man_s.rand(n, rng)
f = lambda x: float(x @ Cw @ x)
eg = lambda x: 2 * Cw @ x
res_s = run_algorithm(man_s, f, eg, x0, variant="riagr", mu=0.9, eta=0.5, eps=1e-8,
                       delta=1e-8, n_iter=800)
xstar = res_s["x_final"]
w, V = np.linalg.eigh(Cw)
xtrue = V[:, 0]
if np.dot(xtrue, xstar) < 0:
    xtrue = -xtrue
feat_names = [n.replace('_', ' ') for n in wine.feature_names]

fig, ax = plt.subplots(figsize=(7.6, 3.0))
idx = np.arange(n)
width = 0.38
ax.bar(idx - width/2, xtrue, width, label="classical trailing eigenvector", color="#7a7a7a")
ax.bar(idx + width/2, xstar, width, label="RIAG-R", color=COLORS["riagr"])
ax.set_xticks(idx)
ax.set_xticklabels(feat_names, rotation=60, ha="right", fontsize=6.5)
ax.set_ylabel("loading")
ax.legend(fontsize=8, frameon=False)
ax.set_title("Application: least-variance (near-collinear) direction of the Wine data, $\\mathbb{S}^{12}$", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_sphere_application.pdf"), bbox_inches="tight")
plt.close(fig)
print("wrote fig_sphere_application.pdf; final f=", res_s["f"][-1] if len(res_s["f"]) else None,
      " true lambda_min=", w[0])

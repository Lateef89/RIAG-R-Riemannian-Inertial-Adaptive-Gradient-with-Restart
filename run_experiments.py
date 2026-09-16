import sys, json, os
sys.path.insert(0, "/home/claude/riagr/code")
import numpy as np
from riag import Sphere, Stiefel, Grassmann, SPD, run_algorithm
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, load_diabetes

OUT = "/home/claude/riagr/results"
os.makedirs(OUT, exist_ok=True)

def drop_constant_columns(X, tol=1e-8):
    sd = X.std(axis=0)
    keep = sd > tol
    return X[:, keep]

def standardized_cov(X):
    X = drop_constant_columns(X)
    X = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    Xs = X / sd
    C = (Xs.T @ Xs) / (Xs.shape[0] - 1)
    C = 0.5 * (C + C.T)
    return C

def correlation_matrix(X):
    C = standardized_cov(X)
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    R = 0.5 * (R + R.T)
    return R

rng_master = np.random.default_rng(42)

N_ITER = 800
DELTA = 1e-8  # matches Algorithm 1's stopping rule; prevents the
              # divide-by-eps numerical blow-up that occurs if iteration
              # is forced to continue after the gradient is already
              # numerically zero
MU = 0.9
ETA = 1.0
EPS = 1e-8

results = {}

def iters_to_tol(gnorm, tol):
    idx = np.argmax(np.asarray(gnorm) < tol) if np.any(np.asarray(gnorm) < tol) else None
    return int(idx) + 1 if idx is not None and gnorm[idx] < tol else None

def run_all_variants(man, f, egrad, x0, seed, n_iter=N_ITER, eta=ETA, mu=MU):
    out = {}
    for variant in ["rgd", "irgm", "riagr"]:
        rng = np.random.default_rng(seed)
        r = run_algorithm(man, f, egrad, x0.copy(), variant=variant,
                           mu=mu, eta=eta, eps=EPS, delta=DELTA, n_iter=n_iter)
        gnorm = r["gnorm"].tolist()
        out[variant] = {
            "f": r["f"].tolist(),
            "gnorm": gnorm,
            "n_iters": r["n_iters"],
            "restarts": int(np.sum(r["restart"])) if len(r["restart"]) else 0,
            "final_gnorm": float(gnorm[-1]) if gnorm else None,
            "iters_to_1e-2": iters_to_tol(gnorm, 1e-2),
            "iters_to_1e-3": iters_to_tol(gnorm, 1e-3),
            "iters_to_1e-4": iters_to_tol(gnorm, 1e-4),
        }
    return out

# =============================================================
# 1. SPHERE: Rayleigh quotient (trailing eigenvector), minimize x^T A x
# =============================================================
print("=== Sphere ===")
sphere_cfgs = {}
for label, loader, eta in [("small_wine", load_wine, 0.5), ("large_digits", load_digits, 0.5)]:
    data = loader()
    Xd = data.data
    A = correlation_matrix(Xd)
    n = A.shape[0]
    man = Sphere()
    rng = np.random.default_rng(7)
    x0 = man.rand(n, rng)
    f = lambda x, A=A: float(x @ A @ x)
    eg = lambda x, A=A: 2 * A @ x
    out = run_all_variants(man, f, eg, x0, seed=123, eta=eta)
    eigvals = np.sort(np.linalg.eigvalsh(A))
    sphere_cfgs[label] = {"n": n, "eigvals": eigvals.tolist(), "runs": out}
    for v in out:
        print(label, v, "final f=", out[v]["f"][-1] if out[v]["f"] else None,
              "final gnorm=", out[v]["final_gnorm"], "iters=", out[v]["n_iters"],
              "restarts=", out[v]["restarts"], "iters@1e-3=", out[v]["iters_to_1e-3"], "iters@1e-4=", out[v]["iters_to_1e-4"])
results["sphere"] = sphere_cfgs

# =============================================================
# 2. STIEFEL: PCA, minimize -trace(X^T C X), p principal directions
# =============================================================
print("=== Stiefel ===")
stiefel_cfgs = {}
for label, loader, p, eta in [("small_breastcancer", load_breast_cancer, 5, 0.3),
                                ("large_digits", load_digits, 10, 0.3)]:
    data = loader()
    Xd = data.data
    C = standardized_cov(Xd)
    n = C.shape[0]
    man = Stiefel(n, p)
    rng = np.random.default_rng(11)
    x0 = man.rand(rng)
    f = lambda X, C=C: -float(np.trace(X.T @ C @ X))
    eg = lambda X, C=C: -2 * C @ X
    out = run_all_variants(man, f, eg, x0, seed=321, eta=eta)
    eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]
    stiefel_cfgs[label] = {"n": n, "p": p, "top_eigvals": eigvals[:p+2].tolist(), "runs": out}
    for v in out:
        print(label, v, "final f=", out[v]["f"][-1] if out[v]["f"] else None,
              "final gnorm=", out[v]["final_gnorm"], "iters=", out[v]["n_iters"],
              "restarts=", out[v]["restarts"], "iters@1e-3=", out[v]["iters_to_1e-3"], "iters@1e-4=", out[v]["iters_to_1e-4"])
results["stiefel"] = stiefel_cfgs

# =============================================================
# 3. SPD: geodesic (Karcher-type) covariance fitting, f(X)=0.5 d(X,S)^2
# =============================================================
print("=== SPD ===")
spd_cfgs = {}
for label, loader, eta in [("small_diabetes", load_diabetes, 0.5),
                             ("large_breastcancer", load_breast_cancer, 0.5)]:
    data = loader()
    Xd = data.data
    Ctarget = standardized_cov(Xd) + 0.5 * np.eye(Xd.shape[1])  # ridge for conditioning
    n = Ctarget.shape[0]
    man = SPD(n)
    rng = np.random.default_rng(5)
    X0 = np.eye(n) * 1.0
    f = lambda X, S=Ctarget, man=man: 0.5 * man.dist(X, S) ** 2
    def eg(X, S=Ctarget, man=man):
        return None  # unused; egrad below is bypassed via custom wrapper
    # SPD needs a special egrad because grad f(X) = -log_X(S) directly (already Riemannian)
    # We wrap run_algorithm's proj() to be identity-passthrough by feeding "egrad" as the
    # true Riemannian gradient and overriding proj for this manifold instance.
    class SPDIdentityProj(SPD):
        def proj(self, X, G):
            return G
    man2 = SPDIdentityProj(n)
    eg2 = lambda X, S=Ctarget, base=SPD(n): -base.log(X, S)
    out = run_all_variants(man2, f, eg2, X0, seed=555, eta=eta)
    spd_cfgs[label] = {"n": n, "runs": out}
    for v in out:
        print(label, v, "final f=", out[v]["f"][-1] if out[v]["f"] else None,
              "final gnorm=", out[v]["final_gnorm"], "iters=", out[v]["n_iters"],
              "restarts=", out[v]["restarts"], "iters@1e-3=", out[v]["iters_to_1e-3"], "iters@1e-4=", out[v]["iters_to_1e-4"])
results["spd"] = spd_cfgs

# =============================================================
# 4. GRASSMANN: principal subspace estimation, minimize -trace(X^T C X)
# =============================================================
print("=== Grassmann ===")
grass_cfgs = {}
for label, loader, p, eta in [("small_wine", load_wine, 3, 0.3),
                                ("large_digits", load_digits, 10, 0.3)]:
    data = loader()
    Xd = data.data
    C = standardized_cov(Xd)
    n = C.shape[0]
    man = Grassmann(n, p)
    rng = np.random.default_rng(13)
    x0 = man.rand(rng)
    f = lambda X, C=C: -float(np.trace(X.T @ C @ X))
    eg = lambda X, C=C: -2 * C @ X
    out = run_all_variants(man, f, eg, x0, seed=999, eta=eta)
    eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]
    grass_cfgs[label] = {"n": n, "p": p, "top_eigvals": eigvals[:p+2].tolist(), "runs": out}
    for v in out:
        print(label, v, "final f=", out[v]["f"][-1] if out[v]["f"] else None,
              "final gnorm=", out[v]["final_gnorm"], "iters=", out[v]["n_iters"],
              "restarts=", out[v]["restarts"], "iters@1e-3=", out[v]["iters_to_1e-3"], "iters@1e-4=", out[v]["iters_to_1e-4"])
results["grassmann"] = grass_cfgs

with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(results, fh)
print("\nSaved to", os.path.join(OUT, "results.json"))

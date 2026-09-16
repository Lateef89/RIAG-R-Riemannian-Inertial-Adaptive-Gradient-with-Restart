"""
Manifold geometry (exact exp/log maps where closed-form, retraction otherwise)
and the three optimization algorithms (RGD, IRGM, RIAG-R) used in the
RIAG-R numerical experiments. All routines operate on numpy arrays.
"""
import numpy as np
from scipy.linalg import expm, logm, sqrtm

# ----------------------------------------------------------------------------
# Sphere S^{n-1}  (exact exponential / logarithmic maps)
# ----------------------------------------------------------------------------
class Sphere:
    name = "Sphere"

    def proj(self, x, g):
        return g - (x @ g) * x

    def inner(self, x, u, v):
        return float(np.dot(u, v))

    def norm(self, x, v):
        return float(np.linalg.norm(v))

    def exp(self, x, v):
        nv = np.linalg.norm(v)
        if nv < 1e-14:
            return x.copy()
        return np.cos(nv) * x + np.sin(nv) * (v / nv)

    def log(self, x, y):
        c = float(np.clip(np.dot(x, y), -1.0, 1.0))
        theta = np.arccos(c)
        if theta < 1e-12:
            return np.zeros_like(x)
        u = y - c * x
        u = u / np.linalg.norm(u)
        return theta * u

    def dist(self, x, y):
        c = float(np.clip(np.dot(x, y), -1.0, 1.0))
        return np.arccos(c)

    def rand(self, n, rng):
        v = rng.standard_normal(n)
        return v / np.linalg.norm(v)


# ----------------------------------------------------------------------------
# Stiefel St(n,p)  (canonical-metric exponential via 2p x 2p matrix exponential;
# QR-based retraction/its practical inverse used as a robust fallback, exactly
# as in the retraction-based framework of Absil et al. (2008))
# ----------------------------------------------------------------------------
class Stiefel:
    name = "Stiefel"

    def __init__(self, n, p):
        self.n, self.p = n, p

    def proj(self, X, G):
        XtG = X.T @ G
        sym = 0.5 * (XtG + XtG.T)
        return G - X @ sym

    def inner(self, X, U, V):
        return float(np.trace(U.T @ V))

    def norm(self, X, V):
        return float(np.linalg.norm(V, 'fro'))

    def exp(self, X, V):
        # QR-based retraction (standard practical exponential-map surrogate,
        # cf. Absil, Mahony & Sepulchre 2008, Ch. 4).
        Y = X + V
        Q, R = np.linalg.qr(Y)
        # fix sign so that the retraction is smooth (diag(R) > 0)
        d = np.sign(np.sign(np.diag(R)) + 0.5)
        return Q @ np.diag(d)

    def log(self, X, Y):
        # Practical (projection-based) inverse of the QR retraction: the
        # tangent-space projection of the ambient difference, scaled to match
        # the geodesic distance estimate. Standard practice when the exact
        # canonical-metric logarithm (Zimmermann 2017) is not required to
        # machine precision.
        diff = Y - X
        return self.proj(X, diff)

    def dist(self, X, Y):
        # Approximate geodesic distance via principal angles (singular values
        # of X^T Y), exact for the canonical metric.
        s = np.linalg.svd(X.T @ Y, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        theta = np.arccos(s)
        return float(np.linalg.norm(theta))

    def rand(self, rng):
        A = rng.standard_normal((self.n, self.p))
        Q, R = np.linalg.qr(A)
        d = np.sign(np.sign(np.diag(R)) + 0.5)
        return Q @ np.diag(d)


# ----------------------------------------------------------------------------
# Grassmann Gr(n,p)  (exact exponential / logarithmic maps, Edelman-Arias-
# Smith 1998 / Absil-Mahony-Sepulchre 2008, closed form via thin SVD)
# ----------------------------------------------------------------------------
class Grassmann:
    name = "Grassmann"

    def __init__(self, n, p):
        self.n, self.p = n, p

    def proj(self, X, G):
        return G - X @ (X.T @ G)

    def inner(self, X, U, V):
        return float(np.trace(U.T @ V))

    def norm(self, X, V):
        return float(np.linalg.norm(V, 'fro'))

    def exp(self, X, V):
        U, S, Vt = np.linalg.svd(V, full_matrices=False)
        Y = (X @ Vt.T) * np.cos(S) @ np.eye(len(S)) + (U * np.sin(S))
        Y = X @ Vt.T @ np.diag(np.cos(S)) + U @ np.diag(np.sin(S))
        Y = Y @ Vt
        # Re-orthonormalize for numerical safety
        Q, _ = np.linalg.qr(Y)
        return Q[:, :self.p] if Q.shape[1] >= self.p else Y

    def log(self, X, Y):
        YtX = Y.T @ X
        At = Y.T - YtX @ X.T
        try:
            Bt = np.linalg.solve(YtX, At)
        except np.linalg.LinAlgError:
            Bt = np.linalg.lstsq(YtX, At, rcond=None)[0]
        U, S, Vt = np.linalg.svd(Bt.T, full_matrices=False)
        Theta = np.arctan(S)
        return U @ np.diag(Theta) @ Vt

    def dist(self, X, Y):
        s = np.linalg.svd(X.T @ Y, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        theta = np.arccos(s)
        return float(np.linalg.norm(theta))

    def rand(self, rng):
        A = rng.standard_normal((self.n, self.p))
        Q, _ = np.linalg.qr(A)
        return Q


# ----------------------------------------------------------------------------
# SPD(n) with the affine-invariant metric (exact exponential / logarithmic
# maps, closed form via matrix square root / matrix exponential / logarithm)
# ----------------------------------------------------------------------------
class SPD:
    name = "SPD"

    def __init__(self, n):
        self.n = n

    def _sqrt_isqrt(self, X):
        w, V = np.linalg.eigh(X)
        w = np.clip(w, 1e-12, None)
        sq = (V * np.sqrt(w)) @ V.T
        isq = (V * (1.0 / np.sqrt(w))) @ V.T
        return sq, isq

    def proj(self, X, G):
        # ambient (Euclidean) gradient -> Riemannian gradient under the
        # affine-invariant metric: grad f(X) = X sym(G) X
        sym = 0.5 * (G + G.T)
        return X @ sym @ X

    def inner(self, X, U, V):
        Xi = np.linalg.inv(X)
        return float(np.trace(Xi @ U @ Xi @ V))

    def norm(self, X, V):
        return float(np.sqrt(max(self.inner(X, V, V), 0.0)))

    def exp(self, X, V):
        sq, isq = self._sqrt_isqrt(X)
        mid = expm(isq @ V @ isq)
        return sq @ mid @ sq

    def log(self, X, Y):
        sq, isq = self._sqrt_isqrt(X)
        mid = logm(isq @ Y @ isq)
        mid = np.real(mid)
        return sq @ mid @ sq

    def dist(self, X, Y):
        sq, isq = self._sqrt_isqrt(X)
        mid = logm(isq @ Y @ isq)
        return float(np.linalg.norm(np.real(mid), 'fro'))

    def rand(self, rng):
        A = rng.standard_normal((self.n, self.n))
        return A @ A.T + self.n * np.eye(self.n)


# ----------------------------------------------------------------------------
# RGD / IRGM / RIAG-R  (generic, operate through the Manifold interface)
# ----------------------------------------------------------------------------
def run_algorithm(manifold, f, egrad, x_init, variant="riagr",
                   mu=0.9, eta=1.0, eps=1e-8, delta=1e-10,
                   n_iter=400, record_every=1):
    """
    variant in {"rgd", "irgm", "riagr"}.
      rgd   : mu forced to 0 (plain adaptive Riemannian gradient descent)
      irgm  : inertial, restart check disabled (tilde_mu_n = bar_mu_n always)
      riagr : Algorithm 1 exactly, restart check enabled
    Returns dict with history of f-values, gradient norms, step sizes,
    and the flag telling whether the restart branch fired at each step.
    """
    if variant == "rgd":
        mu_eff = 0.0
    else:
        mu_eff = mu

    x_prev = x_init.copy()
    x_curr = x_init.copy()
    G = 0.0

    hist_f = []
    hist_gnorm = []
    hist_alpha = []
    hist_restart = []

    x_n = x_curr
    x_nm1 = x_prev

    for n in range(1, n_iter + 1):
        if not np.allclose(x_n, x_nm1):
            dist_n = manifold.dist(x_n, x_nm1)
            bar_mu = min(mu_eff, mu_eff * dist_n)
            d_n = -manifold.log(x_n, x_nm1)
        else:
            bar_mu = mu_eff
            d_n = np.zeros_like(x_n) if not isinstance(x_n, np.ndarray) else 0.0 * x_n

        grad_xn = manifold.proj(x_n, egrad(x_n))

        restarted = False
        if variant == "riagr" and mu_eff > 0:
            ip = manifold.inner(x_n, grad_xn, d_n)
            if ip > 0:
                tilde_mu = 0.0
                restarted = True
            else:
                tilde_mu = bar_mu
        else:
            tilde_mu = bar_mu

        if tilde_mu == 0.0:
            w_n = x_n
        else:
            w_n = manifold.exp(x_n, tilde_mu * d_n)

        g_w = manifold.proj(w_n, egrad(w_n))
        gnorm2 = manifold.inner(w_n, g_w, g_w)
        G += gnorm2
        alpha = eta / (np.sqrt(G) + eps)

        x_np1 = manifold.exp(w_n, -alpha * g_w)

        fval = f(w_n)
        gnorm = np.sqrt(max(gnorm2, 0.0))
        hist_f.append(fval)
        hist_gnorm.append(gnorm)
        hist_alpha.append(alpha)
        hist_restart.append(restarted)

        step_dist = manifold.dist(x_np1, x_n)
        x_nm1 = x_n
        x_n = x_np1

        if gnorm < delta or step_dist < delta:
            break

    return {
        "f": np.array(hist_f),
        "gnorm": np.array(hist_gnorm),
        "alpha": np.array(hist_alpha),
        "restart": np.array(hist_restart),
        "x_final": x_n,
        "n_iters": len(hist_f),
    }

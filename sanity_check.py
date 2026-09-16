import numpy as np, sys
sys.path.insert(0, "/home/claude/riagr/code")
from riag import Sphere, Stiefel, Grassmann, SPD

rng = np.random.default_rng(0)

def check_exp_log(man, x, make_tangent, label, tol=1e-6):
    v = make_tangent(x)
    v = v * 0.3 / (np.linalg.norm(v) + 1e-12)  # small step, well inside injectivity radius
    y = man.exp(x, v)
    v2 = man.log(x, y)
    err = np.linalg.norm(v - v2) / (np.linalg.norm(v) + 1e-12)
    d1 = man.dist(x, y)
    d2 = man.norm(x, v)
    print(f"{label}: log(exp(v))-v rel err = {err:.3e}, dist vs |v|: {d1:.5f} vs {d2:.5f}")

# Sphere
S = Sphere()
x = S.rand(8, rng)
def tan_sphere(x):
    v = rng.standard_normal(8)
    return v - np.dot(v, x) * x
check_exp_log(S, x, tan_sphere, "Sphere")

# Grassmann
G = Grassmann(10, 3)
x = G.rand(rng)
def tan_grass(x):
    v = rng.standard_normal((10, 3))
    return v - x @ (x.T @ v)
check_exp_log(G, x, tan_grass, "Grassmann")

# SPD
P = SPD(5)
x = P.rand(rng)
def tan_spd(x):
    v = rng.standard_normal((5, 5))
    return 0.5 * (v + v.T)
check_exp_log(P, x, tan_spd, "SPD")

# Stiefel (retraction-based; exp(log(.)) roundtrip approximate only)
St = Stiefel(10, 3)
x = St.rand(rng)
def tan_stiefel(x):
    v = rng.standard_normal((10, 3))
    return St.proj(x, v)
check_exp_log(St, x, tan_stiefel, "Stiefel (retraction)")

print("\n--- gradient finite-difference checks ---")

def fd_grad_check(man, x, f, egrad, make_tangent, label, h=1e-6):
    v = make_tangent(x)
    v = v / (man.norm(x, v) + 1e-12)
    g = man.proj(x, egrad(x))
    directional = man.inner(x, g, v)
    y1 = man.exp(x, h * v)
    y2 = man.exp(x, -h * v)
    fd = (f(y1) - f(y2)) / (2 * h)
    print(f"{label}: grad-dir={directional:.6f}  finite-diff={fd:.6f}  rel err={abs(directional-fd)/(abs(fd)+1e-12):.3e}")

# Sphere Rayleigh quotient
n = 8
A = rng.standard_normal((n, n)); A = A @ A.T
S = Sphere()
x = S.rand(n, rng)
f_sph = lambda x: float(x @ A @ x)
eg_sph = lambda x: 2 * A @ x
fd_grad_check(S, x, f_sph, eg_sph, tan_sphere, "Sphere Rayleigh")

# Stiefel PCA
n, p = 10, 3
C = rng.standard_normal((n, n)); C = C @ C.T
St = Stiefel(n, p)
X = St.rand(rng)
f_st = lambda X: -float(np.trace(X.T @ C @ X))
eg_st = lambda X: -2 * C @ X
fd_grad_check(St, X, f_st, eg_st, tan_stiefel, "Stiefel PCA")

# Grassmann subspace
n, p = 10, 3
Gm = Grassmann(n, p)
X = Gm.rand(rng)
f_gr = lambda X: -float(np.trace(X.T @ C @ X))
eg_gr = lambda X: -2 * C @ X
fd_grad_check(Gm, X, f_gr, eg_gr, tan_grass, "Grassmann subspace")

# SPD Karcher-type
n = 5
Pm = SPD(n)
Xspd = Pm.rand(rng)
Starget = Pm.rand(rng)
f_spd = lambda X: 0.5 * Pm.dist(X, Starget) ** 2
def eg_spd(X):
    # Euclidean "gradient" placeholder unused; we directly define Riemannian grad
    return None
# For SPD we test the known closed form grad f(X) = -log_X(Starget) directly
v = tan_spd(Xspd)
v = v / (Pm.norm(Xspd, v) + 1e-12)
g = -Pm.log(Xspd, Starget)
directional = Pm.inner(Xspd, g, v)
h = 1e-6
y1 = Pm.exp(Xspd, h * v); y2 = Pm.exp(Xspd, -h * v)
fd = (f_spd(y1) - f_spd(y2)) / (2 * h)
print(f"SPD Karcher: grad-dir={directional:.6f}  finite-diff={fd:.6f}  rel err={abs(directional-fd)/(abs(fd)+1e-12):.3e}")

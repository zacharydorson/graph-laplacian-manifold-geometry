import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from laplace import estimate_dim, tangents

DIM_COLORS = ["#2f2f2f", "#276fbf", "#2ca25f", "#f28e2b", "#b24aa0"]

# synthetic data samplers

def sample_square(n, boundary_fraction=0.05, seed=21):
    rng = np.random.default_rng(seed)
    nb = int(round(n * boundary_fraction))
    interior = rng.uniform(-1, 1, size=(n - nb, 2))
    edge, t = rng.integers(0, 4, size=nb), rng.uniform(-1, 1, size=nb)
    boundary = np.empty((nb, 2))
    boundary[edge == 0] = np.column_stack((t[edge == 0], np.full((edge == 0).sum(), -1)))
    boundary[edge == 1] = np.column_stack((t[edge == 1], np.full((edge == 1).sum(), 1)))
    boundary[edge == 2] = np.column_stack((np.full((edge == 2).sum(), -1), t[edge == 2]))
    boundary[edge == 3] = np.column_stack((np.full((edge == 3).sum(), 1), t[edge == 3]))
    labels = np.r_[np.zeros(n - nb, dtype=bool), np.ones(nb, dtype=bool)]
    order = rng.permutation(n)
    return np.vstack((interior, boundary))[order], labels[order]

def sample_swiss_roll(n, boundary_fraction=0.15, seed=12):
    rng = np.random.default_rng(seed)
    nb = int(round(n * boundary_fraction))
    t_min, t_max, height = 1.25 * np.pi, 5.75 * np.pi, 7.0
    t, y = rng.uniform(t_min, t_max, n - nb), rng.uniform(-height / 2, height / 2, n - nb)
    edge, tb, yb = rng.integers(0, 4, size=nb), rng.uniform(t_min, t_max, nb), rng.uniform(-height / 2, height / 2, nb)
    tb[edge == 0], tb[edge == 1], yb[edge == 2], yb[edge == 3] = t_min, t_max, -height / 2, height / 2
    t, y = np.concatenate((t, tb)), np.concatenate((y, yb))
    points = np.column_stack((t * np.cos(t), 1.35 * y, t * np.sin(t)))
    points -= points.mean(axis=0, keepdims=True)
    points /= np.linalg.norm(points.std(axis=0), ord=np.inf)
    labels = np.r_[np.zeros(n - nb, dtype=bool), np.ones(nb, dtype=bool)]
    order = rng.permutation(n)
    return points[order], labels[order]

def sample_torus(n=900, R=2.0, r=0.65, seed=4):
    rng = np.random.default_rng(seed)
    theta, phi = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    return np.column_stack(((R + r * np.cos(phi)) * np.cos(theta), (R + r * np.cos(phi)) * np.sin(theta), r * np.sin(phi)))

def sample_one_sheet_hyperboloid(n=900, waist_radius=1.05, height_scale=0.85, z_max=1.35, seed=11):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-z_max, z_max, n)
    radius = waist_radius * np.sqrt(1.0 + (z / height_scale) ** 2)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack((x, y, z))

def sample_sphere(n=1000, seed=13):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 3))
    return points / np.linalg.norm(points, axis=1, keepdims=True)

# plotting functions

def plot_tangent_plane_collection(points, tangent_bundle, plane_count=400, square_scale=0.023, elev=24, azim=-48):
    chosen = np.empty(plane_count, dtype=int)
    chosen[0] = np.argmin(((points - points.mean(axis=0)) ** 2).sum(axis=1))
    nearest = np.full(len(points), np.inf)
    for i in range(1, plane_count):
        nearest = np.minimum(nearest, ((points - points[chosen[i - 1]]) ** 2).sum(axis=1))
        chosen[i] = np.argmax(nearest)

    h = square_scale * 0.5 * np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    vectors = tangent_bundle[:, :, :2]
    patches = [np.array([p - h * v[:, 0] - h * v[:, 1], p + h * v[:, 0] - h * v[:, 1], p + h * v[:, 0] + h * v[:, 1], p - h * v[:, 0] + h * v[:, 1]]) for p, v in zip(points[chosen], vectors[chosen])]

    z = points[chosen, 2]
    w = (z - z.min()) / np.ptp(z)
    low, high = np.array(to_rgb("#de2d26")), np.array(to_rgb("#2ca25f"))

    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(patches, facecolors=[tuple((1 - x) * low + x * high) for x in w], edgecolors="none", linewidths=0))
    c = 0.5 * (points.min(axis=0) + points.max(axis=0))
    r = 0.5 * np.max(points.max(axis=0) - points.min(axis=0)) * 1.03
    ax.set(xlim=(c[0] - r, c[0] + r), ylim=(c[1] - r, c[1] + r), zlim=(c[2] - r, c[2] + r))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax

def plot_dims(ax, points, dims, title, point_size, elev=None, azim=None):
    dims = np.rint(dims).astype(int)
    sc = ax.scatter(*points.T, c=dims, cmap=ListedColormap(DIM_COLORS), norm=BoundaryNorm(np.arange(-0.5, 5.5), len(DIM_COLORS)), s=point_size, linewidths=0, alpha=0.9)
    ax.set_title(title)
    if points.shape[1] == 2:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect(np.ptp(points, axis=0))
    ax.figure.colorbar(sc, ax=ax, ticks=np.arange(5), fraction=0.046, pad=0.04).set_label("pointwise dimension estimate")

def score(dims, boundary):
    dims = np.rint(dims).astype(int)
    interior = ~boundary
    interior_acc, boundary_acc = np.mean(dims[interior] == 2), np.mean(dims[boundary] == 1)
    values, counts = np.unique(dims, return_counts=True)
    return {"score": float(0.5 * interior_acc + 0.5 * boundary_acc), "interior_acc": float(interior_acc), "boundary_acc": float(boundary_acc), "median_dim": int(np.rint(np.median(dims))), "hist": dict(zip(values.tolist(), counts.tolist()))}

def noise_row(points, sigma, rep, n0=52):
    noisy = points + np.random.default_rng(1000 + rep).normal(scale=sigma, size=points.shape)
    dims = estimate_dim(noisy, n0)
    tangent_bundle, _, _ = tangents(noisy, n0)
    normals = tangent_bundle[:, :, 2]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.abs((normals * points).sum(axis=1)), -1, 1)))
    return np.mean(dims == 2), np.median(angles)

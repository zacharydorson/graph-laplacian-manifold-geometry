import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

# sample points from S^2
# https://mathworld.wolfram.com/SpherePointPicking.html
# pick u, v uniformly in (0,1) and then do \theta=2\pi u, \phi = acos(2v-1).
# then x,y,z = sin\phi cos\theta, sin\phi sin\theta, cos\phi .

def sample_sphere(n, r=1):
  rng = np.random.default_rng()

  samples = rng.uniform(0, 1, (n, 2))

  theta = 2 * np.pi * samples[:, 0]
  phi = np.acos(2 * samples[:, 1]-1)

  x = np.sin(phi) * np.cos(theta)
  y = np.sin(phi) * np.sin(theta)
  z = np.cos(phi)

  return np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), axis=1)

# compute discrete approximation to the Laplace operator
# we skip parts of the construction concerning alpha-normalization because we are assuming uniform sampling density.

def graph_laplacian(P, k=16, n0=60):
  n = P.shape[0]
  
  k = np.min((k, n))
  
  neighbors = NearestNeighbors(n_neighbors=k).fit(P)
  d, indices = neighbors.kneighbors(P)
  d = d**2

  rho0 = np.sqrt((d[:,1:8]).mean(axis=1)) # initial variable bandwidth estimate
  d0 = d / (rho0[indices] * rho0.reshape(-1,1))

  epsilons = 2**np.arange(-30,10,0.25) # test bandwidth values

  K0 = np.exp(-d0[:, :, np.newaxis] / (2 * epsilons[np.newaxis, np.newaxis, :]))

  K0_mean = np.mean(K0, axis=(0, 1)) # mean of the kernel matrices for each epsilon.

  c0 = np.diff(np.log(K0_mean)) / np.diff(np.log(epsilons)) # this is a heuristic for choosing \epsilon
                                                            # see https://web.math.princeton.edu/~amits/publications/2Dtomography_published.pdf, https://icml.cc/Conferences/2005/proceedings/papers/037_Intrinsic_HeinAudibert.pdf
  ix0 = np.argmax(c0)
  e0 = epsilons[ix0]
  dim0 = 2 * c0[ix0]

  K1_nz = np.exp(-(d0 / (2 * e0)))

  K1 = np.zeros((n,n))

  for i, x in enumerate(K1_nz): # make K1 dense (n,n) to symmetrize
     K1[i, indices[i]] = x

  K1 = (K1 + K1.T) / 2 

  qest = K1.sum(axis=1) / (n * rho0**dim0) # compute sampling density estimate for variable bandwidth kernel

  rho = qest ** (-1/2) # compute rho with qest, the sampling density estimate
  rho /= np.median(rho) 

  weighted_distances = d / (rho[indices] * rho.reshape(-1, 1))

  K2 = np.exp(-weighted_distances[:, :, np.newaxis] / (4 * epsilons[np.newaxis, np.newaxis, :])) # use new weighted distance matrix to recompute a good choice of \epsilon
  K2_mean = np.mean(K2, axis=(0, 1)) 

  c = np.diff(np.log(K2_mean)) / np.diff(np.log(epsilons))

  ix = np.argmax(c)
  e = epsilons[ix]
  dim = 2 * c[ix]

  Ks = np.exp(-(weighted_distances / (4 * e)))  # compute the final kernel matrix with chosen epsilon. while the theoretical treatment assumes we compute k(x_i, x_j) for each (i,j),
                                          # for appropriately chosen k the distances omitted from the knn distance matrix are numerically insignificant, so we may discard them
                                          # to avoid computing |P|^2 norms

  K = np.zeros((n, n))
  for i, (idx, x) in enumerate(zip(indices, Ks)):
    K[i, idx] = x

  K = (K + K.T) / 2 # symmetrize

  D = np.diag(1 / np.sum(K, axis=1))

  L = (np.diag(1 / rho**2) @ (np.eye(n) - D @ K)) / e

  sample_density = K.sum(axis=1)
  S = np.diag(sample_density ** (-1/2))
  K_sym = S @ K @ S 

  _, u = eigsh(K_sym, n0, which='LA')

  u = (S @ u)[:, ::-1].T

  return L, u, sample_density, dim

# now we may use the above to compute a basis for the tangent space at each point:
# the riemannian metric is the restriction of the euclidean inner product to the tangent space
# we have by the product rule L(fg) = fLg + gLf + 2(grad f, grad g), which gives:
# (1): (grad g, grad f) = 1/2 (L(fg) - fLg - gLf)
# so we may write the inner product (grad f, grad g) in terms of the laplacian of f and g
#
# now say we want to compute the dimension of the tangent space. if A is a matrix with columns
# forming a basis for the tangent space, we can look at rank A, or equivalently rank A^T A.
# but if f_i is the ith coordinate function on R^n, then grad f_i = e_i, so the projection
# of all the grad f_i's onto T_p M is a basis for the tangent space.
# then if our A is comprised of those, A^T A is the gram matrix G_{ij}=(grad f_i, grad f_j).
# so, use the product rule to compute each of those elements, then diagonalize the matrix
# and read off a basis for the tangent space.

def tangents(X_input, n0):
  L, u, sample_density, dim = graph_laplacian(X_input, k=n0)

  n = X_input.shape[0]

  X = u.T @ u*sample_density @ X_input # bandlimit input data

  # we write the above expression for g(grad f, grad g) in terms of the data:
  # 1/2( L(x_i x_j) - x_i Lx_j - x_j Lx_i )
  # we can construct the products x_i Lx_j by reshaping and broadcasting:

  xiLxj = X.reshape(n, -1, 1) * (L @ X).reshape(n, 1, -1)  # L @ (n, 1, -1) gives (n, 1, -1) with L(x_i) as its ith row. then elementwise multiply

  xixj = X.reshape(n, -1, 1) * X.reshape(n, 1, -1)          # construct the products x_i x_j
  Lxixj = (L @ xixj.reshape(n, -1)).reshape(xixj.shape)     # we reshape from (n, d, d) to (n, d^2), and then do L @ (n, d^2) so that
                                                            # the resulting (n, d^2) has columns corresponding to L(x_i * x_j). then reshape back

  G_full = (-1/2) * (Lxixj - xiLxj - xiLxj.transpose((0, 2, 1)))  # find xjLxi by transposing the last 2 dimensions of xiLxj, and construct the approximation to the metric at each point

  G = (u.T @ u*sample_density @ G_full.reshape(n, -1)).reshape(G_full.shape) # weight with the sample density

  tangent_bundle = np.zeros(G.shape)
  eigenvalues = np.zeros((G.shape[0], G.shape[1]))

  for i in range(n):
    e, u = np.linalg.eigh(G[i])
    eigenvalues[i] = e[::-1]
    tangent_bundle[i] = u[:, ::-1]

  return tangent_bundle, eigenvalues, L

def estimate_dim(X, n0):
  tangent_bundles, eigenvalues, L = tangents(X, n0)

  scaled_eigenvalues = eigenvalues / np.abs(np.median(eigenvalues[:, 0])) # we want the eigenvalues to be near 1 for this dimension estimate

  differences = np.diff(scaled_eigenvalues, prepend=1, append=0)
  pointwise_est = np.argmin(differences, axis=1)

  return pointwise_est

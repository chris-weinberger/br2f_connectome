"""
vb_wsbm.py
==========

Variational-Bayes weighted stochastic block model (WSBM), following the model of

    Aicher, Jacobs & Clauset (2015), "Learning latent block structure in
    weighted networks", *Journal of Complex Networks* 3(2):221-248.

Where ``graph-tool`` (used in ``hippocampal_sbm.ipynb``) infers the block count by
MCMC + minimum description length, this module instead fits a **fixed** number of
blocks ``K`` by **mean-field variational Bayes** and lets you choose ``K`` by the
variational log-evidence (the ELBO), exactly as Aicher et al. recommend.

Model
-----
Each ordered pair ``(i, j)`` (i != j, restricted to a set of *modeled* pairs) is
generated in two exponential-family parts, balanced by ``alpha`` in ``[0, 1]``:

* **Edge existence** ``A_ij ~ Bernoulli(theta_{z_i, z_j})``          (weight ``alpha``)
* **Edge weight**    ``W_ij | A_ij=1 ~ Exponential(lambda_{z_i, z_j})`` (weight ``1 - alpha``)

with conjugate priors ``theta ~ Beta(a0, b0)``, ``lambda ~ Gamma(c0, d0)`` (shape,
rate) and block proportions ``pi ~ Dirichlet(gamma0)``. The Exponential weight model
mirrors graph-tool's ``real-exponential`` choice, which suits the positive, skewed
log-transformed connection weights.

The mean-field posterior factorizes as ``q(z) prod q(theta) q(lambda) q(pi)`` with
``q(z_i)`` a categorical responsibility vector ``mu_i`` over the ``K`` blocks. We
alternate closed-form coordinate-ascent updates (VB-EM); the ELBO is non-decreasing
and is returned for convergence checking and model selection.

Usage
-----
    res = fit_vb_wsbm(W, K=5, alpha=0.5, n_init=10, seed=0)
    z   = res['z']            # hard block assignment (argmax responsibility)
    lb  = res['elbo']         # variational log-evidence for this K

    # model selection: pick K with the best ELBO
    scan = select_K(W, range(1, 11))
    bestK = scan['best_K']

Notes
-----
* ``M`` (modeled-pairs mask) lets the *same* core fit a true bipartite model (only
  cross-type pairs are modeled) and the full directed model (all off-diagonal pairs).
* ``allowed`` (per-node block mask) implements the bipartite ``pclabel`` constraint:
  restrict the two node types to disjoint block ranges. See ``bipartite_allowed``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# KL divergences between the conjugate posteriors and their priors (for the ELBO)
# ---------------------------------------------------------------------------
def _beta_kl(a, b, a0, b0):
    # KL( Beta(a, b) || Beta(a0, b0) ), elementwise on arrays.
    lnB  = gammaln(a) + gammaln(b) - gammaln(a + b)
    lnB0 = gammaln(a0) + gammaln(b0) - gammaln(a0 + b0)
    return (lnB0 - lnB
            + (a - a0) * digamma(a)
            + (b - b0) * digamma(b)
            + (a0 - a + b0 - b) * digamma(a + b))


def _gamma_kl(a, b, a0, b0):
    # KL( Gamma(shape=a, rate=b) || Gamma(shape=a0, rate=b0) ), elementwise.
    return ((a - a0) * digamma(a) - gammaln(a) + gammaln(a0)
            + a0 * (np.log(b) - np.log(b0)) + a * (b0 - b) / b)


def _dirichlet_kl(g, g0):
    # KL( Dirichlet(g) || Dirichlet(g0) ) for 1-D arrays.
    g0 = np.broadcast_to(g0, g.shape)
    return (gammaln(g.sum()) - np.sum(gammaln(g))
            - gammaln(g0.sum()) + np.sum(gammaln(g0))
            + np.sum((g - g0) * (digamma(g) - digamma(g.sum()))))


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def bipartite_allowed(node_types, Kh, Kp):
    """Per-node block mask for a bipartite fit.

    Type-0 nodes may occupy blocks ``[0, Kh)``; type-1 nodes ``[Kh, Kh + Kp)``.
    Returns a ``(N, Kh + Kp)`` boolean array (the ``pclabel`` analogue).
    """
    node_types = np.asarray(node_types)
    N = node_types.size
    K = Kh + Kp
    allowed = np.zeros((N, K), bool)
    allowed[node_types == 0, :Kh] = True
    allowed[node_types == 1, Kh:] = True
    return allowed


def cross_type_pairs(node_types):
    """Modeled-pairs mask ``M`` keeping only ordered pairs between different types
    (the bipartite structure): ``M_ij = 1`` iff ``type_i != type_j``."""
    node_types = np.asarray(node_types)
    M = (node_types[:, None] != node_types[None, :]).astype(float)
    np.fill_diagonal(M, 0.0)
    return M


# ---------------------------------------------------------------------------
# Core single-run VB-EM
# ---------------------------------------------------------------------------
def _peaky(labels, allowed):
    # Peaky responsibilities from a hard label vector (respects the block mask).
    N, K = allowed.shape
    mu = np.full((N, K), 1e-3) * allowed
    mu[np.arange(N), labels] += 1.0
    mu /= mu.sum(1, keepdims=True)
    return mu


def _kmeans_labels(features, allowed, seed):
    # Warm-start labels from k-means on the symmetrized connectivity features,
    # clustered separately within each distinct allowed-block pattern so the
    # bipartite (pclabel) constraint is respected.
    N = allowed.shape[0]
    labels = np.zeros(N, int)
    patterns = {}
    for i in range(N):
        patterns.setdefault(tuple(np.where(allowed[i])[0]), []).append(i)
    for key, idx in patterns.items():
        idx = np.array(idx)
        blocks = np.array(key)
        kg = min(len(blocks), len(idx))
        if kg <= 1:
            labels[idx] = blocks[0]
        else:
            lab = KMeans(kg, n_init=5, random_state=seed).fit_predict(features[idx])
            labels[idx] = blocks[lab]
    return labels


def _random_labels(allowed, rng):
    # Random hard label per node drawn from its allowed blocks.
    return np.array([rng.choice(np.where(row)[0]) for row in allowed])


def _fit_once(W, A, M, K, allowed, we, ww, priors, max_iter, tol, mu):
    # we, ww: weights on the edge-existence and edge-weight log-likelihood terms.
    a0, b0, c0, d0, gamma0 = priors
    logmask = np.where(allowed, 0.0, -np.inf)

    elbos = []
    prev = -np.inf
    for _ in range(max_iter):
        # ---- M-step: conjugate posteriors from current responsibilities ----
        n  = mu.sum(0)                       # expected block sizes         (K,)
        P  = mu.T @ M @ mu                   # modeled ordered-pair mass    (K, K)
        E  = mu.T @ A @ mu                   # expected edge mass           (K, K)
        SW = mu.T @ W @ mu                   # expected weight mass         (K, K)
        E    = np.clip(E, 0.0, None)
        nonE = np.clip(P - E, 0.0, None)

        a_post = a0 + E
        b_post = b0 + nonE
        c_post = c0 + E
        d_post = d0 + SW
        g_post = gamma0 + n

        Elog_theta = digamma(a_post) - digamma(a_post + b_post)
        Elog_1mth  = digamma(b_post) - digamma(a_post + b_post)
        Elog_lam   = digamma(c_post) - np.log(d_post)
        Elam       = c_post / d_post
        Elog_pi    = digamma(g_post) - digamma(g_post.sum())

        # ---- ELBO at (mu, posteriors(mu)) -- non-decreasing across iterations ----
        data_ll = float(np.sum(
            we * (E * Elog_theta + nonE * Elog_1mth)
            + ww * (E * Elog_lam - SW * Elam)))
        mlogmu = np.zeros_like(mu)
        pos = mu > 0
        mlogmu[pos] = np.log(mu[pos])
        ent = -float(np.sum(mu * mlogmu))
        z_term = float(n @ Elog_pi) + ent
        kl = (float(np.sum(_beta_kl(a_post, b_post, a0, b0)))
              + float(np.sum(_gamma_kl(c_post, d_post, c0, d0)))
              + float(_dirichlet_kl(g_post, gamma0)))
        elbo = data_ll + z_term - kl
        elbos.append(elbo)
        if elbo - prev < tol * max(1.0, abs(prev)):
            break
        prev = elbo

        # ---- E-step: update responsibilities given the posteriors ----
        Mm_out = M @ mu;   Am_out = A @ mu;   Wm_out = W @ mu     # i as source
        Mm_in  = M.T @ mu; Am_in  = A.T @ mu; Wm_in  = W.T @ mu   # i as target

        out = (we * (Mm_out @ Elog_1mth.T)
               + we * (Am_out @ (Elog_theta - Elog_1mth).T)
               + ww * (Am_out @ Elog_lam.T)
               - ww * (Wm_out @ Elam.T))
        inn = (we * (Mm_in @ Elog_1mth)
               + we * (Am_in @ (Elog_theta - Elog_1mth))
               + ww * (Am_in @ Elog_lam)
               - ww * (Wm_in @ Elam))

        log_mu = Elog_pi[None, :] + out + inn + logmask
        log_mu -= log_mu.max(1, keepdims=True)
        mu = np.exp(log_mu)
        mu /= mu.sum(1, keepdims=True)

    return {
        'mu': mu, 'z': mu.argmax(1), 'elbo': elbos[-1], 'elbos': elbos,
        'a_post': a_post, 'b_post': b_post, 'c_post': c_post, 'd_post': d_post,
        'g_post': g_post, 'K': K,
    }


def fit_vb_wsbm(W, K, M=None, allowed=None, alpha=None,
                a0=1.0, b0=1.0, c0=1.0, d0=1.0, gamma0=1.0,
                n_init=10, max_iter=300, tol=1e-7, seed=0):
    """Fit the VB-WSBM at a fixed ``K`` with ``n_init`` random restarts.

    Returns the restart with the highest ELBO (VB is non-convex, so restarts
    matter). ``W`` is an ``(N, N)`` weight matrix (0 = non-edge; directed OK).
    ``M`` is the modeled-pairs mask (default: all off-diagonal pairs). ``allowed``
    is the per-node block mask (default: every node may take any block).

    ``alpha`` sets the existence/weight balance of Aicher et al. If ``None``
    (default), the two log-likelihood components are combined at *full* strength
    (the proper generative joint), which keeps the ELBO calibrated for choosing
    ``K``. If a float in ``[0, 1]`` is given, the Aicher convex combination is
    used instead (existence weight ``alpha``, weight-term weight ``1 - alpha``);
    note this tempers the data evidence and biases model selection toward fewer
    blocks, so it is best reserved for *fixed*-K topology-vs-weight exploration.
    """
    if alpha is None:
        we, ww = 1.0, 1.0
    else:
        we, ww = float(alpha), 1.0 - float(alpha)
    W = np.array(W, dtype=float)
    np.fill_diagonal(W, 0.0)
    A = (W > 0).astype(float)
    N = W.shape[0]

    if M is None:
        M = 1.0 - np.eye(N)
    else:
        M = np.array(M, dtype=float)
        np.fill_diagonal(M, 0.0)
    if allowed is None:
        allowed = np.ones((N, K), bool)

    priors = (a0, b0, c0, d0, gamma0)
    base = np.random.default_rng(seed)
    features = np.hstack([A + A.T, W + W.T])   # symmetrized connectivity, for the warm start

    best = None
    for r in range(n_init):
        if r == 0:
            labels = _kmeans_labels(features, allowed, seed)    # informed warm start
        else:
            labels = _random_labels(allowed, np.random.default_rng(base.integers(0, 2**32 - 1)))
        res = _fit_once(W, A, M, K, allowed, we, ww, priors, max_iter, tol,
                        _peaky(labels, allowed))
        if best is None or res['elbo'] > best['elbo']:
            best = res
    best['n_blocks_used'] = int(len(np.unique(best['z'])))
    return best


def select_K(W, K_values, **kwargs):
    """Scan ``K`` and pick the value with the best ELBO (variational log-evidence).

    Returns a dict with the per-K ELBOs, the best K, and its full fit result.
    Extra keyword args are forwarded to :func:`fit_vb_wsbm`.
    """
    results = {}
    elbos = {}
    for K in K_values:
        res = fit_vb_wsbm(W, K, **kwargs)
        results[K] = res
        elbos[K] = res['elbo']
    best_K = max(elbos, key=elbos.get)
    return {'elbos': elbos, 'results': results,
            'best_K': best_K, 'best': results[best_K]}


def block_affinity(W, z, agg='mean'):
    """All-cells block-affinity matrix ``omega[r, s]`` (same convention as the
    graph-tool notebook): mean weight over ordered pairs from block r to block s.
    Returns ``(omega, block_ids)``."""
    W = np.asarray(W, float)
    block_ids = sorted(np.unique(z).tolist())
    B = len(block_ids)
    idx = {b: k for k, b in enumerate(block_ids)}
    binv = np.array([idx[b] for b in z])
    om = np.zeros((B, B))
    cn = np.zeros((B, B))
    for r in range(B):
        rmask = binv == r
        for s in range(B):
            cell = W[np.ix_(rmask, binv == s)]
            om[r, s] = cell.sum()
            cn[r, s] = cell.size
    om = np.divide(om, cn, out=np.zeros_like(om), where=cn > 0)
    return om, block_ids

"""
sbm_pca_comparison.py
=====================

Companion to hippocampal_sbm.ipynb. Runs PCA and SBM on six adjacency-matrix
variants (3 direction-types x 2 region-sets) and produces 12 plots so the
number of meaningful PCs can be compared to the SBM-inferred block count B.

Direction-types
---------------
afferent  : only partner -> HPC edges retained.
efferent  : only HPC -> partner edges retained.
both      : bidirectional (HPC <-> partner) edges retained.

Region-sets ("individual" is variant-specific per user spec)
-----------
shared      : the 72 partners with BOTH afferent and efferent HPC connections.
individual  : afferent  -> all ~96 partners that project to HPC.
              efferent  -> all ~165 partners HPC projects to.
              both      -> the union (~189), every region with ANY HPC connection.

For all variants we zero-out:
  - non-HPC <-> non-HPC edges  (partner <-> partner block)
  - HPC <-> HPC edges          (per the new spec; differs from hippocampal_sbm.ipynb)
plus the direction-specific zeroing above.

After the main 12-plot analysis, an **ADDENDUM** at the bottom of the file reruns
the three direction-types with a single shared "union ~189" node set, so the
three scenarios use exactly the same nodes for cross-comparison.

How to run
----------
$ conda activate br2f
$ cd scripts
$ python sbm_pca_comparison.py

Plots are saved to ../output/sbm_pca_comparison/ as PNG files.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graph_tool.all as gt
from sklearn.decomposition import PCA

from helpers import log_weight_transform


# =============================================================================
# Constants
# =============================================================================

HIPPOCAMPAL_REGIONS: list[str] = ['DG', 'CA3', 'CA2', 'CA1v', 'CA1d', 'SUBv', 'SUBd']

# Threshold (cumulative explained-variance fraction) used to define the
# "meaningful number of PCs" we compare against the SBM block count B.
PCA_THRESHOLD = 0.90

# MCMC settings for the SBM consensus. Mirror hippocampal_sbm.ipynb.
N_COLLECT = 1500     # number of MCMC samples after burn-in
BURN_WAIT = 1000     # graph-tool's `wait` for equilibration
SEED      = 42

# Output directory for plots (relative to this script's location).
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH  = SCRIPT_DIR.parent / 'data' / 'average_connectome_data.csv'
OUT_DIR    = SCRIPT_DIR.parent / 'output' / 'sbm_pca_comparison'


# =============================================================================
# 1. Building the adjacency matrices
# =============================================================================

def load_connectome() -> pd.DataFrame:
    """Load the ordinal 0-7 connectome. Index = source region, columns = target region."""
    df = pd.read_csv(DATA_PATH, header=0, index_col=0)
    return df.fillna(0.0)


def get_partner_sets(df_connectome: pd.DataFrame,
                     hpc_regions: list[str]) -> dict[str, list[str]]:
    """
    Return the four partner sets used in the analysis.

    Returns
    -------
    dict with keys:
        'efferent_targets' : non-HPC regions HPC projects to (rows HPC, cols nonzero).
        'afferent_sources' : non-HPC regions that project to HPC.
        'shared'           : intersection (bidirectional). Size = 72 in our data.
        'union'            : 'efferent_targets' OR 'afferent_sources' (any HPC link).
    """
    # Non-HPC indices we'll inspect along the way.
    non_hpc = [r for r in df_connectome.index if r not in hpc_regions]

    # FROM HPC: rows = HPC subregions, columns = non-HPC regions HPC projects to.
    hpc_rows = df_connectome.loc[hpc_regions, non_hpc]
    efferent_targets = hpc_rows.columns[(hpc_rows.values > 0).any(axis=0)].tolist()

    # TO HPC: same idea on the transpose -> non-HPC regions that send to HPC.
    hpc_cols = df_connectome.loc[non_hpc, hpc_regions]
    afferent_sources = hpc_cols.index[(hpc_cols.values > 0).any(axis=1)].tolist()

    shared = sorted(set(afferent_sources) & set(efferent_targets),
                    key=efferent_targets.index)         # preserve a stable order
    union  = sorted(set(afferent_sources) | set(efferent_targets),
                    key=lambda r: (r not in efferent_targets, r))

    return {
        'efferent_targets': efferent_targets,
        'afferent_sources': afferent_sources,
        'shared':           shared,
        'union':            union,
    }


def build_adjacency(df_connectome: pd.DataFrame,
                    hpc_regions: list[str],
                    partner_regions: list[str],
                    direction: str) -> pd.DataFrame:
    """
    Construct a square directed adjacency matrix for one (direction, region-set)
    scenario, applying all required zeroings.

    Parameters
    ----------
    df_connectome
        Full connectome matrix. Entry [i, j] = ordinal weight from i to j.
    hpc_regions
        The 7 hippocampal subregion names.
    partner_regions
        The non-HPC regions to include for this scenario.
    direction
        One of 'afferent', 'efferent', 'both'.

    Returns
    -------
    pd.DataFrame
        Square (N x N) matrix where N = len(hpc_regions) + len(partner_regions).
        Partner<->partner is zeroed always.
        HPC<->HPC is zeroed always (per the new spec).
        Direction-specific zeroing:
            'afferent'  : only partner -> HPC entries remain non-zero.
            'efferent'  : only HPC -> partner entries remain non-zero.
            'both'      : both directions kept.
    """
    region_order = list(partner_regions) + list(hpc_regions)
    A = df_connectome.loc[region_order, region_order].copy()

    # Always: zero non-HPC <-> non-HPC and HPC <-> HPC.
    A.loc[partner_regions, partner_regions] = 0.0
    A.loc[hpc_regions,     hpc_regions]     = 0.0

    if direction == 'afferent':
        # Keep partner -> HPC only; zero HPC -> partner.
        A.loc[hpc_regions, partner_regions] = 0.0
    elif direction == 'efferent':
        # Keep HPC -> partner only; zero partner -> HPC.
        A.loc[partner_regions, hpc_regions] = 0.0
    elif direction == 'both':
        pass     # nothing further to zero
    else:
        raise ValueError(f"direction must be 'afferent', 'efferent', or 'both'; got {direction!r}")

    return A


def feature_vectors_for_pca(df_connectome: pd.DataFrame,
                            hpc_regions: list[str],
                            partner_regions: list[str],
                            direction: str) -> pd.DataFrame:
    """
    Build the feature matrix for PCA so that partner regions are the SAMPLES
    and HPC subregions are the FEATURES (sklearn convention: rows=samples, cols=features).

    Shapes:
        afferent / efferent  ->  (M_partners, 7)
        both                 ->  (M_partners, 14)    [in_<HPC> | out_<HPC>]

    PCA on this matrix asks "how many HPC-profile dimensions do the M partners require?"
    - this is the partner-side richness measure that maps to SBM B / B_eff.
    """
    # df_aff: partner -> HPC weights;  rows=partner, cols=HPC.
    # df_eff: HPC -> partner weights, transposed so rows=partner, cols=HPC.
    df_aff = df_connectome.loc[partner_regions, hpc_regions]
    df_eff = df_connectome.loc[hpc_regions, partner_regions].T

    df_aff.index.name = 'partner'; df_eff.index.name = 'partner'

    if direction == 'afferent':
        return df_aff
    elif direction == 'efferent':
        return df_eff
    elif direction == 'both':
        # Disambiguate column names so the concatenation is unique.
        df_aff = df_aff.add_prefix('in_')
        df_eff = df_eff.add_prefix('out_')
        return pd.concat([df_aff, df_eff], axis=1)
    else:
        raise ValueError(f"direction must be 'afferent', 'efferent', or 'both'; got {direction!r}")


# =============================================================================
# 2. SBM: fit + plot
# =============================================================================

def fit_sbm_directed(adjacency_log: pd.DataFrame,
                     n_collect: int = N_COLLECT,
                     burn_wait: int = BURN_WAIT,
                     seed: int = SEED) -> dict:
    """
    Fit a directed, weighted, degree-corrected SBM (graph-tool defaults) with
    MDL + MCMC consensus, on a log-weighted adjacency matrix.

    `adjacency_log` must already have been passed through `log_weight_transform`
    (positive continuous entries in [0, 1]). Zero entries are treated as non-edges.

    Returns
    -------
    dict with:
        'B'          : int, # blocks in consensus partition.
        'blocks'     : np.ndarray (N,), consensus block label per node.
        'node_names' : list[str], region names matching the row order.
        'omega'      : np.ndarray (B, B), mean log-weight from block r -> block s
                       (all-cells average, including non-edges).
        'graph'      : the gt.Graph used (for downstream inspection).
    """
    gt.seed_rng(seed)
    np.random.seed(seed)

    region_names = list(adjacency_log.index)
    W = adjacency_log.values
    n = len(region_names)

    # -- Build directed graph; only non-zero entries become edges.
    g = gt.Graph(directed=True)
    name_vp   = g.new_vp('string')
    weight_ep = g.new_ep('double')
    vs = [g.add_vertex() for _ in range(n)]
    for i, v in enumerate(vs):
        name_vp[v] = region_names[i]
    for i in range(n):
        for j in range(n):
            w = W[i, j]
            if w > 0:
                e = g.add_edge(vs[i], vs[j])
                weight_ep[e] = w
    g.vp['name']   = name_vp
    g.ep['weight'] = weight_ep

    # -- MDL fit (degree-corrected by default), real-exponential edge weights.
    state = gt.minimize_blockmodel_dl(
        g, state_args=dict(recs=[weight_ep], rec_types=['real-exponential']))

    # -- Burn in, then collect MCMC samples for a consensus partition.
    gt.mcmc_equilibrate(state, wait=burn_wait, mcmc_args=dict(niter=10))
    partitions: list[np.ndarray] = []
    gt.mcmc_equilibrate(
        state, force_niter=n_collect, mcmc_args=dict(niter=10),
        callback=lambda s: partitions.append(s.b.a.copy()),
    )
    pmode = gt.PartitionModeState(partitions, converge=True)
    b_consensus = pmode.get_max(g)

    blocks = np.array([int(b_consensus[v]) for v in g.vertices()])
    block_ids = sorted(set(blocks))
    B = len(block_ids)

    # -- Block-pair affinity (all-cells mean: rho * mu_hat) for visualization.
    binv = np.array([block_ids.index(b) for b in blocks])
    omega = np.zeros((B, B)); counts = np.zeros((B, B))
    for r in range(B):
        rmask = binv == r
        for s in range(B):
            cell = W[np.ix_(rmask, binv == s)]
            omega[r, s] = cell.sum(); counts[r, s] = cell.size
    omega = np.divide(omega, counts, out=np.zeros_like(omega), where=counts > 0)

    return {
        'B':          B,
        'blocks':     blocks,
        'block_ids':  block_ids,
        'binv':       binv,
        'node_names': region_names,
        'omega':      omega,
        'adj_log':    W,
        'graph':      g,
    }


def plot_sbm(result: dict, scenario_label: str, save_path: Path | None) -> None:
    """
    Two-panel SBM figure:
      left  - the log-weighted adjacency with rows/cols reordered by consensus
              block; red lines mark block boundaries.
      right - the B x B block affinity heatmap (mean log-weight per block-pair).
    """
    W         = result['adj_log']
    binv      = result['binv']
    omega     = result['omega']
    B         = result['B']
    block_ids = result['block_ids']

    order = np.argsort(binv, kind='stable')
    W_ord = W[np.ix_(order, order)]
    binv_ord = binv[order]
    boundaries = np.where(np.diff(binv_ord) != 0)[0] + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # -- Left panel: adjacency sorted by block.
    vmax = float(W.max()) if W.max() > 0 else 1.0
    axes[0].imshow(W_ord, cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
    axes[0].set_title('Adjacency (log-weighted, sorted by block)')
    axes[0].set_xlabel('TO node (sorted by block)')
    axes[0].set_ylabel('FROM node (sorted by block)')
    for b in boundaries:
        axes[0].axhline(b - 0.5, color='red', lw=0.5)
        axes[0].axvline(b - 0.5, color='red', lw=0.5)

    # -- Right panel: omega heatmap with annotations.
    sns.heatmap(
        omega, annot=True, fmt='.3f', cmap='viridis', ax=axes[1],
        xticklabels=[f'b{b}' for b in block_ids],
        yticklabels=[f'b{b}' for b in block_ids],
        cbar_kws={'fraction': 0.04, 'pad': 0.02},
    )
    axes[1].set_title(rf'$\omega[r \to s]$  (B = {B} blocks)')
    axes[1].set_xlabel('TO block'); axes[1].set_ylabel('FROM block')

    fig.suptitle(f'SBM: {scenario_label}', fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 3. PCA: fit + plot
# =============================================================================

def l2_normalize_rows(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Divide each sample (row) by its L2 norm so every sample vector has unit length.

    This makes PCA key on the *shape* of each partner's HPC connectivity profile
    rather than its overall connection magnitude. It replaces the earlier
    per-column standardization step (no column z-scoring is applied here).

    Rows that are entirely zero (a partner with no connections in this feature
    block) are left as zeros to avoid divide-by-zero.

    Returns
    -------
    pd.DataFrame
        Row-normalized matrix with the original row index and columns.
    """
    X = feat.values.astype(float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pd.DataFrame(X / norms, index=feat.index, columns=feat.columns)


def fit_pca(feature_matrix: pd.DataFrame, threshold: float = PCA_THRESHOLD) -> dict:
    """
    Fit PCA on a (n_obs x n_features) matrix.

    Returns
    -------
    dict with:
        'explained_variance_ratio'  : per-component variance (sums to ~1).
        'cumulative'                : cumulative variance curve.
        'n_pcs_at_threshold'        : smallest k s.t. cumulative[k-1] >= threshold.
        'pca'                       : the fitted sklearn PCA object.
    """
    X = np.asarray(feature_matrix.values, dtype=float)
    max_pcs = min(X.shape)

    pca = PCA(n_components=max_pcs, svd_solver='full')
    pca.fit(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    # +1 because we count components 1-indexed.
    n_pcs_at_threshold = int(np.searchsorted(cumulative, threshold) + 1)

    return {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative':               cumulative,
        'n_pcs_at_threshold':       n_pcs_at_threshold,
        'pca':                      pca,
    }


def plot_pca(result: dict, scenario_label: str,
             save_path: Path | None, threshold: float = PCA_THRESHOLD) -> None:
    """Bar chart of per-component variance, with the cumulative line + threshold."""
    var = result['explained_variance_ratio'] * 100
    cum = result['cumulative'] * 100
    n   = result['n_pcs_at_threshold']

    components = np.arange(1, len(var) + 1)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(components, var, color='royalblue', edgecolor='k', alpha=0.75,
           label='per-component variance')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Per-component variance (%)', color='royalblue')
    ax.tick_params(axis='y', labelcolor='royalblue')
    ax.set_xticks(components)

    ax2 = ax.twinx()
    ax2.plot(components, cum, 'ro-', label='cumulative variance')
    ax2.axhline(threshold * 100, color='gray', linestyle='--',
                label=f'{int(threshold*100)}% threshold')
    ax2.axvline(n, color='red', linestyle=':',
                label=f'{n} PCs reach threshold')
    ax2.set_ylabel('Cumulative variance (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 105)

    # Single combined legend.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='center right', framealpha=0.9)

    ax.set_title(f'PCA: {scenario_label}')
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 4. Running one scenario end-to-end
# =============================================================================

def run_scenario(df_connectome: pd.DataFrame,
                 partner_regions: list[str],
                 direction: str,
                 region_set_label: str,
                 out_dir: Path) -> dict:
    """
    Run one (direction, region-set) scenario: build adjacency, log-transform,
    fit SBM, fit PCA, save the two plots, and return summary numbers.

    Returns
    -------
    dict with 'direction', 'region_set', 'B' (SBM blocks), 'n_pcs' (PCA threshold count),
    'n_nodes', 'n_partners', and the file stems of the saved plots.
    """
    label = f'{direction} / {region_set_label}'
    print(f'\n--- Scenario: {label}  ({len(partner_regions)} partners) ---')

    # 1. Square directed adjacency with all required zeroings, then log-transform.
    A_raw = build_adjacency(df_connectome, HIPPOCAMPAL_REGIONS, partner_regions, direction)
    A_log = pd.DataFrame(
        log_weight_transform(A_raw.values),
        index=A_raw.index, columns=A_raw.columns,
    )

    # 2. SBM.
    sbm_res = fit_sbm_directed(A_log)
    print(f'    SBM inferred B = {sbm_res["B"]} blocks')

    sbm_path = out_dir / f'sbm__{direction}__{region_set_label}.png'
    plot_sbm(sbm_res, label, sbm_path)

    # 3. PCA on the matching feature vectors (7 HPC rows x M partner cols, log-weighted).
    #    L2-normalize each sample (row) before PCA so PCA keys on each partner's
    #    HPC-profile *shape*, not its overall connection magnitude. No column z-scoring.
    feat = feature_vectors_for_pca(df_connectome, HIPPOCAMPAL_REGIONS, partner_regions, direction)
    feat_log = pd.DataFrame(
        log_weight_transform(feat.values),
        index=feat.index, columns=feat.columns,
    )
    feat_norm = l2_normalize_rows(feat_log)
    pca_res = fit_pca(feat_norm)
    print(f'    PCA: {pca_res["n_pcs_at_threshold"]} PCs reach '
          f'{int(PCA_THRESHOLD*100)}% cumulative variance '
          f'(max possible: {min(feat_norm.shape)})')

    pca_path = out_dir / f'pca__{direction}__{region_set_label}.png'
    plot_pca(pca_res, label, pca_path)

    return {
        'direction':   direction,
        'region_set':  region_set_label,
        'n_partners':  len(partner_regions),
        'n_nodes':     A_raw.shape[0],
        'B':           sbm_res['B'],
        'n_pcs':       pca_res['n_pcs_at_threshold'],
        'sbm_plot':    sbm_path.name,
        'pca_plot':    pca_path.name,
    }


# =============================================================================
# 5. MAIN: variant-specific scenarios (the 6 the user requested -> 12 plots)
# =============================================================================

def run_main_analysis(df_connectome: pd.DataFrame, partner_sets: dict) -> pd.DataFrame:
    """Run all 6 variant-specific scenarios and return a summary table."""
    print('\n' + '=' * 72)
    print('MAIN ANALYSIS  (variant-specific "individual" sets)')
    print('=' * 72)

    # For each direction, "individual" picks the partner set relevant to that direction.
    # "both" uses the union: every region with any HPC connection.
    individual_by_direction = {
        'afferent': partner_sets['afferent_sources'],
        'efferent': partner_sets['efferent_targets'],
        'both':     partner_sets['union'],
    }

    rows = []
    for direction in ['afferent', 'efferent', 'both']:
        # shared (72) version
        rows.append(run_scenario(
            df_connectome, partner_sets['shared'], direction, 'shared', OUT_DIR,
        ))
        # individual (variant-specific) version
        rows.append(run_scenario(
            df_connectome, individual_by_direction[direction], direction, 'individual', OUT_DIR,
        ))

    summary = pd.DataFrame(rows)
    return summary


# =============================================================================
# 6. ADDENDUM: union region set across ALL three direction-types
# =============================================================================
#
#    >>>> ADDENDUM SECTION BELOW <<<<
#
#    The user explicitly requested that this addendum be included at the bottom
#    of the file AND be visually distinct from the main analysis. The addendum
#    reruns the three direction-types (afferent / efferent / both) using ONE
#    consistent node set: the union ~189 of every region with any HPC connection.
#    This makes the three resulting SBMs node-comparable.

ADDENDUM_BANNER = (
    '\n' + '#' * 72 +
    '\n#  ADDENDUM: same direction-types, but a SINGLE union (~189) node set'
    '\n#            across all three. This is NOT part of the main 12-plot'
    '\n#            analysis above; it lets you compare partitions across'
    '\n#            directions on the same nodes.'
    '\n' + '#' * 72 + '\n'
)


def run_addendum(df_connectome: pd.DataFrame, partner_sets: dict) -> pd.DataFrame:
    """Rerun afferent / efferent / both with the union node set throughout."""
    print(ADDENDUM_BANNER)
    union_partners = partner_sets['union']

    rows = []
    for direction in ['afferent', 'efferent', 'both']:
        rows.append(run_scenario(
            df_connectome,
            union_partners,
            direction,
            region_set_label='union_addendum',
            out_dir=OUT_DIR,
        ))
    return pd.DataFrame(rows)


# =============================================================================
# 7. Driver
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Saving plots to: {OUT_DIR}')

    df_connectome = load_connectome()
    print(f'Loaded connectome: {df_connectome.shape}')

    partner_sets = get_partner_sets(df_connectome, HIPPOCAMPAL_REGIONS)
    print('Partner-set sizes:')
    for k, v in partner_sets.items():
        print(f'    {k:>18s} : {len(v):>4d}')

    # --- Main 12-plot analysis -----------------------------------------------
    main_summary = run_main_analysis(df_connectome, partner_sets)
    print('\nMain analysis summary:')
    print(main_summary[['direction', 'region_set', 'n_partners', 'n_nodes',
                        'B', 'n_pcs']].to_string(index=False))
    main_summary.to_csv(OUT_DIR / 'summary_main.csv', index=False)

    # --- Addendum: union node set across all three directions ----------------
    addendum_summary = run_addendum(df_connectome, partner_sets)
    print('\nAddendum (union node set) summary:')
    print(addendum_summary[['direction', 'region_set', 'n_partners', 'n_nodes',
                            'B', 'n_pcs']].to_string(index=False))
    addendum_summary.to_csv(OUT_DIR / 'summary_addendum.csv', index=False)

    print(f'\nDone. {len(main_summary) * 2 + len(addendum_summary) * 2} plots in {OUT_DIR}')


if __name__ == '__main__':
    main()

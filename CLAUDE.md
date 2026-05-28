# Bilateral Rat Connectome Analysis (br2f)

Analysis of the bilateral rat structural connectome from Swanson et al. (2024), with a
focus on the **hippocampal formation** and **left/right hemisphere symmetry**.

> ⚠️ INFERRED — please correct: the precise research question/goal below is my best guess
> from reading the code. Edit this section so future sessions start with the real framing.
> Likely goal: characterise hippocampal connectivity profiles (afferent vs efferent) via
> dimensionality reduction, similarity/clustering, and bilateral symmetry comparison.

## Domain context for Claude
- I (Chris) bring the **neuroscience** knowledge. Focus your explanations on the **code,
  statistics, and ML methods** — assume I understand the anatomy/biology unless I ask.

## The data (`data/`)
- Connectome matrices are **region × region**, directed: entry `[i, j]` = connection
  strength **FROM region i → TO region j** (rows = source, columns = target).
- Weights are **ordinal 0–7**. `helpers.log_weight_transform()` maps them to a
  log-weighted scale (Faskowitz & Sporns 2020).
- ~391 regions. `label_major_division_mapping.csv` maps each region to its major brain
  division (e.g. CNU).
- Hippocampal regions of interest: `DG, CA3, CA2, CA1v, CA1d, SUBv, SUBd` (7 regions).
- `average_connectome_data.csv` is the averaged matrix most analyses load.

## Conventions
- **Afferent = TO** the seed regions (incoming); **efferent = FROM** (outgoing). The
  `get_feature_vectors_*` functions in `helpers.py` build these feature vectors.
- Notebooks run from inside `scripts/`, so paths are relative: `../data/...`, `../output/...`.
- Shared code lives in `scripts/helpers.py` and `scripts/analysis.py` — prefer extending
  these over copy-pasting logic into notebooks.
- Derived data and figures are written to `output/`.

## Code map (`scripts/`)
- Dimensionality reduction: `PCA_hippocampal.ipynb`, `UMAP_analysis.ipynb`
- Similarity / RSA: `Hippocampus_rsa_analysis.ipynb`
- Clustering: `*_community_clustering.ipynb`, `consensus_clustering.ipynb`,
  `spectral_clustering.ipynb`, `stochastic_block_modeling.ipynb`,
  `allegiance_consensus_clustering.ipynb`
- Symmetry: `symmetry_analysis.ipynb`, `frobenius_norm_stats.ipynb`,
  `bootstrapping_connectivity.ipynb`
- **Scratch (not canonical):** `scratch.ipynb`, `rsa_analysis_scratchpad.ipynb`

## Gotchas
- Only **7 hippocampal samples**, so PCA has at most 7 components — watch for that ceiling.
- `.gitignore` currently contains an **unresolved git merge conflict** (`<<<<<<<` markers).
  Worth resolving.

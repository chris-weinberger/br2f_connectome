import bct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def get_feature_vectors_unique():
    df_average = pd.read_csv('../data/average_connectome_data.csv', header=0, index_col=0)
    hippocampal_regions = np.array(['DG','CA3','CA2','CA1v','CA1d','SUBv','SUBd'])
    # FROM hippocampus (efferent)
    df_avg_from = df_average[df_average.index.isin(hippocampal_regions)]
    
    # TO hippocampus (afferent)
    df_average_t = df_average.T
    df_avg_to = df_average_t[df_average_t.index.isin(hippocampal_regions)]
    
    # drop HPC columns
    df_avg_from = df_avg_from.drop(hippocampal_regions, axis=1)
    df_avg_to = df_avg_to.drop(hippocampal_regions, axis=1)
    
    # filter to only include columns and rows with at least one connection
    df_avg_from = df_avg_from.loc[:,df_avg_from.apply(np.count_nonzero, axis=0) >= 1]
    df_avg_to = df_avg_to.loc[:,df_avg_to.apply(np.count_nonzero, axis=0) >= 1]

    return (df_avg_from, df_avg_to)

def get_feature_vectors_shared():
    df_average = pd.read_csv('../data/average_connectome_data.csv', header=0, index_col=0)
    hippocampal_regions = np.array(['DG','CA3','CA2','CA1v','CA1d','SUBv','SUBd'])
    # FROM hippocampus (efferent)
    df_avg_from = df_average[df_average.index.isin(hippocampal_regions)]
    
    # TO hippocampus (afferent)
    df_average_t = df_average.T
    df_avg_to = df_average_t[df_average_t.index.isin(hippocampal_regions)]
    
    # drop HPC columns
    df_avg_from = df_avg_from.drop(hippocampal_regions, axis=1)
    df_avg_to = df_avg_to.drop(hippocampal_regions, axis=1)
    
    # filter to only include columns and rows with at least one connection
    df_avg_from = df_avg_from.loc[:,df_avg_from.apply(np.count_nonzero, axis=0) >= 1]
    df_avg_to = df_avg_to.loc[:,df_avg_to.apply(np.count_nonzero, axis=0) >= 1]

    # find the shared regions
    common_cols = df_avg_to.columns.intersection(df_avg_from.columns)
    df_avg_to_shared = df_avg_to[common_cols]
    df_avg_from_shared = df_avg_from[common_cols]

    return (df_avg_to_shared, df_avg_from_shared)

def get_correlation_matrix(feature_vectors, distance_metric='cosine'):
    '''
    INPUT:
        feature_vectors: k x N dataframe, where k are number of features, N are number of instances
        distance_metric: string indicating distance metric type

    OUTPUT:
        NxN correlation matrix
    '''
    
    if distance_metric=='cosine':
        cosine_values = cosine_similarity(feature_vectors.T)
        cosine_labels = feature_vectors.columns
        
        df_cosine_similarity = pd.DataFrame(cosine_values, 
                             index=cosine_labels, 
                             columns=cosine_labels)
        return df_cosine_similarity
    elif distance_metric=='spearman':
        spearman_df = (feature_vectors.corr(method='spearman').dropna(axis=0, how='all')).dropna(axis=1, how='all')
        return spearman_df
    else:
        print('unrecognized distance metric')

def plot_incoming_outgoing_PCA(incoming_distance, outgoing_distance, n_components=7, scale=False):
    # --- 1. Standardize the Data ---
    # It's a best practice to scale data before running PCA.
    afferent_data = None
    efferent_data = None
    if scale:
        scaler = StandardScaler()
        afferent_data = scaler.fit_transform(incoming_distance)
        efferent_data = scaler.fit_transform(outgoing_distance)
    else:
        afferent_data = incoming_distance
        efferent_data = outgoing_distance
    
    # --- 2. Perform PCA on Both Datasets ---
    # Since each dataset has 7 samples, there can be a maximum of 7 principal components.
    hpc_pca_afferent = PCA(n_components=n_components, svd_solver='full')
    hpc_pca_efferent = PCA(n_components=n_components, svd_solver='full')
    
    # Fit the PCA models to each dataset
    hpc_pca_afferent.fit(afferent_data)
    hpc_pca_efferent.fit(efferent_data)
    
    
    # --- 3. Extract Explained Variance ---
    # The 'explained_variance_ratio_' attribute gives the proportion of variance.
    # Multiply by 100 to convert it to a percentage.
    hpc_variance_explained_afferent = hpc_pca_afferent.explained_variance_ratio_ * 100
    hpc_variance_explained_efferent = hpc_pca_efferent.explained_variance_ratio_ * 100
    
    
    # --- 4. Plot the Results (Scree Plot) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define the x-axis positions for the bars
    component_numbers = np.arange(1, n_components + 1)
    bar_width = 0.35
    
    # Create the bar plot for afferent
    ax.bar(
        component_numbers - bar_width/2,
        hpc_variance_explained_afferent,
        width=bar_width,
        label='Afferent Connections',
        color='royalblue',
        edgecolor='k'
    )
    
    # Create the bar plot for efferent
    ax.bar(
        component_numbers + bar_width/2,
        hpc_variance_explained_efferent,
        width=bar_width,
        label='Efferent Connections',
        color='tomato',
        edgecolor='k'
    )

    max_y = max(np.max(hpc_variance_explained_afferent), 
                np.max(hpc_variance_explained_efferent))
    ax.set_ylim(0, max_y * 1.15)
    
    # --- 5. Final Chart Formatting ---
    ax.set_xlabel('Principal Component', fontsize=20)
    ax.set_ylabel('Percent Variance Explained (%)', fontsize=20)
    ax.set_title('Variance Explained by Principal Components', fontsize=24, weight='bold')
    ax.set_xticks(component_numbers) # Ensure integer ticks for each component
    ax.legend(fontsize=20)
    plt.tight_layout()
    # plt.savefig('../output/PCA_connectivity_profiles.png')
    plt.show()
    return fig

def plot_incoming_outgoing_PCA_cumulative(incoming_distance, outgoing_distance, n_components=7, scale=False):
    # --- 1. Standardize the Data ---
    # (No changes in this section)
    afferent_data = None
    efferent_data = None
    if scale:
        scaler = StandardScaler()
        afferent_data = scaler.fit_transform(incoming_distance)
        efferent_data = scaler.fit_transform(outgoing_distance)
    else:
        afferent_data = incoming_distance
        efferent_data = outgoing_distance
    
    # --- 2. Perform PCA on Both Datasets ---
    # (No changes in this section)
    hpc_pca_afferent = PCA(n_components=n_components, svd_solver='full')
    hpc_pca_efferent = PCA(n_components=n_components, svd_solver='full')
    
    # Fit the PCA models to each dataset
    hpc_pca_afferent.fit(afferent_data)
    hpc_pca_efferent.fit(efferent_data)
    
    
    # --- 3. Extract *Cumulative* Explained Variance ---  <--- MODIFICATION 1
    # Get the individual variance ratios
    hpc_variance_explained_afferent = hpc_pca_afferent.explained_variance_ratio_ * 100
    hpc_variance_explained_efferent = hpc_pca_efferent.explained_variance_ratio_ * 100
    
    # Calculate the cumulative sum
    cumulative_variance_afferent = np.cumsum(hpc_variance_explained_afferent)
    cumulative_variance_efferent = np.cumsum(hpc_variance_explained_efferent)
    
    
    # --- 4. Plot the Results (Cumulative Line Plot) --- <--- MODIFICATION 2
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define the x-axis positions
    component_numbers = np.arange(1, n_components + 1)
    
    # (Removed bar_width and ax.bar() calls)
    
    # Create the line plot for afferent
    ax.plot(
        component_numbers,
        cumulative_variance_afferent,
        label='Afferent Connections',
        color='royalblue',
        marker='o',  # Added marker
        linestyle='-'
    )
    
    # Create the line plot for efferent
    ax.plot(
        component_numbers,
        cumulative_variance_efferent,
        label='Efferent Connections',
        color='tomato',
        marker='s',  # Added different marker
        linestyle='-'
    )
    
    
    # --- 5. Final Chart Formatting --- <--- MODIFICATION 3
    ax.set_xlabel('Number of Principal Components', fontsize=20)
    ax.set_ylabel('Cumulative Percent Variance Explained (%)', fontsize=20) # Updated label
    ax.set_title(
        'Cumulative Variance Explained by Principal Components', 
        fontsize=24, 
        weight='bold'
    ) # Updated title
    ax.set_xticks(component_numbers) # Ensure integer ticks for each component
    
    # Optional: Add a line for a common threshold, like 90% or 95%
    ax.axhline(y=90, color='grey', linestyle=':', label='90% Variance Threshold')
    
    ax.legend(fontsize=24)
    plt.tight_layout()
    # plt.savefig('../output/PCA_cumulative_variance.png') # Updated save filename
    plt.show()
    return fig

def plot_variance_differences(real_differences, 
                              simulated_differences, 
                              bootstrap_ci_lower, 
                              bootstrap_ci_upper):
    """
    Plots the difference in cumulative variance explained between
    real data and a simulated (bootstrapped) average.
    
    Args:
        real_differences (array-like): 7-element vector of real differences.
        simulated_differences (array-like): 7-element vector of simulated differences.
        bootstrap_ci_lower (array-like): 7-element vector of the lower 95% CI bound.
        bootstrap_ci_upper (array-like): 7-element vector of the upper 95% CI bound.
    """
    
    # --- 1. Setup ---
    # Ensure inputs are numpy arrays for easier handling
    real_diff = np.array(real_differences)
    sim_diff = np.array(simulated_differences)
    ci_lower = np.array(bootstrap_ci_lower)
    ci_upper = np.array(bootstrap_ci_upper)
    
    # Get the number of components (should be 7 based on your description)
    n_components = len(real_diff)
    component_numbers = np.arange(1, n_components + 1)
    
    if (len(sim_diff) != n_components or 
        len(ci_lower) != n_components or 
        len(ci_upper) != n_components):
        print("Error: All input vectors must have the same length.")
        return None

    # --- 2. Create the Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the 95% confidence interval band for the simulated data
    ax.fill_between(
        component_numbers,
        ci_lower,
        ci_upper,
        color='gray',
        alpha=0.4,
        label='95% CI (Simulated)'
    )
    
    # Plot the simulated average difference line
    ax.plot(
        component_numbers,
        sim_diff,
        label='Simulated Difference (Avg)',
        color='black',
        marker='x',
        linestyle='--',
        linewidth=2
    )

    # Plot the real difference line
    ax.plot(
        component_numbers,
        real_diff,
        label='Real Difference',
        color='red',
        marker='o',
        linestyle='-',
        linewidth=2.5
    )

    # --- 3. Final Chart Formatting ---
    
    # Add a horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1, label='No Difference')

    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Difference in Cumulative Variance Explained (%)', fontsize=12)
    ax.set_title(
        'Real vs. Simulated Difference in Cumulative Variance', 
        fontsize=18, 
        weight='bold'
    )
    ax.set_xticks(component_numbers)
    
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    # plt.savefig('../output/PCA_difference_plot.png')
    plt.show()
    
    return fig

def plot_rsa(matrix, labels=None, title=None, output_fname=''):
    # ----- 3. Plot the RSA matrix sorted -----
    plt.figure(figsize=(18, 15))
    im = plt.imshow(matrix, cmap='viridis', interpolation='none', aspect='auto')

    tick_frequency = 1
    if labels is not None:
        plt.xticks(np.arange(0, len(labels), tick_frequency),
                   labels=labels[::tick_frequency], rotation=90)
        plt.yticks(np.arange(0, len(labels), tick_frequency),
               labels=labels[::tick_frequency])
    else:
        plt.axis('off')

    if title is not None:
        plt.title(title, fontsize=36)
    # else:
    #     plt.title('Reordered Connections Similarity Matrix', fontsize=36)
    
    # Add a colorbar to show the mapping of colors to similarity values.
    cbar = plt.colorbar(im, pad=0.04, location='right')
    cbar.ax.tick_params(labelsize=20)
    # cbar.set_label('Cosine Similarity')
    
    plt.tight_layout()
    if (len(output_fname) > 0):
        plt.savefig(output_fname)

def louvain_rsa(similarity_matrix_values, labels, gamma=1):
    # ----- 1. Run Louvain on similarity matrix -----
    gamma_resolution = gamma
    community_assignments, modularity_score = bct.community_louvain(similarity_matrix_values, 
                                                                    gamma=gamma_resolution, 
                                                                    seed=42)

    # ----- 2. Sort the RSA matrix according to assignments -----
    # Get the indices that would sort the brain regions by their community assignment.
    sort_indices = np.argsort(community_assignments)
    
    # Reorder the similarity matrix using these sorted indices.
    sorted_matrix = similarity_matrix_values[sort_indices, :][:, sort_indices]
    
    # Reorder the brain labels list to match the sorted matrix.
    sorted_labels = [labels[i] for i in sort_indices]

    # ----- 3. Create the bar chart of community sizes -----
    unique_communities, counts = np.unique(community_assignments, return_counts=True)
    community_names = [f"Community {c}" for c in unique_communities]
    
    plt.figure(figsize=(8, 5))
    plt.bar(community_names, counts, color='skyblue')
    plt.title('Number of Brain Regions per Community', fontsize=16)
    plt.xlabel('Community', fontsize=12)
    plt.ylabel('Number of Brain Regions (Size)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    return (sorted_matrix, sorted_labels)

def get_pca_variances(matrix_afferent, matrix_efferent, n_components, scale=False):
    """
    Runs PCA on two (samples, samples) matrices and returns the
    element-wise their explained variance ratios.
    """
    # CHANGE: We must transpose the matrices to (n_samples, n_features)
    # before fitting them with sklearn's PCA.
    # CHANGE: Sstandardize the data
    scaler = StandardScaler()
    afferent_scaled = scaler.fit_transform(matrix_afferent)
    efferent_scaled = scaler.fit_transform(matrix_efferent)
    
    pca1 = PCA(n_components=n_components, svd_solver='full')
    pca1.fit(afferent_scaled)
    vars_a = pca1.explained_variance_ratio_
    
    pca2 = PCA(n_components=n_components, svd_solver='full')
    pca2.fit(efferent_scaled) # Fit on (N_ROIS, N_FEATURES)
    vars_e = pca2.explained_variance_ratio_
    
    # Return the difference for each component (e.g., [PC1_diff, PC2_diff, ...])
    return (vars_a, vars_e)
    
    

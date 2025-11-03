import bct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def get_feature_vectors_unique():
    df_average = pd.read_csv('../data/average_connectome_data.csv', header=0, index_col=0)
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

def plot_incoming_outgoing_PCA(incoming_distance, outgoing_distance, n_components=10):
    # --- 1. Standardize the Data ---
    # It's a best practice to scale data before running PCA.
    scaler = StandardScaler()
    afferent_scaled = scaler.fit_transform(incoming_distance)
    efferent_scaled = scaler.fit_transform(outgoing_distance)
    
    
    # --- 2. Perform PCA on Both Datasets ---
    # Since each dataset has 7 samples, there can be a maximum of 7 principal components.
    hpc_pca_afferent = PCA(n_components=n_components)
    hpc_pca_efferent = PCA(n_components=n_components)
    
    # Fit the PCA models to each dataset
    hpc_pca_afferent.fit(afferent_scaled)
    hpc_pca_efferent.fit(efferent_scaled)
    
    
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
        label='Incoming Connections',
        color='royalblue',
        edgecolor='k'
    )
    
    # Create the bar plot for efferent
    ax.bar(
        component_numbers + bar_width/2,
        hpc_variance_explained_efferent,
        width=bar_width,
        label='Outgoing Connections',
        color='tomato',
        edgecolor='k'
    )
    
    
    # --- 5. Final Chart Formatting ---
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Percent Variance Explained (%)', fontsize=12)
    ax.set_title('Variance Explained by Principal Components in Incoming vs Outgoing Connections With Hippocampus', fontsize=18, weight='bold')
    ax.set_xticks(component_numbers) # Ensure integer ticks for each component
    ax.legend(fontsize=20)
    plt.tight_layout()
    # plt.savefig('../output/PCA_connectivity_profiles.png')
    plt.show()
    return plt

def plot_rsa(matrix, labels=None, title=None, output_fname=''):
    # ----- 3. Plot the RSA matrix sorted -----
    plt.figure(figsize=(15, 15))
    im = plt.imshow(matrix, cmap='viridis', interpolation='none', aspect='auto')

    tick_frequency = 1
    if labels is not None:
        plt.xticks(np.arange(0, len(labels), tick_frequency),
                   labels=labels[::tick_frequency], rotation=90)
        plt.yticks(np.arange(0, len(labels), tick_frequency),
               labels=labels[::tick_frequency])

    if title is not None:
        plt.title(title, fontsize=36)
    else:
        plt.title('Reordered Connections Similarity Matrix', fontsize=36)
    
    # Add a colorbar to show the mapping of colors to similarity values.
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    
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
    
    

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


# Original function for getting clusters at a single distance threshold
def get_clusters_at_distance(linkage_matrix, n_samples, distance_threshold):
    """
    Extract clusters at a specific distance threshold.
    
    Parameters:
    -----------
    linkage_matrix : numpy.ndarray
        The linkage matrix from scipy's hierarchical clustering
    n_samples : int
        The number of original data points
    distance_threshold : float
        The distance threshold to form clusters
        
    Returns:
    --------
    list
        List of lists, where each inner list contains indices of samples in a cluster
    dict
        Dictionary mapping cluster label to list of sample indices
    """
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    unique_clusters = np.unique(cluster_labels)
    
    clusters = []
    cluster_dict = {}
    for i in unique_clusters:
        cluster_indices = np.where(cluster_labels == i)[0]
        clusters.append(cluster_indices.tolist())
        cluster_dict[i] = cluster_indices.tolist()
    
    return clusters, cluster_dict


def create_proper_sparse_tree(embeddings, linkage_matrix=None, n_samples=None, distance_thresholds=None):
    """
    Create a properly structured sparse tree with parents->children 
    with progressively lower distances.
    
    Parameters:
    -----------
    linkage_matrix : numpy.ndarray
        The linkage matrix from scipy's hierarchical clustering
    n_samples : int
        The number of original data points
    distance_thresholds : list
        A sorted list of distance thresholds (from largest to smallest)
        
    Returns:
    --------
    networkx.DiGraph
        A directed graph representing the sparse hierarchical structure
    """
    if linkage_matrix is None:  
        linkage_matrix = linkage(embeddings, method='ward', metric='euclidean')
    if n_samples is None:
        n_samples = embeddings.shape[0]
    if distance_thresholds is None:
        distance_thresholds = sorted(linkage_matrix[:, 2], reverse=True)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add original data points as leaf nodes
    for i in range(n_samples):
        G.add_node(i, type='sample', level='leaf')
    
    # Generate clusters at each threshold
    all_clusters = {}
    for threshold in distance_thresholds:
        cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
        
        # Group samples by cluster
        clusters = {}
        for i in range(n_samples):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(i)
        
        all_clusters[threshold] = clusters
    
    # Create a unique ID for each cluster
    cluster_id_counter = n_samples
    cluster_node_ids = {}  # Maps (threshold, cluster_id) to node_id
    
    # Build the tree from bottom up (smallest threshold first)
    for threshold in reversed(distance_thresholds):
        clusters = all_clusters[threshold]
        
        for cluster_id, samples in clusters.items():
            # Create a unique node ID for this cluster
            node_id = cluster_id_counter
            cluster_id_counter += 1
            
            # Store the mapping
            cluster_node_ids[(threshold, cluster_id)] = node_id
            
            # Add the cluster node
            G.add_node(node_id, 
                      type='cluster',
                      distance_threshold=threshold,
                      size=len(samples),
                      samples=samples)
            
            # If this is the smallest threshold, connect to samples directly
            if threshold == distance_thresholds[-1]:
                for sample in samples:
                    G.add_edge(node_id, sample)
    
    # Now connect clusters across thresholds
    for i in range(len(distance_thresholds) - 1):
        higher_threshold = distance_thresholds[i]
        lower_threshold = distance_thresholds[i + 1]
        
        higher_clusters = all_clusters[higher_threshold]
        lower_clusters = all_clusters[lower_threshold]
        
        # For each higher threshold cluster, find its children in the lower threshold
        for higher_cluster_id, higher_samples in higher_clusters.items():
            higher_node_id = cluster_node_ids[(higher_threshold, higher_cluster_id)]
            
            # Find all lower clusters that contain samples from this higher cluster
            children = set()
            for lower_cluster_id, lower_samples in lower_clusters.items():
                # Check if there's overlap between the sample sets
                if any(sample in higher_samples for sample in lower_samples):
                    lower_node_id = cluster_node_ids[(lower_threshold, lower_cluster_id)]
                    children.add(lower_node_id)
            
            # Connect the higher cluster to its children
            for child_id in children:
                G.add_edge(higher_node_id, child_id)
    
    return G






##
## metrics
##
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster, cophenet, inconsistent
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances


def find_optimal_tree_thresholds(embeddings, method='ward', metric='euclidean', 
                                 min_clusters=2, max_clusters=None, 
                                 n_thresholds=3, min_samples_per_cluster=5):
    """
    Find optimal distance thresholds for creating a sparse hierarchical tree.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        The embedded data with shape (n_samples, n_features)
    method : str, default='ward'
        The linkage method for hierarchical clustering
    metric : str, default='euclidean'
        The distance metric to use
    min_clusters : int, default=2
        Minimum number of clusters to consider
    max_clusters : int, default=None
        Maximum number of clusters to consider (defaults to sqrt(n_samples))
    n_thresholds : int, default=3
        Number of distance thresholds to select
    min_samples_per_cluster : int, default=5
        Minimum samples required for a valid cluster
        
    Returns:
    --------
    tuple
        (optimal_thresholds, linkage_matrix, evaluation_metrics)
    """
    n_samples = embeddings.shape[0]
    
    # Set default max_clusters if not provided
    if max_clusters is None:
        max_clusters = min(int(np.sqrt(n_samples)), 30)  # Reasonable upper limit
    
    # Compute linkage matrix
    distances = pdist(embeddings, metric=metric)
    Z = linkage(distances, method=method)
    
    # Get the cophenetic correlation coefficient
    c, coph_dists = cophenet(Z, distances)
    print(f"Cophenetic Correlation Coefficient: {c:.4f}")
    
    # Get all possible thresholds from the linkage matrix
    all_thresholds = sorted(Z[:, 2], reverse=True)
    
    # Calculate metrics for different numbers of clusters
    metrics = []
    cluster_counts = range(min_clusters, max_clusters + 1)
    silhouette_scores = []
    
    for n_clusters in cluster_counts:
        threshold = _get_threshold_for_n_clusters(Z, n_samples, n_clusters)
        labels = fcluster(Z, threshold, criterion='distance')
        
        # Skip if any cluster is too small
        cluster_sizes = Counter(labels)
        if min(cluster_sizes.values()) < min_samples_per_cluster:
            silhouette_scores.append(-1)  # Invalid score
            continue
            
        # Calculate silhouette score if more than one cluster
        if len(np.unique(labels)) > 1:
            try:
                sil_score = silhouette_score(embeddings, labels, metric=metric)
            except:
                sil_score = -1  # In case of errors
        else:
            sil_score = -1  # Invalid for only one cluster
            
        silhouette_scores.append(sil_score)
        
        # Calculate inconsistency for each merge
        incons = inconsistent(Z)
        mean_incons = np.mean(incons[:, 3])
        
        metrics.append({
            'n_clusters': n_clusters,
            'threshold': threshold,
            'silhouette': sil_score,
            'inconsistency': mean_incons
        })
    
    # Find good candidates using multiple methods
    candidates = []
    
    # 1. Use silhouette score peaks
    valid_scores = [(i, score) for i, score in enumerate(silhouette_scores) if score > 0]
    if valid_scores:
        indices, scores = zip(*valid_scores)
        # Find local maxima in silhouette scores
        for i in range(1, len(scores) - 1):
            if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                cluster_num = cluster_counts[indices[i]]
                threshold = _get_threshold_for_n_clusters(Z, n_samples, cluster_num)
                candidates.append(threshold)
    
    # 2. Use the elbow method on cluster hierarchy
    try:
        x = np.array(list(range(len(Z) - max_clusters + 1, len(Z) - min_clusters + 2)))
        y = Z[x, 2]
        
        if len(x) > 2:  # Need at least 3 points for elbow detection
            kneedle = KneeLocator(x, y, S=1.0, curve='convex', direction='decreasing')
            if kneedle.knee is not None:
                knee_idx = kneedle.knee
                n_clusters_knee = len(Z) + 2 - knee_idx
                threshold_knee = _get_threshold_for_n_clusters(Z, n_samples, n_clusters_knee)
                candidates.append(threshold_knee)
    except:
        pass  # If KneeLocator fails, just skip this method
    
    # 3. Use inconsistency coefficient
    incons = inconsistent(Z)
    incons_thresholds = []
    for i in range(len(Z)):
        if incons[i, 3] > 0.9:  # High inconsistency suggests significant merge
            incons_thresholds.append(Z[i, 2])
    
    if incons_thresholds:
        candidates.extend(incons_thresholds[:3])  # Add top inconsistent thresholds
    
    # Add candidates for two extremes: root and fine-grained
    # Root node (2 clusters)
    root_threshold = _get_threshold_for_n_clusters(Z, n_samples, 2)
    candidates.append(root_threshold)
    
    # Very fine-grained level
    fine_threshold = _get_threshold_for_n_clusters(Z, n_samples, min(max_clusters, int(n_samples/10)))
    candidates.append(fine_threshold)
    
    # Sort, deduplicate and select best candidates
    candidates = sorted(set(candidates), reverse=True)
    
    # If we don't have enough candidates, add thresholds at equal intervals
    if len(candidates) < n_thresholds:
        max_thresh = Z[-1, 2]  # Highest threshold
        min_thresh = Z[0, 2]   # Lowest non-zero threshold
        step = (max_thresh - min_thresh) / (n_thresholds + 1)
        
        # Generate evenly spaced thresholds
        even_thresholds = [max_thresh - step * (i + 1) for i in range(n_thresholds)]
        candidates.extend(even_thresholds)
        candidates = sorted(set(candidates), reverse=True)
    
    # Select the final thresholds
    if len(candidates) <= n_thresholds:
        optimal_thresholds = candidates
    else:
        # Prioritize keeping the root and fine-grained level, and select others to maximize spacing
        optimal_thresholds = [candidates[0]]  # Start with root
        
        # If we need more than 2 thresholds, select intermediate ones
        if n_thresholds > 2:
            remaining = candidates[1:-1]
            step = len(remaining) // (n_thresholds - 2)
            
            if step > 0:
                for i in range(0, len(remaining), step):
                    if len(optimal_thresholds) < n_thresholds - 1:
                        optimal_thresholds.append(remaining[i])
        
        # Add the finest level if we have room
        if len(optimal_thresholds) < n_thresholds:
            optimal_thresholds.append(candidates[-1])
    
    # Ensure we have exactly n_thresholds values
    optimal_thresholds = sorted(optimal_thresholds[:n_thresholds], reverse=True)
    
    return optimal_thresholds, Z, metrics

def _get_threshold_for_n_clusters(Z, n_samples, n_clusters):
    """
    Get the distance threshold that yields n_clusters.
    
    Parameters:
    -----------
    Z : numpy.ndarray
        The linkage matrix
    n_samples : int
        The number of samples
    n_clusters : int
        The desired number of clusters
        
    Returns:
    --------
    float
        The distance threshold
    """
    if n_clusters <= 1:
        return Z[-1, 2] + 0.1  # Slightly higher than the highest threshold
    elif n_clusters >= n_samples:
        return 0  # Zero distance threshold for n_samples clusters
    else:
        # The (n_samples - n_clusters)th merge in Z corresponds to the threshold
        merge_idx = n_samples - n_clusters
        if merge_idx < len(Z):
            return Z[merge_idx, 2]
        else:
            return 0


def find_optimal_distances(embeddings, linkage_matrix=None, method='ward', 
                          min_levels=3, max_levels=7, 
                          min_branching_factor=2, candidate_thresholds=None,
                          n_top_candidates=100, verbose=True):
    """
    Find the optimal set of distance thresholds for creating a sparse hierarchical tree.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        The embedded data with shape (n_samples, n_features)
    linkage_matrix : numpy.ndarray, optional
        Pre-computed linkage matrix. If None, it will be computed
    method : str, default='ward'
        The linkage method if linkage_matrix is not provided
    min_levels : int, default=3
        Minimum number of hierarchical levels to consider
    max_levels : int, default=7
        Maximum number of hierarchical levels to consider
    min_branching_factor : int, default=2
        Minimum average branching factor required at each level
    candidate_thresholds : list, optional
        List of candidate distance thresholds to consider. If None, they will be automatically determined
    n_top_candidates : int, default=100
        Number of top candidate combinations to evaluate in detail
    verbose : bool, default=True
        Whether to print progress information
    
    Returns:
    --------
    tuple
        (optimal_thresholds, best_score, metrics_dict, all_candidates_results)
    """
    n_samples = embeddings.shape[0]
    
    # Compute linkage matrix if not provided
    if linkage_matrix is None:
        if verbose:
            print("Computing linkage matrix...")
        linkage_matrix = linkage(embeddings, method=method)
    
    # Extract candidate thresholds if not provided
    if candidate_thresholds is None:
        # Extract unique meaningful thresholds from the linkage matrix
        unique_heights = np.unique(linkage_matrix[:, 2])
        
        # Filter out very small and very large thresholds
        min_height = np.percentile(unique_heights, 5)
        max_height = np.percentile(unique_heights, 95)
        candidate_thresholds = unique_heights[(unique_heights >= min_height) & 
                                              (unique_heights <= max_height)]
        
        # Take a reasonable number of evenly spaced thresholds if there are too many
        if len(candidate_thresholds) > 30:
            indices = np.linspace(0, len(candidate_thresholds)-1, 30, dtype=int)
            candidate_thresholds = candidate_thresholds[indices]
        
        if verbose:
            print(f"Generated {len(candidate_thresholds)} candidate thresholds")
    
    # Compute pairwise distances for cophenetic correlation
    distances = pdist(embeddings)
    
    # Pre-compute metrics that don't depend on the specific threshold combination
    cophenetic_dists = cophenet(linkage_matrix, distances)[0]
    cophenetic_corr = np.corrcoef(distances, cophenetic_dists)[0, 1]
    
    # Generate all possible combinations of thresholds for each level count
    all_candidate_combinations = []
    for n_levels in range(min_levels, min(max_levels + 1, len(candidate_thresholds) + 1)):
        # Take evenly spaced thresholds from the candidates
        indices = np.linspace(0, len(candidate_thresholds)-1, n_levels, dtype=int)
        evenly_spaced = sorted(candidate_thresholds[indices], reverse=True)
        all_candidate_combinations.append(evenly_spaced)
        
        # Also consider combinations of thresholds where we take more from areas with higher gradient
        # This is to ensure we have more levels where the data structure changes more rapidly
        gradients = np.diff(sorted(candidate_thresholds))
        gradient_weighted_indices = np.argsort(-gradients)[:n_levels]
        if len(gradient_weighted_indices) >= n_levels:
            weighted_thresholds = sorted([candidate_thresholds[i] for i in gradient_weighted_indices], reverse=True)
            if weighted_thresholds not in all_candidate_combinations:
                all_candidate_combinations.append(weighted_thresholds)
    
    # Add all possible combinations of a smaller number of thresholds (for efficiency, limit to top N)
    for n_levels in range(min_levels, min(max_levels + 1, len(candidate_thresholds) + 1)):
        # Adding some random combinations for more diversity
        all_combs = list(itertools.combinations(candidate_thresholds, n_levels))
        if len(all_combs) > 50:  # Limit to avoid excessive computation
            np.random.seed(42)
            random_indices = np.random.choice(len(all_combs), 50, replace=False)
            selected_combs = [all_combs[i] for i in random_indices]
        else:
            selected_combs = all_combs
            
        for comb in selected_combs:
            sorted_comb = sorted(comb, reverse=True)
            if sorted_comb not in all_candidate_combinations:
                all_candidate_combinations.append(sorted_comb)
    
    if verbose:
        print(f"Evaluating {len(all_candidate_combinations)} threshold combinations...")
    
    # First pass: Quick evaluation of candidates
    candidate_scores = []
    for thresholds in all_candidate_combinations:
        # Calculate initial metrics
        score, metrics = _evaluate_threshold_set_quick(thresholds, linkage_matrix, n_samples, min_branching_factor)
        candidate_scores.append((thresholds, score, metrics))
    
    # Sort candidates by score
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top candidates for detailed evaluation
    top_candidates = candidate_scores[:min(n_top_candidates, len(candidate_scores))]
    
    if verbose:
        print(f"Performing detailed evaluation on top {len(top_candidates)} candidates...")
    
    # Second pass: Detailed evaluation of top candidates
    detailed_results = []
    for thresholds, initial_score, _ in tqdm(top_candidates) if verbose else top_candidates:
        # Create sparse tree
        sparse_tree = create_proper_sparse_tree(linkage_matrix, n_samples, thresholds)
        
        # Calculate detailed metrics
        score, metrics = _evaluate_threshold_set_detailed(
            thresholds, sparse_tree, linkage_matrix, embeddings, 
            cophenetic_corr, n_samples, min_branching_factor
        )
        
        detailed_results.append((thresholds, score, metrics))
    
    # Sort by final score
    detailed_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best threshold set
    optimal_thresholds, best_score, best_metrics = detailed_results[0]
    
    if verbose:
        print(f"Optimal thresholds: {optimal_thresholds}")
        print(f"Best score: {best_score:.4f}")
        print("Best metrics:")
        for k, v in best_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    return optimal_thresholds, best_score, best_metrics, detailed_results

def _evaluate_threshold_set_quick(thresholds, linkage_matrix, n_samples, min_branching_factor):
    """
    Quick evaluation of a threshold set based on basic criteria.
    """
    # Check if we have a root node (first threshold should create very few clusters)
    root_labels = fcluster(linkage_matrix, thresholds[0], criterion='distance')
    n_root_clusters = len(np.unique(root_labels))
    has_root = n_root_clusters < 5  # Should have very few top-level clusters
    
    # Check branching factor at each level
    branching_factors = []
    prev_n_clusters = n_root_clusters
    
    for i in range(1, len(thresholds)):
        cluster_labels = fcluster(linkage_matrix, thresholds[i], criterion='distance')
        n_clusters = len(np.unique(cluster_labels))
        
        if prev_n_clusters > 0:
            branching = n_clusters / prev_n_clusters
            branching_factors.append(branching)
        
        prev_n_clusters = n_clusters
    
    # Check for leaf nodes (last threshold should create many small clusters)
    n_leaf_clusters = prev_n_clusters
    has_leaves = n_leaf_clusters > n_samples / 10  # Should have many leaf clusters
    
    # Calculate average branching factor
    avg_branching = np.mean(branching_factors) if branching_factors else 0
    good_branching = avg_branching >= min_branching_factor
    
    # Calculate sparsity (ratio of levels to max possible levels)
    sparsity = len(thresholds) / (n_samples - 1)
    good_sparsity = sparsity < 0.1  # Should be much sparser than full tree
    
    # Basic quality score
    score = (2 * has_root + 2 * has_leaves + 3 * good_branching + 3 * good_sparsity) / 10
    
    # Add a small bonus for more balanced level distribution
    if branching_factors:
        balance = 1 - np.std(branching_factors) / np.mean(branching_factors)
        score += 0.1 * max(0, balance)
    
    metrics = {
        'n_levels': len(thresholds),
        'n_root_clusters': n_root_clusters, 
        'n_leaf_clusters': n_leaf_clusters,
        'avg_branching': avg_branching,
        'sparsity': sparsity,
        'has_root': has_root,
        'has_leaves': has_leaves,
        'good_branching': good_branching,
        'good_sparsity': good_sparsity
    }
    
    return score, metrics

def _evaluate_threshold_set_detailed(thresholds, sparse_tree, linkage_matrix, 
                                    embeddings, cophenetic_corr, n_samples, 
                                    min_branching_factor):
    """
    Detailed evaluation of a threshold set based on tree structure and clustering metrics.
    """
    # Get basic metrics first
    base_score, metrics = _evaluate_threshold_set_quick(
        thresholds, linkage_matrix, n_samples, min_branching_factor
    )
    
    # Calculate tree structure metrics
    n_nodes = sparse_tree.number_of_nodes()
    n_edges = sparse_tree.number_of_edges()
    
    # Check if tree has a single root
    roots = [n for n, d in sparse_tree.in_degree() if d == 0]
    has_single_root = len(roots) == 1
    
    # Check leaf distribution
    leaves = [n for n, d in sparse_tree.out_degree() if d == 0]
    leaf_ratio = len(leaves) / n_nodes
    
    # Calculate level distribution metrics
    levels = {}
    for node in sparse_tree.nodes():
        if sparse_tree.nodes[node].get('type') == 'cluster':
            level = sparse_tree.nodes[node].get('level', 0)
            if level not in levels:
                levels[level] = 0
            levels[level] += 1
    
    level_counts = [levels.get(i, 0) for i in range(max(levels.keys()) + 1)] if levels else [0]
    level_entropy = -sum((c/sum(level_counts)) * np.log2(c/sum(level_counts)) 
                        for c in level_counts if c > 0)
    
    # Calculate structural metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Silhouette score at each level
            silhouette_scores = []
            for threshold in thresholds:
                if len(np.unique(embeddings)) > 1:  # Needs at least 2 different points
                    cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
                    if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters
                        try:
                            ss = silhouette_score(embeddings, cluster_labels)
                            silhouette_scores.append(ss)
                        except Exception:
                            silhouette_scores.append(-1)  # Error in calculation
            
            avg_silhouette = np.mean(silhouette_scores) if silhouette_scores else -1
        except Exception:
            avg_silhouette = -1
    
    # Calculate node increase ratio between levels
    node_increase_ratios = []
    prev_count = levels.get(0, 1)  # Start with root level
    for i in range(1, max(levels.keys()) + 1) if levels else []:
        curr_count = levels.get(i, 0)
        if prev_count > 0:
            node_increase_ratios.append(curr_count / prev_count)
        prev_count = curr_count
    
    avg_increase_ratio = np.mean(node_increase_ratios) if node_increase_ratios else 0
    
    # Update metrics dictionary
    metrics.update({
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'has_single_root': has_single_root,
        'leaf_ratio': leaf_ratio,
        'level_entropy': level_entropy,
        'avg_silhouette': avg_silhouette,
        'cophenetic_corr': cophenetic_corr,
        'avg_increase_ratio': avg_increase_ratio
    })
    
    # Calculate final score with weightings for different factors
    score = (
        0.15 * (1 if has_single_root else 0) +  # Single root
        0.15 * min(1, leaf_ratio / 0.7) +  # Good leaf distribution
        0.15 * min(1, level_entropy / 2) +  # Good level entropy
        0.15 * min(1, avg_increase_ratio / min_branching_factor) +  # Good branching
        0.20 * (avg_silhouette + 1) / 2 +  # Silhouette score (normalized to 0-1)
        0.10 * cophenetic_corr +  # Cophenetic correlation
        0.10 * base_score  # Basic criteria
    )
    
    return score, metrics


def evaluate_cophenetic_correlation(embeddings, linkage_matrix, metric='euclidean'):
    """
    Calculate the cophenetic correlation coefficient to measure how well
    the hierarchical clustering preserves the original distances.
    
    Returns a value between -1 and 1, where higher values indicate better fit.
    """
    original_distances = pdist(embeddings, metric=metric)
    coph_corr, _ = cophenet(linkage_matrix, original_distances)
    return coph_corr

def plot_elbow_method(linkage_matrix):
    """
    Plot the distance at each merge step to identify significant jumps,
    which can indicate appropriate thresholds.
    """
    last_distances = linkage_matrix[:, 2]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(last_distances)+1), last_distances, 'bo-')
    plt.title('Elbow Method for Optimal Distance Thresholds')
    plt.xlabel('Merge Step')
    plt.ylabel('Distance')
    plt.grid(True)
    
    # Calculate acceleration (second derivative)
    first_diff = np.diff(last_distances)
    second_diff = np.diff(first_diff)
    
    # Find peaks in acceleration (large jumps)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(second_diff, height=0.05*max(second_diff))
    
    # Mark potential thresholds
    potential_thresholds = [last_distances[i+1] for i in peaks]
    plt.plot(peaks+2, [last_distances[i+1] for i in peaks], 'ro', markersize=8)
    
    for i, peak in enumerate(peaks):
        plt.annotate(f'{last_distances[peak+1]:.2f}', 
                     xy=(peak+2, last_distances[peak+1]), 
                     xytext=(5, 5), textcoords='offset points')
    
    plt.show()
    
    return potential_thresholds


def evaluate_thresholds_silhouette(embeddings, linkage_matrix, thresholds):
    """
    Calculate silhouette scores for clusters formed at different distance thresholds.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        The embedded data
    linkage_matrix : numpy.ndarray
        The linkage matrix from hierarchical clustering
    thresholds : list
        List of distance thresholds to evaluate
        
    Returns:
    --------
    dict
        Dictionary mapping thresholds to silhouette scores and cluster counts
    """
    results = {}
    
    for threshold in thresholds:
        # Get cluster assignments at this threshold
        cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
        
        # Need at least 2 clusters for silhouette score
        n_clusters = len(np.unique(cluster_labels))
        if n_clusters < 2:
            results[threshold] = {'silhouette': None, 'n_clusters': n_clusters}
            continue
        
        # Calculate silhouette score
        try:
            sil_score = silhouette_score(embeddings, cluster_labels)
            results[threshold] = {'silhouette': sil_score, 'n_clusters': n_clusters}
        except:
            results[threshold] = {'silhouette': None, 'n_clusters': n_clusters}
    
    return results

def gap_statistic(embeddings, max_clusters=20, n_refs=10, random_seed=42):
    """
    Calculate the Gap statistic for determining optimal number of clusters.
    """
    np.random.seed(random_seed)
    
    # Calculate dispersion for actual data
    ondata = np.zeros(max_clusters)
    for k in range(1, max_clusters+1):
        clustering = AgglomerativeClustering(n_clusters=k).fit(embeddings)
        labels = clustering.labels_
        
        # Calculate within-cluster dispersion
        within = 0
        for i in range(k):
            cluster_points = embeddings[labels == i]
            if len(cluster_points) > 1:
                within += np.sum(pairwise_distances(cluster_points)) / (2 * len(cluster_points))
        
        ondata[k-1] = np.log(within) if within > 0 else 0
    
    # Calculate dispersion for reference data
    ref_dispersions = np.zeros((n_refs, max_clusters))
    for i in range(n_refs):
        # Create random reference data
        ref_min = embeddings.min(axis=0)
        ref_max = embeddings.max(axis=0)
        ref_data = np.random.uniform(ref_min, ref_max, embeddings.shape)
        
        for k in range(1, max_clusters+1):
            clustering = AgglomerativeClustering(n_clusters=k).fit(ref_data)
            labels = clustering.labels_
            
            # Calculate within-cluster dispersion
            within = 0
            for j in range(k):
                cluster_points = ref_data[labels == j]
                if len(cluster_points) > 1:
                    within += np.sum(pairwise_distances(cluster_points)) / (2 * len(cluster_points))
            
            ref_dispersions[i, k-1] = np.log(within) if within > 0 else 0
    
    # Calculate gap statistic
    gap = ref_dispersions.mean(axis=0) - ondata
    
    # Calculate standard deviation of reference dispersions
    sdk = np.std(ref_dispersions, axis=0) * np.sqrt(1 + 1/n_refs)
    
    # Find optimal number of clusters
    optimal_clusters = 1
    for k in range(1, max_clusters):
        if gap[k-1] >= gap[k] - sdk[k]:
            optimal_clusters = k
            break
    
    return optimal_clusters, gap


def analyze_inconsistency(linkage_matrix, depth=2):
    """
    Calculate inconsistency coefficients to identify natural divisions.
    Higher values indicate more distinct clusters.
    """
    incons = inconsistent(linkage_matrix, depth)
    
    # Plot the inconsistency coefficients
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(incons)+1), incons[:, 3], 'bo-')
    plt.title(f'Inconsistency Coefficients (depth={depth})')
    plt.xlabel('Merge Step')
    plt.ylabel('Inconsistency Coefficient')
    plt.grid(True)
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(incons[:, 3], height=0.7*max(incons[:, 3]))
    
    # Mark potential thresholds
    plt.plot(peaks+1, incons[peaks, 3], 'ro', markersize=8)
    
    # Get corresponding distance thresholds
    potential_thresholds = [linkage_matrix[i, 2] for i in peaks]
    
    for i, peak in enumerate(peaks):
        plt.annotate(f'{linkage_matrix[peak, 2]:.2f}', 
                     xy=(peak+1, incons[peak, 3]), 
                     xytext=(5, 5), textcoords='offset points')
    
    plt.show()
    
    return potential_thresholds

def find_optimal_thresholds(embeddings, linkage_matrix, metric='euclidean'):
    """
    Find optimal distance thresholds using multiple methods.
    """
    # Calculate cophenetic correlation
    coph_corr = evaluate_cophenetic_correlation(embeddings, linkage_matrix, metric)
    print(f"Cophenetic Correlation: {coph_corr:.4f}")
    
    # Find potential thresholds using elbow method
    print("\nElbow Method Analysis:")
    elbow_thresholds = plot_elbow_method(linkage_matrix)
    
    # Find potential thresholds using inconsistency analysis
    print("\nInconsistency Coefficient Analysis:")
    incons_thresholds = analyze_inconsistency(linkage_matrix, depth=2)
    
    # Combine all potential thresholds and sort
    all_thresholds = sorted(set(elbow_thresholds + incons_thresholds), reverse=True)
    
    # Evaluate with silhouette scores
    print("\nEvaluating potential thresholds with Silhouette Analysis:")
    silhouette_results = evaluate_thresholds_silhouette(embeddings, linkage_matrix, all_thresholds)
    
    # Display results
    print("\nRecommended thresholds:")
    for threshold in all_thresholds:
        result = silhouette_results[threshold]
        if result['silhouette'] is not None:
            print(f"Distance: {threshold:.2f}, Clusters: {result['n_clusters']}, "
                  f"Silhouette: {result['silhouette']:.4f}")
    
    # Provide recommended thresholds (top 3 by silhouette score)
    valid_results = [(t, r['silhouette']) for t, r in silhouette_results.items() if r['silhouette'] is not None]
    recommended = sorted(valid_results, key=lambda x: x[1], reverse=True)[:3]
    
    return [t for t, _ in recommended]
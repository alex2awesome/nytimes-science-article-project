import networkx as nx

def convert_sklearn_agglomerative_model_to_networkx(model, n_samples):
    # children_[i] = (left_child, right_child) that were merged at step i
    children = model.children_

    # 2) Build the directed graph
    G = nx.DiGraph()

    # a) Add leaf nodes
    for i in range(n_samples):
        G.add_node(i, indices=[i], size=1)

    # b) Add internal nodes and edges
    #    New node IDs run from n_samples to n_samples + len(children) - 1
    for step, (left, right) in enumerate(children):
        parent = n_samples + step

        # gather data-point indices under this cluster
        left_indices  = G.nodes[left]['indices']
        right_indices = G.nodes[right]['indices']
        all_indices   = left_indices + right_indices

        # add the new cluster node
        G.add_node(parent,
                   indices=all_indices,
                   size=len(all_indices))

        # connect parent -> its two children
        G.add_edge(parent, left)
        G.add_edge(parent, right)

    # The final merge is the true root
    root = n_samples + children.shape[0] - 1

    return G, root


def linkage_to_networkx(Z, n_samples):
    """
    Convert a linkage matrix to a NetworkX directed graph.
    
    Parameters:
    -----------
    Z : numpy.ndarray
        The linkage matrix from scipy's hierarchical clustering
    n_samples : int
        The number of original data points
    
    Returns:
    --------
    networkx.DiGraph
        A directed graph representing the hierarchical structure
    """
    G = nx.DiGraph()
    
    # Add original data points as nodes
    for i in range(n_samples):
        G.add_node(i, type='sample')
    
    # Add cluster nodes and edges
    for i, (node1, node2, dist, count) in enumerate(Z):
        # The new cluster index will be n_samples + i
        new_cluster_id = n_samples + i
        
        # Add the new cluster node
        G.add_node(new_cluster_id, type='cluster', distance=dist, size=count)
        
        # Add edges from the new cluster to its children
        for child in [node1, node2]:
            G.add_edge(new_cluster_id, child, weight=dist)
    
    return G

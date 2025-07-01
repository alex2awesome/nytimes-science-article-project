import networkx as nx
from tqdm.auto import tqdm
import numpy as np
from basic_util import call_openai
from prompts import LONG_SUMMARY_NODE_PROMPT, SINGLE_SUMMARY_NODE_PROMPT


# Prune the tree to include nodes with more than 5 descendants in their entire subtree
def prune_tree_by_subtree_size(G, min_descendants=5, max_depth=8):
    """
    Prune a tree graph (NetworkX DiGraph) so that only nodes whose entire subtree
    (all descendants) has more than `min_descendants` nodes are kept.

    Parameters:
    -----------
    G : networkx.DiGraph
        The input tree graph.
    min_descendants : int
        Minimum number of descendants required for a node to be kept.

    Returns:
    --------
    pruned_G : networkx.DiGraph
        A new graph that contains only nodes with more than `min_descendants` descendants.
    """    
    # Identify nodes where the count of descendants is greater than the threshold
    root_node = [n for n,d in G.in_degree() if d==0][0]
    keep_nodes = [node for node in G.nodes() 
                  if len(nx.descendants(G, node)) >= min_descendants
                  and (nx.shortest_path_length(G, root_node, node) <= max_depth)
                  # and len(list(G.successors(node))) > 1
                 ]
    
    
    # Create a subgraph with only the nodes that meet the criteria
    pruned_G = G.subgraph(keep_nodes).copy()
    return pruned_G

from collections import defaultdict
def propagate_data_up_tree(G, data_dict, label_name):
    """
    Propagate data up the tree from leaf nodes to root nodes.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The input tree graph.
    data_dict : dict
        A dictionary mapping node IDs to their respective data.
    label_name : str
        The name of the attribute to store the combined data.
        
    Returns:
    --------
    dict
        A dictionary where each node ID maps to a combined list of data from its subtree.
    """
    # Initialize a dictionary to store combined data for each node
    combined_data = defaultdict(list)

    # Function to recursively propagate data up the tree
    def propagate(node):
        # If the node is a leaf, start with its data
        if G.out_degree(node) == 0:
            combined_data[node] = data_dict.get(node, {})
        else:
            for child in G.successors(node):
                propagate(child)
                d = combined_data[child]
                if isinstance(d, dict):
                    combined_data[node].append(d)
                else:
                    combined_data[node].extend(d)
                
    
    # Start propagation from the root node
    root_node = get_root(G)
    propagate(root_node)
    
    # Attach the combined data to each node in the graph
    for node, data in combined_data.items():
        G.nodes[node][label_name] = data
    
    return combined_data



def get_root(G):
    """
    Returns the root of the tree (node with no incoming edges).
    Assumes there is exactly one such node.
    """
    for node in G.nodes():
        if G.in_degree(node) == 0:
            return node
    raise ValueError("No root found in the graph.")


def get_all_leaf_nodes(G):
    """
    Returns all leaf nodes in the tree.
    """
    return [node for node in G.nodes() if G.out_degree(node) == 0]


def compute_subtree_sizes(G, node, leaf_node_counts=None, count_attr='subtree_size', recount=False):
    """
    Recursively computes the total number of datapoints in the subtree
    rooted at 'node' and stores it as an attribute 'subtree_size'.
    
    For a leaf node, it uses the node's 'count' attribute (default 0 if missing).
    """
    if (recount) or (count_attr not in G.nodes[node]):
        # leaf node
        if G.out_degree(node) == 0:
            size = leaf_node_counts[node]
            G.nodes[node][count_attr] = size
            return size
    
        # For inner nodes, sum the sizes of all children
        size = 0
        for child in G.successors(node):
            size += compute_subtree_sizes(G, child, leaf_node_counts, count_attr)
        G.nodes[node][count_attr] = size
    return G.nodes[node][count_attr]


def label_hierarchical_tree_based_on_node_labels(
    G, leaf_node_counts=None,
    min_descendants=1,
    max_depth=13
):
    """
    Cluster a hierarchical tree using the specified method and parameters.

    Parameters:
    -----------
    G : networkx.DiGraph
        The input tree graph.
    leaf_node_counts : dict, optional
        A dictionary mapping node IDs to their respective counts. If None, each node is assumed
        to have a count of 1.
    min_descendants : int, default=1
        The minimum number of descendants a node must have to be retained in the pruned tree.
    max_depth : int, default=13
        The maximum depth of the tree to be considered during pruning.

    Returns:
    --------
    networkx.DiGraph
        A pruned and labeled hierarchical tree represented as a directed graph.
    """    
    # compute subtree sizes
    if leaf_node_counts is None:
        leaf_node_counts = {k: 1 for k in G.nodes()}

    root_node = get_root(G)
    compute_subtree_sizes(G, root_node, leaf_node_counts)
    
    # prune the graph and label inner tree
    pruned_G = prune_tree_by_subtree_size(G, min_descendants=min_descendants, max_depth=max_depth)
    inner_nodes = [node for node in G.nodes if G.out_degree(node) > 0]
    pruned_inner_nodes = [node for node in inner_nodes if node in pruned_G.nodes]
    num_samples_to_label = 10
    inner_node_label_dict = {}
    inner_node_summary_dict = {}
    
    # label the inner tree
    for p in tqdm(inner_nodes):
        labeled_nodes_for_target_node = list(filter(lambda x: 'label' in G.nodes[x], G.successors(p)))
        node_sizes = list(map(lambda x: G.nodes[x]['subtree_size'], labeled_nodes_for_target_node))
        sample_nodes = np.random.choice(labeled_nodes_for_target_node, num_samples_to_label, p=np.array(node_sizes) / sum(node_sizes))
        sample_labels = list(map(lambda x: G.nodes[x]['label'], sample_nodes))
        
        # generate a definition for the node
        generate_definition_prompt = LONG_SUMMARY_NODE_PROMPT.format(summaries='\n\n'.join(sample_labels))
        summarized_label = call_openai(generate_definition_prompt)
        
        # summarize the definition into a single label
        single_label_prompt = SINGLE_SUMMARY_NODE_PROMPT.format(summary=summarized_label)
        label_to_plot = call_openai(single_label_prompt)
        nx.set_node_attributes(G, {p: summarized_label}, 'summary')
        nx.set_node_attributes(G, {p: label_to_plot}, 'label_to_plot')
        nx.set_node_attributes(pruned_G, {p: summarized_label}, 'summary')
        nx.set_node_attributes(pruned_G, {p: label_to_plot}, 'label_to_plot')
        inner_node_summary_dict[p] = summarized_label
        inner_node_label_dict[p] = label_to_plot
    
    return G, pruned_G, inner_node_summary_dict, inner_node_label_dict
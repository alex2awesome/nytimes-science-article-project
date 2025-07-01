# agglomerative_clustering.py

import sys
import pandas as pd
import datetime
import os
import warnings
import json
import os
from openai import OpenAI
import ast
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from tqdm.auto import tqdm
import argparse

# Insert path for create_trees module
# TODO: Make this path dynamic or configurable
sys.path.insert(0, '../../../google-research/reasoning-schema/make_label_hierarchy/')
import create_trees as c

# -----------------
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# TODO: Abstract API key loading
os.environ['OPENAI_API_KEY'] = open('/Users/spangher/.openai-bloomberg-project-key.txt').read().strip()
warnings.simplefilter(action='ignore')
client = OpenAI()


def simple_json_parse(j):
    try:
        return json.loads(j)
    except:
        try:
            return ast.literal_eval(j)
        except:
            return None


def prompt_openai(prompt, model='gpt-4o-mini'):
    completion = client.chat.completions.create(
        model=model,
        messages=[{'''role''': '''user''', '''content''': prompt}]
    )
    return completion.choices[0].message.content


# --- Constants --- #
NODE_PROMPT = '''You are a helpful assistant. I will give you a list of news headlines and summaries. You will summarize them and 
return a single, specific topic label and a description in the following forward: "Label": Description. 
Please condense them into a single, specific label. Be precise and concise. Ignore labels that are too generic.
Please return just one 2-3 word label and one description.

Here are some examples of how I want my outputs:
<examples>
output:
"Space Industry": These articles cover missions planned either by government or private companies into space.

output:
"Heart Health": The step covers medical advances impacting cardiovascular systems.
</examples>

Now it's your turn. Here are the article headlines and summaries:
<articles>
{articles_and_summaries}
</articles>

output:
'''

# --- Main Logic --- #
def run_agglomerative_clustering(input_file_path, output_dir_path):
    '''
    Performs agglomerative clustering on the input data.

    Sample input_file_path format (JSONL or JSON):
    {"headline": "Headline 1", "summary": "Summary 1"}
    {"headline": "Headline 2", "summary": "Summary 2"}
    ...
    '''

    # Load data
    # TODO: Add error handling for file loading and parsing
    science_articles_df = pd.read_json(input_file_path)

    # Create embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = science_articles_df.pipe(lambda df: df['headline']).tolist() # Using only headlines as in the notebook
    embeddings = model.encode(sentences)

    # K-Means clustering
    # TODO: Make n_clusters configurable
    kmeans = KMeans(n_clusters=128, n_init="auto", random_state=42).fit(embeddings) # Added random_state for reproducibility
    science_articles_df['kmeans_128_cluster_center'] = kmeans.labels_

    # Generate cluster summaries
    cluster_summaries = {}
    clusters = science_articles_df['kmeans_128_cluster_center'].drop_duplicates().tolist()
    for cluster in tqdm(clusters, desc="Generating Cluster Summaries"):
        summs = (
            science_articles_df
            .loc[lambda df: df['kmeans_128_cluster_center'] == cluster]
            .apply(lambda x: 'Headline: ' + x['headline'] + ' Summary: ' + x['summary'], axis=1)
        )
        input_str = '\n'.join(summs.tolist())
        prompt = NODE_PROMPT.format(articles_and_summaries=input_str)
        # TODO: Make OpenAI model configurable
        summary = prompt_openai(prompt, model='gpt-4o') # Using gpt-4o as per user's original notebook, adjust if needed
        cluster_summaries[cluster] = summary

    cluster_summaries_df = pd.Series(cluster_summaries).to_frame('label')
    cluster_summaries_df = (
        cluster_summaries_df
        .reset_index().rename(columns={'index': 'node_id'}).sort_values('node_id')
        .reset_index(drop=True)
    )

    # Save initial K-Means labels
    os.makedirs(output_dir_path, exist_ok=True)
    kmeans_labels_output_path = os.path.join(output_dir_path, 'kmeans-initial-labels.csv')
    cluster_summaries_df.to_csv(kmeans_labels_output_path, index=False)
    print(f"K-Means initial labels saved to: {kmeans_labels_output_path}")

    # Prepare for hierarchical clustering
    leaf_node_counts = science_articles_df['kmeans_128_cluster_center'].value_counts().to_dict()
    kmeans_centers = kmeans.cluster_centers_

    # Hierarchical clustering
    # TODO: Make min_cluster_size configurable
    G, pruned_G, inner_node_label_dict = c.cluster_hierarchical_tree(
        kmeans_centers, cluster_summaries_df, leaf_node_counts, min_cluster_size=2
    )

    # TODO: Determine what to do with G, pruned_G, and inner_node_label_dict
    # For now, let's print some info about the pruned graph
    print(f"Number of nodes in pruned graph: {pruned_G.number_of_nodes()}")
    print(f"Number of edges in pruned graph: {pruned_G.number_of_edges()}")

    # Example of saving graph data (adjust as needed)
    # import networkx as nx
    # pruned_graph_output_path = os.path.join(output_dir_path, 'pruned_graph.gml')
    # nx.write_gml(pruned_G, pruned_graph_output_path)
    # print(f"Pruned graph saved to: {pruned_graph_output_path}")

    inner_node_label_df = pd.Series(inner_node_label_dict).to_frame('label')
    inner_node_labels_output_path = os.path.join(output_dir_path, 'inner_node_labels.csv')
    inner_node_label_df.to_csv(inner_node_labels_output_path)
    print(f"Inner node labels saved to: {inner_node_labels_output_path}")

    print("Agglomerative clustering process complete.")


def main():
    parser = argparse.ArgumentParser(description="Perform agglomerative clustering on input data.")
    parser.add_argument("input_file", help="Path to the input JSON or JSONL file.")
    parser.add_argument("output_dir", help="Directory to save the output files.")
    # TODO: Add arguments for configurable parameters like n_clusters, min_cluster_size, OpenAI model, etc.

    args = parser.parse_args()

    run_agglomerative_clustering(args.input_file, args.output_dir)

if __name__ == "__main__":
    main() 
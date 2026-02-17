"""
Step 3: Extend Coactivation Graph with Concepts
=================================================
Combines the base coactivation graph (Step 1) with
concept associations (Step 2) into the full Semantic
Coactivation Graph (SCAG).

Graph structure (following Meeting 4 slides):
  - Neuron nodes: (layer, neuron_id)
      Attributes: layer, block, inlayer_id, node_type='neuron'
      Patch metadata: top_images, top_activations (from Step 2)
  - Concept nodes: ('concept', concept_id)
      Attributes: concept_name, supercategory, node_type='concept'
  - Neuron↔Neuron edges: edge_type='coactivates_with', weight=ρ
  - Neuron→Concept edges: edge_type='activates_on', weight=IoU

The neuron→concept edge encodes the full n→p→c chain:
  - The patch (top-k% activation region) is implicit in the IoU score
  - Patch metadata is stored on the neuron node for traceability

Reads: base_coactivation_graph.graphml + concept_associations.pkl
Writes: extended_coactivation_graph.graphml

Usage:
    python step3_extend_graph.py
"""

import pickle
import json
import networkx as nx
import numpy as np

from config import (
    OUTPUT_DIR, BASE_GRAPH_PATH, EXTENDED_GRAPH_PATH, CONCEPT_PKL_PATH,
    CONCEPT_LAYER, TAU_ASSOC, COCO_ID_TO_NAME, SUPERCATEGORY_MAP,
    ensure_dirs, concept_name, supercategory,
)
from coactivation_utils import load_coactivation_graph


def load_base_graph():
    """Load the base coactivation graph from Step 1."""
    print(f"  Loading base graph: {BASE_GRAPH_PATH}")
    graph = load_coactivation_graph(
        folder=str(OUTPUT_DIR) + '/',
        file_name=BASE_GRAPH_PATH.name,
    )
    # Tag existing edges
    for u, v, data in graph.edges(data=True):
        if 'edge_type' not in data:
            data['edge_type'] = 'coactivates_with'
    # Tag existing nodes
    for n, data in graph.nodes(data=True):
        if 'node_type' not in data:
            data['node_type'] = 'neuron'
    print(f"    Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    return graph


def load_concept_data():
    """Load concept associations and patch metadata from Step 2."""
    print(f"  Loading concept data: {CONCEPT_PKL_PATH}")
    with open(CONCEPT_PKL_PATH, 'rb') as f:
        data = pickle.load(f)
    associations = data['associations']
    patch_metadata = data['patch_metadata']
    config = data.get('config', {})
    print(f"    Concepts: {len(associations)}, Layer: {config.get('layer', '?')}")
    return associations, patch_metadata, config


def extend_graph(graph, associations, patch_metadata, config):
    """
    Add concept nodes, neuron→concept edges, and patch metadata to graph.
    """
    layer = config.get('layer', CONCEPT_LAYER)

    # ----- 1. Add patch metadata to neuron nodes -----
    n_patched = 0
    for neuron_idx, meta in patch_metadata.items():
        node_id = (layer, neuron_idx)
        # GraphML can only store strings, so serialise lists as JSON
        if graph.has_node(str(node_id)) or graph.has_node(node_id):
            # Handle both tuple and string node IDs (GraphML converts to string)
            nid = node_id if graph.has_node(node_id) else str(node_id)
            graph.nodes[nid]['top_images'] = json.dumps(meta['top_images'])
            graph.nodes[nid]['top_activations'] = json.dumps(
                [round(float(v), 4) for v in meta['top_activations']]
            )
            n_patched += 1
    print(f"    Patch metadata added to {n_patched} neurons")

    # ----- 2. Add concept nodes -----
    n_concept_nodes = 0
    for cat_id in associations.keys():
        cname = concept_name(cat_id)
        supercat = supercategory(cname)
        node_id = ('concept', cat_id)
        graph.add_node(
            node_id,
            node_type='concept',
            concept_name=cname,
            concept_id=int(cat_id),
            supercategory=supercat,
        )
        n_concept_nodes += 1
    print(f"    Concept nodes added: {n_concept_nodes}")

    # ----- 3. Add neuron→concept edges -----
    n_edges = 0
    for cat_id, scores in associations.items():
        for neuron_idx, iou_score in enumerate(scores):
            if iou_score > TAU_ASSOC:
                neuron_node = (layer, neuron_idx)
                concept_node = ('concept', cat_id)
                graph.add_edge(
                    neuron_node,
                    concept_node,
                    edge_type='activates_on',
                    weight=float(round(iou_score, 6)),
                )
                n_edges += 1
    print(f"    Concept edges added: {n_edges} (IoU > {TAU_ASSOC})")

    return graph


def print_summary(graph):
    """Print summary statistics of the extended graph."""
    n_neurons = sum(1 for _, d in graph.nodes(data=True) if d.get('node_type') != 'concept')
    n_concepts = sum(1 for _, d in graph.nodes(data=True) if d.get('node_type') == 'concept')
    n_coact = sum(1 for _, _, d in graph.edges(data=True) if d.get('edge_type') == 'coactivates_with')
    n_concept_edges = sum(1 for _, _, d in graph.edges(data=True) if d.get('edge_type') == 'activates_on')

    print(f"\n  Extended Graph Summary:")
    print(f"    Neuron nodes:      {n_neurons}")
    print(f"    Concept nodes:     {n_concepts}")
    print(f"    Total nodes:       {graph.number_of_nodes()}")
    print(f"    Coactivation edges:{n_coact}")
    print(f"    Concept edges:     {n_concept_edges}")
    print(f"    Total edges:       {graph.number_of_edges()}")

    # Concepts with most detecting neurons
    concept_degree = {}
    for u, v, d in graph.edges(data=True):
        if d.get('edge_type') == 'activates_on':
            cname = graph.nodes[v].get('concept_name', str(v))
            concept_degree[cname] = concept_degree.get(cname, 0) + 1

    print(f"\n  Top 10 most-detected concepts:")
    for cname, deg in sorted(concept_degree.items(), key=lambda x: -x[1])[:10]:
        print(f"    {cname:20s}: {deg} neurons")


def main():
    print("=" * 60)
    print("  Step 3: Extend Graph with Concepts")
    print("=" * 60)

    graph = load_base_graph()
    associations, patch_metadata, config = load_concept_data()
    graph = extend_graph(graph, associations, patch_metadata, config)
    print_summary(graph)

    # Save
    ensure_dirs()
    nx.write_graphml(graph, str(EXTENDED_GRAPH_PATH))
    print(f"\n  Saved: {EXTENDED_GRAPH_PATH}")
    print("=" * 60)

    return graph


if __name__ == '__main__':
    main()
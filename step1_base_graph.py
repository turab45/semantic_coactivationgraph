"""
Step 1: Build Base Coactivation Graph
======================================
RQ1: Which neurons are correlated / fire together?

Pipeline (following Vitor's methodology):
  1. Extract activations from ResNet-18 on COCO images
  2. Compute ranks per neuron (Spearman)
  3. Compute pairwise Spearman correlations
  4. Build NetworkX DiGraph with threshold τ_coact

Uses: coactivation_utils.py (Vitor's code, untouched)
Reads: COCO images
Writes: results/base_coactivation_graph.graphml

Usage:
    python step1_base_graph.py              # Build from scratch
    python step1_base_graph.py --load       # Load existing graph
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CocoDetection
import pandas as pd
from pathlib import Path

from config import (
    COCO_ROOT, COCO_ANN, OUTPUT_DIR, ACTIVATIONS_DIR,
    MODEL_NAME, NUM_IMAGES, BATCH_SIZE, TAU_COACT,
    BASE_GRAPH_PATH,
    resolve_layers, resolve_device, ensure_dirs,
)
from coactivation_utils import (
    export_activations,
    export_all_ranks,
    export_all_correlations,
    build_coactivation_graph,
    save_coactivation_graph,
    load_coactivation_graph,
)


def register_hooks(model, layer_names):
    """Register forward hooks to capture activations for specified layers."""
    hooks = []
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    return hooks, activations


def extract_activations(model, dataset, layer_names, device, num_images):
    """
    Extract activations from ResNet-18 for specified layers.
    Activations are average-pooled for conv layers (following Vitor's method)
    and exported to CSV via coactivation_utils.export_activations.
    """
    print(f"\n  Extracting activations for {len(layer_names)} layers...")
    hooks, activations = register_hooks(model, layer_names)
    model.eval()

    first_batch = True
    with torch.no_grad():
        for idx in range(min(num_images, len(dataset))):
            if idx % 100 == 0:
                print(f"    Image {idx}/{num_images}")

            image, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)
            _ = model(image)

            for layer_name in layer_names:
                if layer_name in activations:
                    export_activations(
                        activations[layer_name],
                        init=first_batch,
                        file_name=str(ACTIVATIONS_DIR / f"{layer_name}.csv"),
                    )
            first_batch = False

    for h in hooks:
        h.remove()

    print(f"    Done: {min(num_images, len(dataset))} images processed.")


def create_layer_index(layer_names):
    """Create layer_index.csv required by Vitor's correlation pipeline."""
    df = pd.DataFrame({'module name': layer_names})
    index_path = str(OUTPUT_DIR / 'layer_index.csv')
    df.to_csv(index_path, index=False)
    return index_path


def build(args):
    """Full pipeline: activations → ranks → correlations → graph."""
    ensure_dirs()
    device = resolve_device()
    layer_names = resolve_layers()

    print("=" * 60)
    print("  Step 1: Build Base Coactivation Graph")
    print("=" * 60)
    print(f"  Layers: {layer_names}")
    print(f"  Device: {device}")
    print(f"  Images: {NUM_IMAGES}")
    print(f"  τ_coact: {TAU_COACT}")

    # 1. Load model
    print("\n1. Loading ResNet-18...")
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    ).to(device)
    model.eval()

    # 2. Load COCO
    print("2. Loading COCO dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = CocoDetection(root=COCO_ROOT, annFile=COCO_ANN, transform=transform)
    print(f"   Dataset: {len(dataset)} images")

    # 3. Extract activations
    print("3. Extracting activations...")
    extract_activations(model, dataset, layer_names, device, NUM_IMAGES)

    # 4. Create layer index
    index_path = create_layer_index(layer_names)

    # 5. Compute ranks (Spearman: correlate on ranks, not raw values)
    print("4. Computing ranks...")
    export_all_ranks(
        folder=str(OUTPUT_DIR) + '/',
        index_name='layer_index.csv',
        verbose=True,
        extended=True,
    )

    # 6. Compute Spearman correlations
    print("5. Computing Spearman correlations...")
    export_all_correlations(
        folder=str(OUTPUT_DIR) + '/',
        index_name='layer_index.csv',
        verbose=True,
    )

    # 7. Build graph
    print("6. Building coactivation graph...")
    graph = build_coactivation_graph(
        folder=str(OUTPUT_DIR) + '/',
        index_name='layer_index.csv',
        thresh=TAU_COACT,
        verbose=True,
    )

    # 8. Save
    print("7. Saving graph...")
    save_coactivation_graph(
        graph,
        folder=str(OUTPUT_DIR) + '/',
        file_name=BASE_GRAPH_PATH.name,
    )

    print(f"\n  Graph statistics:")
    print(f"    Nodes: {graph.number_of_nodes()}")
    print(f"    Edges: {graph.number_of_edges()}")
    print(f"    Saved: {BASE_GRAPH_PATH}")
    print("=" * 60)

    return graph


def load(args):
    """Load existing base graph."""
    print(f"Loading existing graph: {BASE_GRAPH_PATH}")
    graph = load_coactivation_graph(
        folder=str(OUTPUT_DIR) + '/',
        file_name=BASE_GRAPH_PATH.name,
    )
    print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    return graph


def main():
    parser = argparse.ArgumentParser(description='Step 1: Build base coactivation graph')
    parser.add_argument('--load', action='store_true',
                        help='Load existing graph instead of building from scratch')
    args = parser.parse_args()

    if args.load and BASE_GRAPH_PATH.exists():
        return load(args)
    else:
        return build(args)


if __name__ == '__main__':
    main()
# SCAG: Semantic Coactivation Graph

**Extending coactivation graphs with visual concept associations for interpretable neural network analysis.**

## Hypothesis

> The knowledge contained in a neural network can be captured more faithfully by a representation that integrates:
> 1. Statistical dependencies among neuron activations (coactivation graph)
> 2. Visual patches associated with neuron activations (spatial activation regions)
> 3. Semantic concepts that describe what those patches represent (COCO category labels)

## Graph Structure

```
neuron ──[coactivates_with (ρ)]──> neuron      (RQ1: neuron-neuron correlation)
   │
   └──[activates_on (IoU)]──> concept           (RQ2+RQ3: patch extraction + concept matching)
         │
         └── attributes: name, supercategory
```

Each neuron also stores **patch metadata** (top activating images + bounding boxes)
so you can trace back: "neuron 33 activates on *these regions* in *these images*,
and those regions overlap with concept *stop sign*".

## Project Structure

```
scag/
├── config.py                        # All paths, thresholds, hyperparameters
├── coactivation_utils.py            # Vitor's core utilities (untouched)
│
├── step1_base_graph.py              # RQ1: activations → ranks → Spearman ρ → graph
├── step2_concept_associations.py    # RQ2+RQ3: spatial IoU → neuron-concept scores
├── step3_extend_graph.py            # Combine base graph + concepts → SCAG
├── step4_analysis.py                # All analyses + figures (Part A + Part B)
├── step5_visualize.py               # Interactive D3.js HTML explorer
│
├── run_all.py                       # Run full pipeline
└── results/                         # All outputs (created automatically)
    ├── base_coactivation_graph.graphml
    ├── extended_coactivation_graph.graphml
    ├── concept_associations.pkl
    ├── graph_explorer.html
    └── figures/
        ├── A1_iou_distribution.pdf
        ├── A2_concept_table.tex
        ├── A3_supercategory.pdf
        ├── A4_polysemanticity.pdf
        ├── A5_freq_detect.pdf
        ├── B1_coact_concept_scatter.pdf
        ├── B2_ensemble_table.tex
        ├── B3_case_studies.tex
        └── B4_community_heatmap.pdf
```

## Usage

### Full pipeline (first time)
```bash
python run_all.py
```

### Run specific steps
```bash
python run_all.py --only 4 5       # Analysis + visualisation only
python run_all.py --skip 1 2       # Skip extraction (use existing data)
```

### Individual steps
```bash
python step1_base_graph.py              # Build from scratch
python step1_base_graph.py --load       # Load existing
python step2_concept_associations.py
python step3_extend_graph.py
python step4_analysis.py                # All analyses
python step4_analysis.py --part A       # Network Dissection-style only
python step4_analysis.py --part B       # Graph-specific only
python step5_visualize.py
```

### Configuration
Edit `config.py` to change:
- **LAYER_SCOPE**: `'layer4'`, `'layer3'`, `'all'`, or custom list
- **CONCEPT_LAYER**: which layer gets concept associations
- **TAU_COACT / TAU_ASSOC**: graph edge thresholds
- **TOP_K_PERCENT**: activation threshold for patch extraction (default 10%)
- **NUM_IMAGES**: number of COCO images to process

## Requirements

```
torch
torchvision
numpy
pandas
networkx
scipy
matplotlib
scikit-learn
pycocotools
```

## Data

Download COCO 2017 validation set:
```bash
mkdir -p data/coco
wget http://images.cocodataset.org/zips/val2017.zip -O data/coco/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/coco/annotations.zip
unzip data/coco/val2017.zip -d data/coco/
unzip data/coco/annotations.zip -d data/coco/
```

## Methodology

### Step 1: Base Coactivation Graph (Vitor's method)
- Extract activations from ResNet-18 conv/fc layers
- Rank-transform activations (Spearman)
- Compute pairwise Spearman correlation
- Threshold at τ_coact → directed graph

### Step 2: Concept Associations (spatial IoU)
- For each image: get full spatial activation maps (C × H × W)
- For each neuron: threshold top-k% → binary patch mask
- Upsample patch to image resolution
- Compute IoU with COCO segmentation masks per category
- Average IoU across all images containing each concept
- Store top-k activating image IDs as patch metadata

### Step 3: Graph Extension
- Add 78 COCO concept nodes with semantic attributes
- Add neuron→concept edges where IoU > τ_assoc
- Attach patch metadata to neuron nodes

### Step 4: Analysis
**Part A** (concept-level, inspired by Network Dissection):
- IoU distribution, specialisation rate
- Per-concept detectability ranking
- Supercategory analysis
- Monosemantic vs polysemantic classification
- Frequency–detectability correlation

**Part B** (graph-level, testing the hypothesis):
- Coactivation–concept alignment (central result)
- Semantic ensemble queries
- Case studies with verified neuron pairs
- Community detection with semantic profiling

### Step 5: Interactive Visualisation
- Single HTML file with embedded D3.js
- Dual sliders for ρ and IoU thresholds
- Layer block toggles (layer1-4, fc)
- Concept focus dropdown
- Click-to-highlight neighbourhood
- Patch metadata in info panel
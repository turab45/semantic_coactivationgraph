"""
SCAG Configuration
==================
All paths, thresholds, and hyperparameters in one place.
Change values here — every script reads from this file.
"""

from pathlib import Path

# =============================================================
# PATHS
# =============================================================
COCO_ROOT = '../data/coco/val2017'
COCO_ANN  = '../data/coco/annotations/instances_val2017.json'
OUTPUT_DIR = Path('./results')

# Sub-directories (created automatically)
ACTIVATIONS_DIR = OUTPUT_DIR / 'activations'
RANKS_DIR       = OUTPUT_DIR / 'ranks'
STATS_DIR       = OUTPUT_DIR / 'stats'
CORRELATIONS_DIR= OUTPUT_DIR / 'correlations'
FIGURES_DIR     = OUTPUT_DIR / 'figures'

# Graph files
BASE_GRAPH_PATH     = OUTPUT_DIR / 'base_coactivation_graph.graphml'
EXTENDED_GRAPH_PATH = OUTPUT_DIR / 'extended_coactivation_graph.graphml'
CONCEPT_PKL_PATH    = OUTPUT_DIR / 'concept_associations.pkl'
EXPLORER_HTML_PATH  = OUTPUT_DIR / 'graph_explorer.html'

# =============================================================
# MODEL
# =============================================================
MODEL_NAME = 'resnet18'
DEVICE = 'cuda'  # 'cuda', 'mps', or 'cpu' — auto-detected at runtime

# =============================================================
# LAYER SELECTION
# =============================================================
# Which layers to include in the coactivation graph.
# Options:
#   'layer4'  → only layer4 block (4 conv layers, 2048 neurons total)
#   'layer3'  → only layer3 block
#   'all'     → all 17 conv/fc layers (full network)
#   ['layer4.1.conv2', 'layer4.0.conv2']  → custom list
LAYER_SCOPE = 'layer4'

# Which single layer to compute concept associations on.
# Must be a conv layer (not fc) — needs spatial activation maps.
CONCEPT_LAYER = 'layer4.1.conv2'

# Full list of available layers in ResNet-18:
ALL_LAYERS = [
    'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2',
    'layer2.0.conv1', 'layer2.0.conv2', 'layer2.1.conv1', 'layer2.1.conv2',
    'layer3.0.conv1', 'layer3.0.conv2', 'layer3.1.conv1', 'layer3.1.conv2',
    'layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2',
    'fc',
]

# =============================================================
# HYPERPARAMETERS
# =============================================================
NUM_IMAGES      = 500       # Number of COCO images to process
BATCH_SIZE      = 1         # Forward pass batch size (1 for per-image patch extraction)
TOP_K_PERCENT   = 0.10      # Top 10% of activation values define the "patch"
TAU_ASSOC       = 0.1       # Minimum IoU to create neuron→concept edge
TAU_COACT       = 0.5       # Minimum Spearman ρ to create neuron↔neuron edge
TOP_K_IMAGES    = 5         # Number of top activating images to store per neuron (patch metadata)

# Analysis thresholds (for step4)
MONO_THRESHOLD_FACTOR = 1.5  # Monosemantic if top IoU > factor × second IoU
IOU_SPECIALISATION    = 0.2  # Neuron "specialised" if max IoU > this

# =============================================================
# COCO CATEGORY METADATA
# =============================================================
COCO_ID_TO_NAME = {
    1:'person',2:'bicycle',3:'car',4:'motorcycle',5:'airplane',6:'bus',
    7:'train',8:'truck',9:'boat',10:'traffic light',11:'fire hydrant',
    13:'stop sign',14:'parking meter',15:'bench',16:'bird',17:'cat',
    18:'dog',19:'horse',20:'sheep',21:'cow',22:'elephant',23:'bear',
    24:'zebra',25:'giraffe',27:'backpack',28:'umbrella',31:'handbag',
    32:'tie',33:'suitcase',34:'frisbee',35:'skis',36:'snowboard',
    37:'sports ball',38:'kite',39:'baseball bat',40:'baseball glove',
    41:'skateboard',42:'surfboard',43:'tennis racket',44:'bottle',
    46:'wine glass',47:'cup',48:'fork',49:'knife',50:'spoon',51:'bowl',
    52:'banana',53:'apple',54:'sandwich',55:'orange',56:'broccoli',
    57:'carrot',58:'hot dog',59:'pizza',60:'donut',61:'cake',62:'chair',
    63:'couch',64:'potted plant',65:'bed',67:'dining table',70:'toilet',
    72:'tv',73:'laptop',74:'mouse',75:'remote',76:'keyboard',
    77:'cell phone',78:'microwave',79:'oven',80:'toaster',81:'sink',
    82:'refrigerator',84:'book',85:'clock',86:'vase',87:'scissors',
    88:'teddy bear',89:'hair drier',90:'toothbrush',
}

SUPERCATEGORY_MAP = {
    'person':'person','bicycle':'vehicle','car':'vehicle',
    'motorcycle':'vehicle','airplane':'vehicle','bus':'vehicle',
    'train':'vehicle','truck':'vehicle','boat':'vehicle',
    'traffic light':'outdoor','fire hydrant':'outdoor',
    'stop sign':'outdoor','parking meter':'outdoor','bench':'outdoor',
    'bird':'animal','cat':'animal','dog':'animal','horse':'animal',
    'sheep':'animal','cow':'animal','elephant':'animal','bear':'animal',
    'zebra':'animal','giraffe':'animal','backpack':'accessory',
    'umbrella':'accessory','handbag':'accessory','tie':'accessory',
    'suitcase':'accessory','frisbee':'sports','skis':'sports',
    'snowboard':'sports','sports ball':'sports','kite':'sports',
    'baseball bat':'sports','baseball glove':'sports',
    'skateboard':'sports','surfboard':'sports','tennis racket':'sports',
    'bottle':'kitchen','wine glass':'kitchen','cup':'kitchen',
    'fork':'kitchen','knife':'kitchen','spoon':'kitchen','bowl':'kitchen',
    'banana':'food','apple':'food','sandwich':'food','orange':'food',
    'broccoli':'food','carrot':'food','hot dog':'food','pizza':'food',
    'donut':'food','cake':'food','chair':'furniture','couch':'furniture',
    'potted plant':'furniture','bed':'furniture','dining table':'furniture',
    'toilet':'furniture','tv':'electronic','laptop':'electronic',
    'mouse':'electronic','remote':'electronic','keyboard':'electronic',
    'cell phone':'electronic','microwave':'appliance','oven':'appliance',
    'toaster':'appliance','sink':'appliance','refrigerator':'appliance',
    'book':'indoor','clock':'indoor','vase':'indoor','scissors':'indoor',
    'teddy bear':'indoor','hair drier':'indoor','toothbrush':'indoor',
}


# =============================================================
# HELPERS
# =============================================================

def resolve_layers():
    """Return list of layer names based on LAYER_SCOPE setting."""
    if isinstance(LAYER_SCOPE, list):
        return LAYER_SCOPE
    if LAYER_SCOPE == 'all':
        return ALL_LAYERS
    # Match by block prefix: 'layer4' → all layer4.* layers
    return [l for l in ALL_LAYERS if l.startswith(LAYER_SCOPE)]


def resolve_device():
    """Auto-detect best available device."""
    import torch
    if DEVICE == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    if DEVICE == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def ensure_dirs():
    """Create all output directories."""
    for d in [OUTPUT_DIR, ACTIVATIONS_DIR, RANKS_DIR, STATS_DIR,
              CORRELATIONS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def concept_name(concept_id):
    """Resolve concept ID to human name."""
    if isinstance(concept_id, str):
        return concept_id
    return COCO_ID_TO_NAME.get(concept_id, str(concept_id))


def supercategory(concept_name_str):
    """Get COCO supercategory for a concept name."""
    return SUPERCATEGORY_MAP.get(concept_name_str, 'other')
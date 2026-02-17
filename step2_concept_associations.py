"""
Step 2: Neuron-Concept Associations via Spatial IoU
====================================================
RQ2: What visual patches do these neurons fire on?
RQ3: What semantic concept do those patches represent?

Pipeline:
  1. For each image, extract FULL spatial activation maps (not averaged)
  2. For each neuron: threshold top-k% activations → that IS the patch
  3. Compute IoU between patch and COCO segmentation masks → concept score
  4. Aggregate across images → neuron-concept association matrix
  5. Store top-k activating images + patch bounding boxes per neuron

The patch (step 2) and concept matching (step 3) are computed together
but stored separately:
  - Association scores: concept_associations.pkl (for graph edges)
  - Patch metadata: stored per neuron (for traceability/visualisation)

Reads: COCO images + annotations + ResNet-18
Writes: results/concept_associations.pkl

Usage:
    python step2_concept_associations.py
"""

import pickle
import json
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import CocoDetection
import numpy as np
from collections import defaultdict
from typing import Dict, List

from config import (
    COCO_ROOT, COCO_ANN, OUTPUT_DIR, CONCEPT_PKL_PATH, CONCEPT_LAYER,
    NUM_IMAGES, TOP_K_PERCENT, TAU_ASSOC, TOP_K_IMAGES, COCO_ID_TO_NAME,
    resolve_device, ensure_dirs,
)


# =============================================================
# Spatial Activation Hook
# =============================================================

class SpatialActivationHook:
    """Captures FULL spatial activation maps (no pooling)."""

    def __init__(self):
        self.activations = {}

    def get_hook(self, name):
        def hook(module, inp, out):
            self.activations[name] = out.detach()
        return hook

    def register(self, model, layer_names):
        hooks = []
        for name, module in model.named_modules():
            if name in layer_names:
                h = module.register_forward_hook(self.get_hook(name))
                hooks.append(h)
        return hooks


# =============================================================
# COCO Mask Extraction
# =============================================================

def get_concept_masks(coco_api, annotations):
    """
    Extract binary masks per COCO category for one image.
    Multiple instances of the same category are merged (OR).
    
    Returns: dict {category_id: bool ndarray (H, W)}
    """
    masks = {}
    for ann in annotations:
        cat_id = ann['category_id']
        mask = coco_api.annToMask(ann)
        if cat_id in masks:
            masks[cat_id] = np.logical_or(masks[cat_id], mask)
        else:
            masks[cat_id] = mask.astype(bool)
    return masks


# =============================================================
# Spatial IoU: Patch Extraction + Concept Matching
# =============================================================

def compute_spatial_iou(activation_map, concept_mask, top_k_percent):
    """
    For each channel in activation_map:
      1. Upsample to image size
      2. Threshold top-k% → binary patch mask
      3. Compute IoU with concept_mask

    Also returns the bounding box of the patch (for metadata).

    Args:
        activation_map: (C, H_act, W_act) tensor
        concept_mask: (H_img, W_img) bool ndarray
        top_k_percent: float, e.g. 0.10 for top 10%

    Returns:
        iou_scores: (C,) ndarray
        patch_bboxes: list of (x, y, w, h) per channel (or None if no activation)
    """
    H_img, W_img = concept_mask.shape
    C = activation_map.shape[0]

    # Upsample activation maps to image resolution
    upsampled = F.interpolate(
        activation_map.unsqueeze(0),
        size=(H_img, W_img),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)  # (C, H_img, W_img)

    iou_scores = np.zeros(C)

    for c in range(C):
        act = upsampled[c].cpu().numpy()

        # Top-k% threshold → binary patch mask
        threshold = np.percentile(act, (1 - top_k_percent) * 100)
        patch_mask = act >= threshold

        # IoU
        intersection = np.logical_and(patch_mask, concept_mask).sum()
        union = np.logical_or(patch_mask, concept_mask).sum()
        iou_scores[c] = intersection / union if union > 0 else 0.0

    return iou_scores


def compute_patch_bbox(activation_map, channel_idx, top_k_percent, img_h, img_w):
    """
    Compute bounding box of the top-k% activation region for one channel.
    Returns (x, y, w, h) in image coordinates, or None if empty.
    """
    upsampled = F.interpolate(
        activation_map[channel_idx].unsqueeze(0).unsqueeze(0),
        size=(img_h, img_w),
        mode='bilinear',
        align_corners=False,
    ).squeeze().cpu().numpy()

    threshold = np.percentile(upsampled, (1 - top_k_percent) * 100)
    mask = upsampled >= threshold

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min()), int(ys.max() - ys.min()))


# =============================================================
# Main Pipeline
# =============================================================

def build_associations():
    """
    Full pipeline: for each image, extract spatial activations,
    compute IoU with COCO masks, aggregate across dataset.
    """
    ensure_dirs()
    device = resolve_device()

    print("=" * 60)
    print("  Step 2: Neuron-Concept Associations")
    print("=" * 60)
    print(f"  Layer: {CONCEPT_LAYER}")
    print(f"  Images: {NUM_IMAGES}")
    print(f"  Top-k: {TOP_K_PERCENT * 100:.0f}%")
    print(f"  τ_assoc: {TAU_ASSOC}")

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
    coco_api = dataset.coco

    # 3. Register spatial hook
    hook = SpatialActivationHook()
    hooks = hook.register(model, [CONCEPT_LAYER])

    # 4. Process images
    print("3. Computing spatial IoU...")
    concept_scores = defaultdict(list)      # concept_id → list of (C,) arrays
    neuron_activations = defaultdict(list)   # neuron_idx → list of (mean_act, img_id)

    n_processed = 0
    n_skipped = 0

    with torch.no_grad():
        for idx in range(min(NUM_IMAGES, len(dataset))):
            if idx % 100 == 0 and idx > 0:
                print(f"    {idx}/{NUM_IMAGES} images...")

            image, target = dataset[idx]

            if len(target) == 0:
                n_skipped += 1
                continue

            image_tensor = image.unsqueeze(0).to(device)
            _ = model(image_tensor)

            activation_map = hook.activations[CONCEPT_LAYER][0]  # (C, H, W)
            C = activation_map.shape[0]

            # Get image ID
            img_id = target[0]['image_id']

            # Store mean activation per neuron (for top-k image ranking)
            mean_per_channel = activation_map.mean(dim=(1, 2)).cpu().numpy()  # (C,)
            for c_idx in range(C):
                neuron_activations[c_idx].append((float(mean_per_channel[c_idx]), img_id))

            # Get concept masks
            concept_masks = get_concept_masks(coco_api, target)

            # Compute IoU for each concept present in this image
            for cat_id, mask in concept_masks.items():
                iou_scores = compute_spatial_iou(activation_map, mask, TOP_K_PERCENT)
                concept_scores[cat_id].append(iou_scores)

            n_processed += 1

    for h in hooks:
        h.remove()

    print(f"    Processed: {n_processed}, Skipped (no annotations): {n_skipped}")

    # 5. Aggregate: mean IoU across images where concept appears
    print("4. Aggregating scores...")
    associations = {}
    for cat_id, score_list in concept_scores.items():
        associations[cat_id] = np.mean(score_list, axis=0)  # (C,)

    print(f"   Concepts found: {len(associations)}")

    # 6. Build patch metadata per neuron
    print("5. Building patch metadata...")
    C = len(next(iter(neuron_activations.values())))  # infer from first entry... 
    # Actually C is number of entries per neuron
    # Let me get it properly
    C = activation_map.shape[0]  # from last processed image

    patch_metadata = {}
    for n_idx in range(C):
        acts = neuron_activations[n_idx]
        # Sort by activation strength, take top-k
        acts_sorted = sorted(acts, key=lambda x: x[0], reverse=True)[:TOP_K_IMAGES]
        patch_metadata[n_idx] = {
            'top_images': [img_id for _, img_id in acts_sorted],
            'top_activations': [act_val for act_val, _ in acts_sorted],
        }

    # 7. Save
    print("6. Saving...")
    output = {
        'associations': associations,          # {concept_id: (C,) ndarray}
        'patch_metadata': patch_metadata,      # {neuron_idx: {top_images, top_activations}}
        'config': {
            'layer': CONCEPT_LAYER,
            'num_images': n_processed,
            'top_k_percent': TOP_K_PERCENT,
            'tau_assoc': TAU_ASSOC,
        },
    }

    with open(CONCEPT_PKL_PATH, 'wb') as f:
        pickle.dump(output, f)

    # Print summary
    n_concepts = len(associations)
    n_neurons = next(iter(associations.values())).shape[0]
    n_edges = sum(
        (scores > TAU_ASSOC).sum() for scores in associations.values()
    )

    print(f"\n  Summary:")
    print(f"    Neurons: {n_neurons}")
    print(f"    Concepts: {n_concepts}")
    print(f"    Edges (IoU > {TAU_ASSOC}): {n_edges}")
    print(f"    Saved: {CONCEPT_PKL_PATH}")
    print("=" * 60)

    return associations, patch_metadata


if __name__ == '__main__':
    build_associations()
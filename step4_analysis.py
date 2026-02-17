"""
Step 4: Analysis & Figures (CVPR Quality)
==========================================
Uses the exact CVPR styling established in earlier analyses.

Part A — Network Dissection-style concept analysis:
  A1. IoU distribution histogram
  A2. Per-concept best-neuron table (LaTeX)
  A3. Supercategory bar chart + concept coverage
  A4. Monosemantic vs polysemantic (scatter + ratio histogram)
  A5. Frequency–detectability correlation

Part B — Graph-specific (Coactivation + Concepts):
  B1. Coactivation–concept alignment (hexbin + Spearman ρ)
  B2. Semantic ensemble queries (LaTeX table)
  B3. Case studies (verified neuron pairs, LaTeX)
  B4. Community analysis (heatmap + within/between table)

Part C — Qualitative Results:
  C1. Top concept profiles (per-neuron bar charts)
  C2. Ensemble pair comparison (side-by-side profiles)
  C3. Degree distributions (coactivation + concept)
  C4. Correlation matrix of top neurons

Reads: extended_coactivation_graph.graphml + concept_associations.pkl
Writes: results/figures/*.pdf + *.png + *.tex

Usage:
    python step4_analysis.py               # Run all
    python step4_analysis.py --part A      # Only Part A
    python step4_analysis.py --part B      # Only Part B
    python step4_analysis.py --part C      # Only Part C (qualitative)
"""

import argparse
import pickle
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

from config import (
    OUTPUT_DIR, FIGURES_DIR, EXTENDED_GRAPH_PATH, CONCEPT_PKL_PATH,
    CONCEPT_LAYER, TAU_ASSOC, TAU_COACT, TOP_K_PERCENT,
    MONO_THRESHOLD_FACTOR, IOU_SPECIALISATION,
    COCO_ID_TO_NAME, SUPERCATEGORY_MAP,
    ensure_dirs, concept_name, supercategory,
)


# =============================================================
# CVPR Style (exact match to earlier figures)
# =============================================================

SUPERCATEGORY_ORDER = [
    'person', 'vehicle', 'outdoor', 'animal', 'accessory',
    'sports', 'kitchen', 'food', 'furniture', 'electronic',
    'appliance', 'indoor',
]

SUPERCATEGORY_COLORS = {
    'person': '#E24A33', 'vehicle': '#348ABD', 'outdoor': '#988ED5',
    'animal': '#8EBA42', 'accessory': '#FBC15E', 'sports': '#FFB5B8',
    'kitchen': '#777777', 'food': '#E5AE38', 'furniture': '#6D904F',
    'electronic': '#FC4F30', 'appliance': '#008FD5', 'indoor': '#A8786E',
    'other': '#999999',
}


def setup_cvpr_style():
    """Configure matplotlib for CVPR-quality figures."""
    mpl.rcParams.update({
        # Font settings — use serif (Times) to match CVPR LaTeX template
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,

        # Use LaTeX-style math rendering
        'mathtext.fontset': 'stix',

        # Clean axes
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Figure defaults
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def save_fig(fig, name):
    """Save figure as both PDF (vector for CVPR) and PNG (preview)."""
    pdf_path = str(FIGURES_DIR / f'{name}.pdf')
    png_path = str(FIGURES_DIR / f'{name}.png')
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  Saved: {name}.pdf + .png")


def resolve_concept_name(concept_id, concept_ids=None):
    """Try to resolve a concept id to a human-readable name."""
    if isinstance(concept_id, str):
        return concept_id
    return COCO_ID_TO_NAME.get(concept_id, str(concept_id))


# =============================================================
# Data Loading
# =============================================================

def load_data():
    """Load extended graph and concept associations."""
    print("Loading data...")

    with open(CONCEPT_PKL_PATH, 'rb') as f:
        concept_data = pickle.load(f)
    associations = concept_data['associations']
    patch_meta = concept_data.get('patch_metadata', {})

    concept_ids = sorted(associations.keys())
    concept_names = [resolve_concept_name(cid) for cid in concept_ids]
    N = next(iter(associations.values())).shape[0]

    # Association matrix: (N_neurons, N_concepts)
    assoc_matrix = np.zeros((N, len(concept_ids)))
    for j, cid in enumerate(concept_ids):
        assoc_matrix[:, j] = associations[cid]

    # Extended graph
    G = nx.read_graphml(str(EXTENDED_GRAPH_PATH))

    # Extract correlation matrix for neurons in CONCEPT_LAYER
    corr_matrix = np.zeros((N, N))
    for u, v, d in G.edges(data=True):
        if d.get('edge_type', 'coactivates_with') == 'coactivates_with':
            u_layer = G.nodes[u].get('layer', '')
            v_layer = G.nodes[v].get('layer', '')
            if u_layer == CONCEPT_LAYER and v_layer == CONCEPT_LAYER:
                u_idx = int(G.nodes[u].get('inlayer_id', -1))
                v_idx = int(G.nodes[v].get('inlayer_id', -1))
                if 0 <= u_idx < N and 0 <= v_idx < N:
                    w = float(d.get('weight', 0))
                    corr_matrix[u_idx, v_idx] = w
                    corr_matrix[v_idx, u_idx] = w
    np.fill_diagonal(corr_matrix, 1.0)

    print(f"  Neurons: {N}, Concepts: {len(concept_ids)}")
    print(f"  Concept IDs sample: {concept_ids[:5]}")
    print(f"  Score matrix shape: {assoc_matrix.shape}")

    return {
        'assoc_matrix': assoc_matrix,
        'corr_matrix': corr_matrix,
        'concept_ids': concept_ids,
        'concept_names': concept_names,
        'associations': associations,
        'patch_meta': patch_meta,
        'graph': G,
        'N': N,
    }


# =============================================================
# PART A: Network Dissection-style Analysis
# =============================================================

def A1_iou_distribution(data):
    """Histogram of max IoU per neuron — exact CVPR style."""
    print("\n" + "=" * 60)
    print("  A1: IoU Distribution Histogram")
    print("=" * 60)
    setup_cvpr_style()

    M = data['assoc_matrix']
    max_ious = M.max(axis=1)
    mean_val = np.mean(max_ious)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    bins = np.linspace(
        np.floor(max_ious.min() * 50) / 50,
        np.ceil(max_ious.max() * 50) / 50,
        31,
    )

    n, bin_edges, patches = ax.hist(
        max_ious, bins=bins,
        color='#4878CF',       # muted academic blue
        edgecolor='white',
        linewidth=0.4,
        alpha=0.9,
        zorder=3,
    )

    # Specialisation threshold
    ax.axvline(
        x=IOU_SPECIALISATION, color='#2CA02C', linestyle='-.', linewidth=1.0,
        zorder=4, label=r'threshold ($\mathrm{IoU} > ' + f'{IOU_SPECIALISATION}' + r'$)',
    )

    # Mean line
    ax.axvline(
        x=mean_val, color='#333333', linestyle=':', linewidth=1.0,
        zorder=4, label=f'Mean = {mean_val:.3f}',
    )

    # Annotate specialisation percentage
    pct_above = 100 * np.mean(max_ious > IOU_SPECIALISATION)
    ax.annotate(
        f'{pct_above:.1f}% of neurons\n'
        r'with $\mathrm{IoU}_{\max} > ' + f'{IOU_SPECIALISATION}' + r'$',
        xy=(IOU_SPECIALISATION, n.max() * 0.55),
        xytext=(IOU_SPECIALISATION + 0.10, n.max() * 0.55),
        fontsize=9, ha='left', va='center',
        arrowprops=dict(
            arrowstyle='->', color='#2CA02C',
            connectionstyle='arc3,rad=0.15', lw=1.0,
        ),
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.5),
    )

    ax.set_xlabel(r'Max IoU score per neuron ($\max_c\; s(n,c)$)')
    ax.set_ylabel('Number of neurons')
    ax.legend(loc='best', frameon=True, framealpha=0.9,
              edgecolor='#cccccc', fancybox=False)
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, 'A1_iou_distribution')

    # --- Summary statistics for the paper ---
    print(f"\n  --- Summary statistics ---")
    print(f"  N neurons:        {len(max_ious)}")
    print(f"  Mean max IoU:     {mean_val:.4f}")
    print(f"  Std max IoU:      {np.std(max_ious):.4f}")
    print(f"  Median max IoU:   {np.median(max_ious):.4f}")
    print(f"  Min max IoU:      {np.min(max_ious):.4f}")
    print(f"  Max max IoU:      {np.max(max_ious):.4f}")
    for t in [0.04, 0.10, 0.15, 0.20, 0.25, 0.30]:
        print(f"  % > {t}:           {100*np.mean(max_ious > t):.1f}%")


def A2_per_concept_table(data):
    """Per-concept best-neuron table — LaTeX export (exact CVPR format)."""
    print("\n" + "=" * 60)
    print("  A2: Per-Concept Best Neuron Table")
    print("=" * 60)

    M = data['assoc_matrix']
    names = data['concept_names']
    cids = data['concept_ids']

    results = []
    for j, (cid, cname) in enumerate(zip(cids, names)):
        scores = M[:, j]
        best_neuron = int(scores.argmax())
        best_iou = scores[best_neuron]
        supercat = SUPERCATEGORY_MAP.get(cname, 'unknown')
        results.append({
            'name': cname, 'id': cid, 'best_neuron': best_neuron,
            'best_iou': best_iou, 'supercat': supercat,
        })

    results.sort(key=lambda x: x['best_iou'], reverse=True)

    # Print full table
    print(f"\n  {'Rank':<5} {'Concept':<20} {'Supercategory':<14} {'Best Neuron':<12} {'IoU':<8}")
    print("  " + "-" * 60)
    for i, r in enumerate(results):
        print(f"  {i+1:<5} {r['name']:<20} {r['supercat']:<14} {r['best_neuron']:<12} {r['best_iou']:.4f}")

    # LaTeX table: top 20 + bottom 10
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Per-concept best neuron association scores. "
                       r"Top 20 and bottom 10 concepts ranked by maximum IoU "
                       r"across all " + str(data['N']) + r" neurons.}")
    latex_lines.append(r"\label{tab:per_concept}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{@{}rlcr@{}}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Rank} & \textbf{Concept} & \textbf{Best Neuron} & \textbf{IoU} \\")
    latex_lines.append(r"\midrule")
    for i, r in enumerate(results[:20]):
        latex_lines.append(f"{i+1} & {r['name']} & {r['best_neuron']} & {r['best_iou']:.3f} \\\\")
    latex_lines.append(r"\midrule")
    for i, r in enumerate(results[-10:]):
        rank = len(results) - 10 + i + 1
        latex_lines.append(f"{rank} & {r['name']} & {r['best_neuron']} & {r['best_iou']:.3f} \\\\")
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    tex_path = FIGURES_DIR / 'A2_per_concept_table.tex'
    with open(str(tex_path), 'w') as f:
        f.write("\n".join(latex_lines))
    print(f"\n  Saved: A2_per_concept_table.tex")

    # Summary
    ious = [r['best_iou'] for r in results]
    print(f"\n  --- Per-Concept Summary ---")
    print(f"  Best detected concept:  {results[0]['name']} (IoU = {results[0]['best_iou']:.4f})")
    print(f"  Worst detected concept: {results[-1]['name']} (IoU = {results[-1]['best_iou']:.4f})")
    print(f"  Mean best IoU:          {np.mean(ious):.4f}")
    print(f"  Concepts with IoU > {TAU_ASSOC}: {sum(1 for x in ious if x > TAU_ASSOC)}/{len(ious)}")

    return results


def A3_supercategory_analysis(data):
    """Supercategory bar chart + coverage — exact CVPR style."""
    print("\n" + "=" * 60)
    print("  A3: Supercategory Analysis")
    print("=" * 60)
    setup_cvpr_style()

    M = data['assoc_matrix']
    names = data['concept_names']
    n_neurons = data['N']

    # Map each concept column to supercategory
    col_to_supercat = {}
    for j, cname in enumerate(names):
        col_to_supercat[j] = SUPERCATEGORY_MAP.get(cname, 'unknown')

    # Per-supercategory stats
    supercat_neuron_counts = {}
    supercat_unique_concepts = {}
    supercat_total_concepts = {}

    for sc in SUPERCATEGORY_ORDER:
        cols = [j for j, s in col_to_supercat.items() if s == sc]
        supercat_total_concepts[sc] = len(cols)
        if not cols:
            supercat_neuron_counts[sc] = 0
            supercat_unique_concepts[sc] = 0
            continue
        sub = M[:, cols]
        supercat_neuron_counts[sc] = int(np.any(sub > TAU_ASSOC, axis=1).sum())
        supercat_unique_concepts[sc] = int(np.any(sub > TAU_ASSOC, axis=0).sum())

    cats = SUPERCATEGORY_ORDER
    counts = [supercat_neuron_counts[c] for c in cats]
    colors = [SUPERCATEGORY_COLORS[c] for c in cats]

    # --- Plot 1: Neurons per supercategory ---
    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    bars = ax.bar(range(len(cats)), counts, color=colors,
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.capitalize() for c in cats], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(f'Neurons detecting category\n($s(n,c) > {TAU_ASSOC}$)')
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax.set_axisbelow(True)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    str(count), ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    save_fig(fig, 'A3_supercategory_neurons')

    # --- Plot 2: Coverage (total vs detected concepts) ---
    fig2, ax2 = plt.subplots(figsize=(6.5, 3.2))

    total = [supercat_total_concepts[c] for c in cats]
    detected = [supercat_unique_concepts[c] for c in cats]
    x = np.arange(len(cats))
    width = 0.35

    ax2.bar(x - width/2, total, width, color='#cccccc',
            edgecolor='white', linewidth=0.5, label='Total concepts')
    ax2.bar(x + width/2, detected, width, color=colors,
            edgecolor='white', linewidth=0.5, label=f'Detected (IoU > {TAU_ASSOC})')

    ax2.set_xticks(x)
    ax2.set_xticklabels([c.capitalize() for c in cats], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Number of concepts')
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9,
               edgecolor='#cccccc', fancybox=False)
    ax2.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax2.set_axisbelow(True)

    fig2.tight_layout()
    save_fig(fig2, 'A3_supercategory_coverage')

    # Print summary
    print(f"\n  {'Supercategory':<14} {'Total Concepts':<16} {'Detected':<10} {'Neurons':<10}")
    print("  " + "-" * 50)
    for sc, tc, uc, nc in zip(cats, total, detected, counts):
        print(f"  {sc:<14} {tc:<16} {uc:<10} {nc:<10}")


def A4_mono_poly_semantic(data):
    """Monosemantic vs polysemantic — scatter + ratio histogram (exact CVPR style)."""
    print("\n" + "=" * 60)
    print("  A4: Monosemantic vs Polysemantic")
    print("=" * 60)
    setup_cvpr_style()

    M = data['assoc_matrix']
    N = data['N']

    top1_ious = []
    top2_ious = []
    ratios = []
    n_concepts_per = (M > TAU_ASSOC).sum(axis=1)

    for n in range(N):
        sorted_scores = np.sort(M[n, :])[::-1]
        top1_ious.append(sorted_scores[0])
        top2_ious.append(sorted_scores[1])
        if sorted_scores[1] > 0:
            ratios.append(sorted_scores[0] / sorted_scores[1])
        else:
            ratios.append(float('inf'))

    top1_ious = np.array(top1_ious)
    top2_ious = np.array(top2_ious)
    ratios = np.array(ratios)

    mono_threshold = MONO_THRESHOLD_FACTOR
    mono_mask = ratios >= mono_threshold
    poly_mask = ~mono_mask
    n_mono = mono_mask.sum()
    n_poly = poly_mask.sum()
    pct_mono = 100 * n_mono / N
    pct_poly = 100 * n_poly / N

    print(f"\n  Monosemantic (top1/top2 >= {mono_threshold}): {n_mono} ({pct_mono:.1f}%)")
    print(f"  Polysemantic (top1/top2 < {mono_threshold}):  {n_poly} ({pct_poly:.1f}%)")
    print(f"  Median top1/top2 ratio: {np.median(ratios[np.isfinite(ratios)]):.2f}")
    print(f"  Mean concepts/neuron: {n_concepts_per.mean():.1f}")

    # --- Plot 1: Top-1 vs Top-2 scatter ---
    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    ax.scatter(top1_ious[mono_mask], top2_ious[mono_mask],
               s=12, alpha=0.5, color='#348ABD',
               label=f'Monosemantic ({pct_mono:.0f}%)',
               edgecolors='none', zorder=3)
    ax.scatter(top1_ious[poly_mask], top2_ious[poly_mask],
               s=12, alpha=0.5, color='#E24A33',
               label=f'Polysemantic ({pct_poly:.0f}%)',
               edgecolors='none', zorder=3)

    lim_max = max(top1_ious.max(), top2_ious.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], '--', color='#999999', linewidth=0.8, zorder=2)
    x_line = np.linspace(0, lim_max, 100)
    ax.plot(x_line, x_line / mono_threshold, ':', color='#2CA02C', linewidth=1.0,
            zorder=2, label=f'top1/top2 = {mono_threshold}')

    ax.set_xlabel(r'Top-1 concept IoU ($\max_c\; s(n,c)$)')
    ax.set_ylabel('Top-2 concept IoU')
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', frameon=True, framealpha=0.9,
              edgecolor='#cccccc', fancybox=False, fontsize=9)
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax.xaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, 'A4_mono_poly_scatter')

    # --- Plot 2: Ratio histogram ---
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.0))

    finite_ratios = ratios[np.isfinite(ratios)]
    ax2.hist(finite_ratios, bins=30, color='#4878CF', edgecolor='white',
             linewidth=0.4, alpha=0.9, zorder=3)
    ax2.axvline(x=mono_threshold, color='#2CA02C', linestyle='--', linewidth=1.2,
                label=f'Threshold = {mono_threshold}', zorder=4)

    ax2.set_xlabel('Top-1 / Top-2 IoU ratio')
    ax2.set_ylabel('Number of neurons')
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9,
               edgecolor='#cccccc', fancybox=False)
    ax2.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax2.set_axisbelow(True)

    fig2.tight_layout()
    save_fig(fig2, 'A4_mono_poly_ratio')

    # --- Plot 3: Concepts per neuron distribution ---
    fig3, ax3 = plt.subplots(figsize=(4.5, 3.0))
    ax3.hist(n_concepts_per, bins=range(0, int(n_concepts_per.max()) + 2),
             color='#988ED5', edgecolor='white', linewidth=0.4, alpha=0.9, zorder=3)
    ax3.set_xlabel(f'Concepts per neuron ($s(n,c) > {TAU_ASSOC}$)')
    ax3.set_ylabel('Number of neurons')
    ax3.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax3.set_axisbelow(True)
    fig3.tight_layout()
    save_fig(fig3, 'A4_concepts_per_neuron')


def A5_frequency_detectability(data):
    """Frequency vs detectability scatter — exact CVPR style with supercategory colours."""
    print("\n" + "=" * 60)
    print("  A5: Frequency–Detectability Correlation")
    print("=" * 60)
    setup_cvpr_style()

    M = data['assoc_matrix']
    names = data['concept_names']

    best_ious = np.max(M, axis=0)
    n_detecting = np.sum(M > TAU_ASSOC, axis=0)

    # Supercategory colours
    colors_list = [SUPERCATEGORY_COLORS.get(SUPERCATEGORY_MAP.get(n, 'other'), '#999999')
                   for n in names]

    rho, pval = stats.spearmanr(n_detecting, best_ious)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    ax.scatter(n_detecting, best_ious, s=25, alpha=0.7, c=colors_list,
               edgecolors='white', linewidth=0.3, zorder=3)

    # Label top 5 by IoU
    for idx in np.argsort(best_ious)[-5:]:
        ax.annotate(names[idx], xy=(n_detecting[idx], best_ious[idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, color='#333333')

    # Label bottom 5 by IoU
    for idx in np.argsort(best_ious)[:5]:
        ax.annotate(names[idx], xy=(n_detecting[idx], best_ious[idx]),
                    xytext=(5, -8), textcoords='offset points',
                    fontsize=7, color='#999999')

    ax.set_xlabel(f'Number of neurons detecting concept ($s(n,c) > {TAU_ASSOC}$)')
    ax.set_ylabel(r'Best neuron IoU ($\max_n\; s(n,c)$)')
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax.xaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax.set_axisbelow(True)

    # Supercategory legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=SUPERCATEGORY_COLORS[sc],
               markersize=6, label=sc.capitalize())
        for sc in SUPERCATEGORY_ORDER
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, framealpha=0.9,
              edgecolor='#cccccc', fancybox=False, fontsize=7, ncol=2)

    # Correlation annotation
    ax.annotate(
        f'Spearman $\\rho$ = {rho:.3f}\n$p$ = {pval:.1e}',
        xy=(0.97, 0.05), xycoords='axes fraction', ha='right', va='bottom',
        fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', lw=0.5),
    )

    fig.tight_layout()
    save_fig(fig, 'A5_freq_detectability')

    print(f"\n  Spearman ρ (n_detecting vs best_IoU): {rho:.3f}, p = {pval:.2e}")
    top3_freq = np.argsort(n_detecting)[-3:][::-1]
    print(f"  Concepts detected by most neurons: ", end="")
    for idx in top3_freq:
        print(f"{names[idx]} ({n_detecting[idx]}), ", end="")
    print()


# =============================================================
# PART B: Graph-Specific Analysis
# =============================================================

def B1_coactivation_concept_scatter(data):
    """Hexbin scatter: ρ vs shared concepts — the central result."""
    print("\n" + "=" * 60)
    print("  B1: Coactivation–Concept Alignment")
    print("=" * 60)
    setup_cvpr_style()

    M = data['assoc_matrix']
    R = data['corr_matrix']
    N = data['N']
    binary = (M > TAU_ASSOC).astype(int)

    corr_vals, shared_vals = [], []
    for i in range(N):
        for j in range(i + 1, N):
            rho = R[i, j]
            if rho > 0:
                shared = int(np.logical_and(binary[i], binary[j]).sum())
                corr_vals.append(rho)
                shared_vals.append(shared)

    corr_vals = np.array(corr_vals)
    shared_vals = np.array(shared_vals)

    spearman_rho, pval = stats.spearmanr(corr_vals, shared_vals)

    # Top vs bottom 10%
    top_mask = corr_vals >= np.percentile(corr_vals, 90)
    bot_mask = corr_vals <= np.percentile(corr_vals, 10)
    top_mean = shared_vals[top_mask].mean()
    bot_mean = shared_vals[bot_mask].mean()
    ratio = top_mean / bot_mean if bot_mean > 0 else float('inf')
    _, tpval = stats.ttest_ind(shared_vals[top_mask], shared_vals[bot_mask])

    print(f"\n  Pairs analysed: {len(corr_vals):,}")
    print(f"  Spearman ρ: {spearman_rho:.3f}, p = {pval:.2e}")
    print(f"  Top 10% mean shared: {top_mean:.2f}")
    print(f"  Bottom 10% mean shared: {bot_mean:.2f}")
    print(f"  Ratio: {ratio:.2f}x (t-test p = {tpval:.2e})")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    hb = ax.hexbin(corr_vals, shared_vals, gridsize=40, cmap='Blues',
                   mincnt=1, linewidths=0.2, zorder=3)
    fig.colorbar(hb, ax=ax, shrink=0.7, label='Pair count')

    # Binned means
    bins = np.linspace(corr_vals.min(), corr_vals.max(), 20)
    bin_idx = np.digitize(corr_vals, bins)
    bin_centers, bin_means = [], []
    for b in range(1, len(bins)):
        mask = bin_idx == b
        if mask.sum() > 10:
            bin_centers.append((bins[b - 1] + bins[b]) / 2)
            bin_means.append(shared_vals[mask].mean())
    ax.plot(bin_centers, bin_means, 'o-', color='#E24A33', markersize=3,
            linewidth=1.2, label='Binned mean', zorder=4)

    ax.set_xlabel(r'Coactivation strength ($\rho$)')
    ax.set_ylabel('Shared concepts')
    ax.legend(fontsize=8, loc='upper left', frameon=True, framealpha=0.9,
              edgecolor='#cccccc', fancybox=False)
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax.set_axisbelow(True)

    ax.annotate(
        f'Spearman $\\rho_s$ = {spearman_rho:.3f}\n'
        f'Top/Bottom 10%: {ratio:.2f}×',
        xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
        fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', lw=0.5),
    )

    fig.tight_layout()
    save_fig(fig, 'B1_coact_concept_scatter')


def B2_semantic_ensembles(data):
    """Semantic ensemble queries — LaTeX table."""
    print("\n" + "=" * 60)
    print("  B2: Semantic Ensembles")
    print("=" * 60)

    M = data['assoc_matrix']
    R = data['corr_matrix']
    N = data['N']
    names = data['concept_names']
    binary = (M > TAU_ASSOC).astype(int)

    ensemble_counts = defaultdict(int)
    total = 0

    for i in range(N):
        for j in range(i + 1, N):
            if R[i, j] > TAU_COACT:
                shared = np.logical_and(binary[i], binary[j])
                for k in range(len(names)):
                    if shared[k]:
                        ensemble_counts[names[k]] += 1
                        total += 1

    n_with = sum(1 for v in ensemble_counts.values() if v > 0)
    print(f"\n  Total ensemble (pair, concept) tuples: {total:,}")
    print(f"  Concepts with ≥1 ensemble: {n_with}/{len(names)}")

    top15 = sorted(ensemble_counts.items(), key=lambda x: -x[1])[:15]
    print(f"\n  Top 15:")
    for cname, count in top15:
        print(f"    {cname:20s}: {count}")

    # LaTeX
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Top 15 concepts by semantic ensemble count. "
        r"An ensemble is a neuron pair ($\rho > " + f"{TAU_COACT}" + r"$) "
        r"where both neurons detect the same concept (IoU $> " + f"{TAU_ASSOC}" + r"$).}",
        r"\label{tab:ensembles}", r"\small",
        r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
        r"\textbf{Concept} & \textbf{Ensembles} & \textbf{Supercategory} \\",
        r"\midrule",
    ]
    for cname, count in top15:
        sc = SUPERCATEGORY_MAP.get(cname, 'other')
        lines.append(f"{cname} & {count} & {sc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(str(FIGURES_DIR / 'B2_ensemble_table.tex'), 'w') as f:
        f.write("\n".join(lines))
    print(f"  Saved: B2_ensemble_table.tex")


def B3_case_studies(data):
    """Detailed case studies — verified neuron pairs, LaTeX export."""
    print("\n" + "=" * 60)
    print("  B3: Case Studies")
    print("=" * 60)

    M = data['assoc_matrix']
    R = data['corr_matrix']
    N = data['N']
    names = data['concept_names']
    binary = (M > TAU_ASSOC).astype(int)

    # --- Strongest coactivating pair ---
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    best = np.unravel_index((R * mask).argmax(), R.shape)
    i, j = best
    shared = [names[k] for k in range(len(names)) if binary[i, k] and binary[j, k]]
    print(f"\n  Strongest pair: ({i}, {j}), ρ = {R[i,j]:.3f}")
    print(f"    Shared concepts ({len(shared)}): {', '.join(shared[:8])}")

    # --- Most shared concepts ---
    max_sh, best_pair = 0, (0, 0)
    for ii in range(N):
        for jj in range(ii + 1, N):
            if R[ii, jj] > TAU_COACT:
                ns = int(np.logical_and(binary[ii], binary[jj]).sum())
                if ns > max_sh:
                    max_sh = ns
                    best_pair = (ii, jj)
    i2, j2 = best_pair
    shared2 = [names[k] for k in range(len(names)) if binary[i2, k] and binary[j2, k]]
    print(f"\n  Most shared: ({i2}, {j2}), ρ = {R[i2,j2]:.3f}")
    print(f"    Shared concepts ({len(shared2)}): {', '.join(shared2[:10])}")

    # --- Best ensemble per concept ---
    best_per = {}
    for k, cname in enumerate(names):
        detectors = np.where(binary[:, k])[0]
        if len(detectors) < 2:
            continue
        best_rho, bp = 0, (0, 0)
        for di in range(len(detectors)):
            for dj in range(di + 1, len(detectors)):
                rho = R[detectors[di], detectors[dj]]
                if rho > best_rho:
                    best_rho = rho
                    bp = (detectors[di], detectors[dj])
        if best_rho > 0:
            best_per[cname] = {'pair': bp, 'rho': best_rho,
                               'iou_a': M[bp[0], k], 'iou_b': M[bp[1], k]}

    # Print and LaTeX
    print(f"\n  Best ensemble per concept (top 15):")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Best semantic ensemble per concept: strongest coactivating "
        r"pair among neurons that both detect the concept.}",
        r"\label{tab:case_studies}", r"\small",
        r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"\textbf{Concept} & \textbf{$n_i$} & \textbf{$n_j$} & "
        r"\textbf{$\rho$} & \textbf{IoU$_i$} & \textbf{IoU$_j$} \\",
        r"\midrule",
    ]

    for cname, info in sorted(best_per.items(), key=lambda x: -x[1]['rho'])[:15]:
        p = info['pair']
        print(f"    {cname:16s}: ({p[0]:3d}, {p[1]:3d}) "
              f"ρ={info['rho']:.3f}  IoU=({info['iou_a']:.3f}, {info['iou_b']:.3f})")
        lines.append(f"{cname} & {p[0]} & {p[1]} & {info['rho']:.3f} & "
                     f"{info['iou_a']:.3f} & {info['iou_b']:.3f} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(str(FIGURES_DIR / 'B3_case_studies.tex'), 'w') as f:
        f.write("\n".join(lines))
    print(f"  Saved: B3_case_studies.tex")

    return best_per


def B4_community_analysis(data):
    """Hierarchical clustering + community-concept heatmap."""
    print("\n" + "=" * 60)
    print("  B4: Community Analysis")
    print("=" * 60)
    setup_cvpr_style()

    R = data['corr_matrix']
    M = data['assoc_matrix']
    N = data['N']
    names = data['concept_names']
    binary = (M > TAU_ASSOC).astype(int)

    dist = 1 - np.abs(R)
    np.fill_diagonal(dist, 0)
    condensed = dist[np.triu_indices(N, k=1)]
    Z = linkage(condensed, method='ward')

    K_values = [5, 8, 10, 15, 20]
    results_rows = []

    for K in K_values:
        lab = fcluster(Z, t=K, criterion='maxclust')
        w, b = [], []
        for i in range(N):
            for j in range(i + 1, N):
                if R[i, j] > TAU_COACT:
                    ns = int(np.logical_and(binary[i], binary[j]).sum())
                    if lab[i] == lab[j]:
                        w.append(ns)
                    else:
                        b.append(ns)
        wm = np.mean(w) if w else 0
        bm = np.mean(b) if b else 0
        r = wm / bm if bm > 0 else 0
        _, p = stats.ttest_ind(w, b) if w and b else (0, 1)
        results_rows.append({'K': K, 'Within': round(wm, 2), 'Between': round(bm, 2),
                             'Ratio': round(r, 3), 'p': f'{p:.2e}'})
        print(f"  K={K:2d}: within={wm:.2f}, between={bm:.2f}, ratio={r:.3f}x, p={p:.2e}")

    # LaTeX
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Within-community vs between-community shared concepts "
        r"at different numbers of clusters $K$.}",
        r"\label{tab:communities}", r"\small",
        r"\begin{tabular}{@{}rrrrr@{}}", r"\toprule",
        r"\textbf{K} & \textbf{Within} & \textbf{Between} & \textbf{Ratio} & \textbf{p} \\",
        r"\midrule",
    ]
    for row in results_rows:
        lines.append(f"{row['K']} & {row['Within']} & {row['Between']} & "
                     f"{row['Ratio']} & {row['p']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(str(FIGURES_DIR / 'B4_community_table.tex'), 'w') as f:
        f.write("\n".join(lines))
    print(f"  Saved: B4_community_table.tex")

    # --- Heatmap for K=10 ---
    K = 10
    labels = fcluster(Z, t=K, criterion='maxclust')

    profile = np.zeros((K, len(names)))
    for c in range(1, K + 1):
        members = np.where(labels == c)[0]
        if len(members) > 0:
            profile[c - 1] = M[members].mean(axis=0)

    # Top 20 most discriminative concepts
    variance = profile.var(axis=0)
    top_idx = np.argsort(variance)[::-1][:20]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(profile[:, top_idx].T, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')

    ax.set_xticks(range(K))
    ax.set_xticklabels([f'C{c+1}\n(n={np.sum(labels==c+1)})' for c in range(K)], fontsize=8)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([names[c] for c in top_idx], fontsize=8)
    ax.set_xlabel('Community')

    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label('Mean IoU', fontsize=10)

    fig.tight_layout()
    save_fig(fig, 'B4_community_heatmap')


# =============================================================
# PART C: Qualitative Results
# =============================================================

def C1_neuron_concept_profiles(data):
    """
    Top 12 concepts: for each, show the best neuron's full concept profile
    as a horizontal bar chart. 3×4 grid figure.
    """
    print("\n" + "=" * 60)
    print("  C1: Neuron-Concept Profiles (Top 12)")
    print("=" * 60)
    setup_cvpr_style()

    M = data['assoc_matrix']
    names = data['concept_names']
    patch_meta = data['patch_meta']

    # Top 12 concepts by best IoU
    best_per_concept = []
    for j, cname in enumerate(names):
        best_n = int(M[:, j].argmax())
        best_per_concept.append((cname, j, best_n, M[best_n, j]))
    best_per_concept.sort(key=lambda x: -x[3])
    top12 = best_per_concept[:12]

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (cname, cj, neuron, iou) in enumerate(top12):
        ax = axes[idx]
        sc = SUPERCATEGORY_MAP.get(cname, 'other')
        color = SUPERCATEGORY_COLORS.get(sc, '#999')

        # This neuron's top 8 concepts
        neuron_scores = M[neuron, :]
        top_concepts_idx = np.argsort(neuron_scores)[::-1][:8]
        top_names = [names[k] for k in top_concepts_idx]
        top_scores = neuron_scores[top_concepts_idx]
        bar_colors = [SUPERCATEGORY_COLORS.get(SUPERCATEGORY_MAP.get(n, 'other'), '#999')
                      for n in top_names]

        ax.barh(range(len(top_names)), top_scores,
                color=bar_colors, edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=7)
        ax.set_xlim(0, max(0.4, top_scores[0] * 1.3))
        ax.invert_yaxis()
        ax.set_title(f'n{neuron} → {cname}\nIoU = {iou:.3f}', fontsize=9,
                     fontweight='bold', color=color)
        ax.tick_params(axis='x', labelsize=7)
        ax.xaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
        ax.set_axisbelow(True)

        # Patch info
        meta = patch_meta.get(neuron, {})
        imgs = meta.get('top_images', [])
        if imgs:
            ax.annotate(f'Top imgs: {", ".join(str(i) for i in imgs[:3])}',
                        xy=(0.95, 0.02), xycoords='axes fraction',
                        fontsize=5, ha='right', va='bottom', color='#888',
                        family='monospace')

    fig.suptitle(f'Top 12 Concepts — Best Detecting Neuron Profile ({CONCEPT_LAYER})',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, 'C1_neuron_concept_profiles')


def C2_ensemble_pair_profiles(data, best_per_concept=None):
    """
    Side-by-side concept profiles for top ensemble pairs.
    Shows what each neuron in the pair detects, highlighting overlap.
    """
    print("\n" + "=" * 60)
    print("  C2: Ensemble Pair Profiles")
    print("=" * 60)
    setup_cvpr_style()

    M = data['assoc_matrix']
    R = data['corr_matrix']
    names = data['concept_names']
    binary = (M > TAU_ASSOC).astype(int)

    # Find best pairs if not provided
    if best_per_concept is None:
        best_per_concept = {}
        for k, cname in enumerate(names):
            detectors = np.where(binary[:, k])[0]
            if len(detectors) < 2:
                continue
            best_rho, bp = 0, (0, 0)
            for di in range(len(detectors)):
                for dj in range(di + 1, len(detectors)):
                    rho = R[detectors[di], detectors[dj]]
                    if rho > best_rho:
                        best_rho = rho
                        bp = (detectors[di], detectors[dj])
            if best_rho > 0:
                best_per_concept[cname] = {'pair': bp, 'rho': best_rho,
                                           'iou_a': M[bp[0], k], 'iou_b': M[bp[1], k]}

    # Pick top 6 ensembles by ρ
    top6 = sorted(best_per_concept.items(), key=lambda x: -x[1]['rho'])[:6]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()

    for idx, (cname, info) in enumerate(top6):
        ax = axes[idx]
        ni, nj = info['pair']
        rho = info['rho']

        # Union of top 10 concepts from both neurons
        top_k = 10
        scores_i = M[ni, :]
        scores_j = M[nj, :]
        top_i = set(np.argsort(scores_i)[::-1][:top_k])
        top_j = set(np.argsort(scores_j)[::-1][:top_k])
        union = sorted(top_i | top_j, key=lambda x: -(scores_i[x] + scores_j[x]))[:12]

        concept_labels = [names[c] for c in union]
        y_pos = np.arange(len(concept_labels))
        bar_h = 0.35

        # Side-by-side bars
        bars_i = ax.barh(y_pos - bar_h / 2, [scores_i[c] for c in union], bar_h,
                         color='#348ABD', alpha=0.85, label=f'Neuron {ni}')
        bars_j = ax.barh(y_pos + bar_h / 2, [scores_j[c] for c in union], bar_h,
                         color='#E24A33', alpha=0.85, label=f'Neuron {nj}')

        # Highlight shared concepts
        for y, c_idx in enumerate(union):
            if binary[ni, c_idx] and binary[nj, c_idx]:
                ax.annotate('★', xy=(max(scores_i[c_idx], scores_j[c_idx]) + 0.005, y),
                            fontsize=8, color='#2CA02C', va='center')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(concept_labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('IoU', fontsize=9)

        sc = SUPERCATEGORY_MAP.get(cname, 'other')
        color = SUPERCATEGORY_COLORS.get(sc, '#333')
        ax.set_title(f'"{cname}" ensemble\nn{ni} ↔ n{nj}   $\\rho$ = {rho:.3f}',
                     fontsize=10, fontweight='bold', color=color)
        ax.legend(fontsize=7, loc='lower right', frameon=True, framealpha=0.9,
                  edgecolor='#cccccc', fancybox=False)
        ax.tick_params(axis='x', labelsize=7)
        ax.xaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle('Semantic Ensembles — Coactivating Neuron Pairs With Shared Concepts  '
                 '(★ = both detect)',
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, 'C2_ensemble_pair_profiles')


def C3_degree_distributions(data):
    """Coactivation degree vs concept degree — dual histogram + scatter."""
    print("\n" + "=" * 60)
    print("  C3: Degree Distributions")
    print("=" * 60)
    setup_cvpr_style()

    M = data['assoc_matrix']
    R = data['corr_matrix']
    N = data['N']

    coact_degree = np.array([(R[i, :] > TAU_COACT).sum() - 1 for i in range(N)])
    concept_degree = (M > TAU_ASSOC).sum(axis=1)

    # --- Dual histogram ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))

    ax1.hist(coact_degree, bins=30, color='#E24A33', edgecolor='white',
             linewidth=0.4, alpha=0.9, zorder=3)
    ax1.set_xlabel(f'Coactivation degree ($\\rho > {TAU_COACT}$)')
    ax1.set_ylabel('Number of neurons')
    ax1.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax1.set_axisbelow(True)

    ax2.hist(concept_degree, bins=range(0, int(concept_degree.max()) + 2),
             color='#008FD5', edgecolor='white', linewidth=0.4, alpha=0.9, zorder=3)
    ax2.set_xlabel(f'Concept degree (IoU $> {TAU_ASSOC}$)')
    ax2.set_ylabel('Number of neurons')
    ax2.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, 'C3_degree_histograms')

    # --- Scatter: coactivation degree vs concept degree ---
    fig2, ax3 = plt.subplots(figsize=(4.5, 4.0))

    ax3.scatter(coact_degree, concept_degree, s=15, alpha=0.5, color='#4878CF',
                edgecolors='none', zorder=3)

    rho, p = stats.spearmanr(coact_degree, concept_degree)
    ax3.set_xlabel(f'Coactivation degree ($\\rho > {TAU_COACT}$)')
    ax3.set_ylabel(f'Concept degree (IoU $> {TAU_ASSOC}$)')
    ax3.xaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax3.yaxis.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax3.set_axisbelow(True)

    ax3.annotate(
        f'Spearman $\\rho$ = {rho:.3f}\n$p$ = {p:.1e}',
        xy=(0.97, 0.05), xycoords='axes fraction', ha='right', va='bottom',
        fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', lw=0.5),
    )

    fig2.tight_layout()
    save_fig(fig2, 'C3_degree_scatter')

    print(f"\n  Coactivation degree: mean={coact_degree.mean():.1f}, "
          f"max={coact_degree.max()}")
    print(f"  Concept degree: mean={concept_degree.mean():.1f}, "
          f"max={concept_degree.max()}")
    print(f"  Correlation (coact ↔ concept degree): ρ={rho:.3f}, p={p:.2e}")


def C4_top_neuron_correlation_matrix(data):
    """
    Correlation matrix heatmap for the top-N most concept-rich neurons.
    Shows whether concept-rich neurons coactivate with each other.
    """
    print("\n" + "=" * 60)
    print("  C4: Top Neuron Correlation Matrix")
    print("=" * 60)
    setup_cvpr_style()

    M = data['assoc_matrix']
    R = data['corr_matrix']
    N = data['N']
    names = data['concept_names']

    # Select top 30 neurons by number of concepts
    concept_counts = (M > TAU_ASSOC).sum(axis=1)
    top_neurons = np.argsort(concept_counts)[::-1][:30]

    # Sub-matrix
    sub_R = R[np.ix_(top_neurons, top_neurons)]

    # Labels: "n{idx} ({top_concept})"
    labels = []
    for ni in top_neurons:
        top_c = names[M[ni, :].argmax()]
        labels.append(f'n{ni} ({top_c})')

    fig, ax = plt.subplots(figsize=(10, 9))

    # Mask diagonal
    mask = np.eye(len(top_neurons), dtype=bool)
    sub_R_masked = np.ma.array(sub_R, mask=mask)

    im = ax.imshow(sub_R_masked, cmap='RdBu_r', vmin=-0.3, vmax=1.0,
                   interpolation='nearest')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)

    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label(r'Spearman $\rho$', fontsize=10)

    ax.set_title('Coactivation Matrix — Top 30 Concept-Rich Neurons', fontsize=12)

    fig.tight_layout()
    save_fig(fig, 'C4_top_neuron_correlation')

    # Stats
    upper = sub_R[np.triu_indices(len(top_neurons), k=1)]
    print(f"\n  Top 30 neurons: mean pairwise ρ = {upper.mean():.3f}")
    print(f"  % pairs with ρ > {TAU_COACT}: {100*np.mean(upper > TAU_COACT):.1f}%")


def C5_concept_cooccurrence_matrix(data):
    """
    Concept co-occurrence matrix: how often two concepts are detected
    by the same neuron. Reveals semantic groupings.
    """
    print("\n" + "=" * 60)
    print("  C5: Concept Co-occurrence Matrix")
    print("=" * 60)
    setup_cvpr_style()

    M = data['assoc_matrix']
    names = data['concept_names']
    binary = (M > TAU_ASSOC).astype(int)

    # Co-occurrence: (n_concepts, n_concepts)
    coocc = binary.T @ binary  # how many neurons detect both concepts
    np.fill_diagonal(coocc, 0)

    # Normalise by geometric mean of individual counts (Jaccard-like)
    counts = binary.sum(axis=0)
    norm = np.outer(counts, counts)
    norm = np.sqrt(norm)
    norm[norm == 0] = 1
    coocc_norm = coocc / norm

    # Select top 30 most detected concepts
    top_concepts = np.argsort(counts)[::-1][:30]
    sub = coocc_norm[np.ix_(top_concepts, top_concepts)]
    top_names = [names[c] for c in top_concepts]

    # Supercategory colour bar
    sc_colors = [SUPERCATEGORY_COLORS.get(SUPERCATEGORY_MAP.get(n, 'other'), '#999')
                 for n in top_names]

    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(sub, cmap='YlOrRd', interpolation='nearest', vmin=0)

    ax.set_xticks(range(len(top_names)))
    ax.set_xticklabels(top_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=7)

    # Colour the tick labels by supercategory
    for i, (xl, yl) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
        xl.set_color(sc_colors[i])
        yl.set_color(sc_colors[i])

    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label('Normalised co-occurrence', fontsize=10)

    ax.set_title('Concept Co-occurrence — Top 30 Most Detected Concepts', fontsize=12)
    fig.tight_layout()
    save_fig(fig, 'C5_concept_cooccurrence')

    # Print top co-occurring pairs
    print(f"\n  Top 10 concept co-occurrence pairs:")
    pairs = []
    for i in range(len(top_concepts)):
        for j in range(i + 1, len(top_concepts)):
            pairs.append((top_names[i], top_names[j], sub[i, j], coocc[top_concepts[i], top_concepts[j]]))
    pairs.sort(key=lambda x: -x[2])
    for n1, n2, norm_val, raw in pairs[:10]:
        print(f"    {n1:16s} × {n2:16s}: {norm_val:.3f} ({int(raw)} shared neurons)")


# =============================================================
# MAIN
# =============================================================

def run_part_a(data):
    print("\n" + "=" * 60)
    print("  PART A: Network Dissection-style Analysis")
    print("=" * 60)
    A1_iou_distribution(data)
    A2_per_concept_table(data)
    A3_supercategory_analysis(data)
    A4_mono_poly_semantic(data)
    A5_frequency_detectability(data)


def run_part_b(data):
    print("\n" + "=" * 60)
    print("  PART B: Graph-Specific Analysis")
    print("=" * 60)
    B1_coactivation_concept_scatter(data)
    B2_semantic_ensembles(data)
    best_per = B3_case_studies(data)
    B4_community_analysis(data)
    return best_per


def run_part_c(data, best_per=None):
    print("\n" + "=" * 60)
    print("  PART C: Qualitative Results")
    print("=" * 60)
    C1_neuron_concept_profiles(data)
    C2_ensemble_pair_profiles(data, best_per)
    C3_degree_distributions(data)
    C4_top_neuron_correlation_matrix(data)
    C5_concept_cooccurrence_matrix(data)


def main():
    parser = argparse.ArgumentParser(description='Step 4: Analysis & Figures (CVPR)')
    parser.add_argument('--part', choices=['A', 'B', 'C'], default=None,
                        help='Run only Part A, B, or C (default: all)')
    args = parser.parse_args()

    ensure_dirs()
    data = load_data()

    best_per = None
    if args.part == 'A':
        run_part_a(data)
    elif args.part == 'B':
        best_per = run_part_b(data)
    elif args.part == 'C':
        run_part_c(data)
    else:
        run_part_a(data)
        best_per = run_part_b(data)
        run_part_c(data, best_per)

    print("\n" + "=" * 60)
    print("  All analyses complete.")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
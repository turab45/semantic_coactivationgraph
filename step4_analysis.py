"""
Step 4: Analysis & Figures
===========================
Part A — Network Dissection-style concept analysis:
  A1. IoU distribution histogram
  A2. Per-concept top neurons table
  A3. Supercategory analysis (bar chart)
  A4. Monosemantic vs polysemantic classification
  A5. Frequency–detectability correlation

Part B — Graph-specific (Coactivation + Concepts):
  B1. Coactivation–concept alignment (scatter plot + Spearman ρ)
  B2. Semantic ensemble queries
  B3. Case studies (verified neuron pairs)
  B4. Community analysis (hierarchical clustering + heatmap)

Reads: extended_coactivation_graph.graphml + concept_associations.pkl
Writes: results/figures/*.pdf + printed statistics + LaTeX tables

Usage:
    python step4_analysis.py               # Run all
    python step4_analysis.py --part A      # Only Part A
    python step4_analysis.py --part B      # Only Part B
"""

import argparse
import pickle
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
# Global style
# =============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


# =============================================================
# Data loading
# =============================================================

def load_data():
    """Load extended graph and concept associations."""
    print("Loading data...")

    # Concept associations
    with open(CONCEPT_PKL_PATH, 'rb') as f:
        concept_data = pickle.load(f)
    associations = concept_data['associations']
    patch_meta = concept_data.get('patch_metadata', {})

    # Build association matrix: (N_neurons, N_concepts)
    concept_ids = sorted(associations.keys())
    concept_names = [concept_name(cid) for cid in concept_ids]
    N = next(iter(associations.values())).shape[0]

    assoc_matrix = np.zeros((N, len(concept_ids)))
    for j, cid in enumerate(concept_ids):
        assoc_matrix[:, j] = associations[cid]

    # Extended graph
    G = nx.read_graphml(str(EXTENDED_GRAPH_PATH))

    # Extract correlation matrix for neurons in CONCEPT_LAYER
    # Build from graph edges
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
    print(f"  Correlation matrix shape: {corr_matrix.shape}")

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
    """Histogram of max IoU per neuron."""
    print("\n--- A1: IoU Distribution ---")
    M = data['assoc_matrix']
    max_ious = M.max(axis=1)

    specialised = (max_ious > IOU_SPECIALISATION).sum()
    pct = 100 * specialised / data['N']
    print(f"  Specialised neurons (IoU > {IOU_SPECIALISATION}): {specialised}/{data['N']} ({pct:.1f}%)")
    print(f"  Mean max IoU: {max_ious.mean():.3f} ± {max_ious.std():.3f}")

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.hist(max_ious, bins=40, color='#3b82f6', alpha=0.8, edgecolor='#1e3a5f', linewidth=0.4)
    ax.axvline(IOU_SPECIALISATION, color='#ef4444', linestyle='--', linewidth=1,
               label=f'τ = {IOU_SPECIALISATION}')
    ax.set_xlabel('Maximum IoU')
    ax.set_ylabel('Number of neurons')
    ax.set_title(f'Neuron Specialisation ({CONCEPT_LAYER})')
    ax.legend(fontsize=8)
    fig.savefig(str(FIGURES_DIR / 'A1_iou_distribution.pdf'))
    plt.close()
    print(f"  Saved: A1_iou_distribution.pdf")


def A2_per_concept_table(data):
    """Table: top neurons per concept + LaTeX export."""
    print("\n--- A2: Per-Concept Top Neurons ---")
    M = data['assoc_matrix']
    names = data['concept_names']

    rows = []
    for j, cname in enumerate(names):
        scores = M[:, j]
        n_detect = (scores > TAU_ASSOC).sum()
        if n_detect == 0:
            continue
        best_idx = scores.argmax()
        best_iou = scores[best_idx]
        rows.append({
            'Concept': cname,
            'Best Neuron': best_idx,
            'Best IoU': round(best_iou, 3),
            'Detecting Neurons': int(n_detect),
            'Mean IoU': round(scores[scores > TAU_ASSOC].mean(), 3),
        })

    df = pd.DataFrame(rows).sort_values('Detecting Neurons', ascending=False)
    print(df.head(15).to_string(index=False))

    # LaTeX
    tex_path = FIGURES_DIR / 'A2_concept_table.tex'
    df.head(20).to_latex(str(tex_path), index=False, float_format='%.3f',
                         caption='Top COCO concepts by number of detecting neurons.',
                         label='tab:concept_summary')
    print(f"  Saved: A2_concept_table.tex")
    return df


def A3_supercategory_analysis(data):
    """Bar chart: mean max IoU per supercategory."""
    print("\n--- A3: Supercategory Analysis ---")
    M = data['assoc_matrix']
    names = data['concept_names']

    supercat_ious = defaultdict(list)
    for j, cname in enumerate(names):
        sc = supercategory(cname)
        mean_iou = M[:, j].max()  # best neuron for this concept
        supercat_ious[sc].append(mean_iou)

    sc_names = sorted(supercat_ious.keys(), key=lambda x: -np.mean(supercat_ious[x]))
    sc_means = [np.mean(supercat_ious[sc]) for sc in sc_names]
    sc_stds = [np.std(supercat_ious[sc]) for sc in sc_names]

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6',
              '#ec4899', '#06b6d4', '#15803d', '#dc2626', '#6b7280',
              '#92400e', '#64748b', '#eab308']
    ax.barh(range(len(sc_names)), sc_means, xerr=sc_stds,
            color=colors[:len(sc_names)], alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(sc_names)))
    ax.set_yticklabels(sc_names, fontsize=8)
    ax.set_xlabel('Mean Best IoU')
    ax.set_title('Detectability by Supercategory')
    ax.invert_yaxis()
    fig.savefig(str(FIGURES_DIR / 'A3_supercategory.pdf'))
    plt.close()
    print(f"  Saved: A3_supercategory.pdf")


def A4_mono_poly_semantic(data):
    """Classify neurons as monosemantic or polysemantic."""
    print("\n--- A4: Mono/Polysemantic Classification ---")
    M = data['assoc_matrix']
    names = data['concept_names']
    N = data['N']

    n_concepts_per_neuron = (M > TAU_ASSOC).sum(axis=1)
    mono_count = 0
    poly_count = 0
    dead_count = 0

    for i in range(N):
        active = M[i, M[i] > TAU_ASSOC]
        if len(active) == 0:
            dead_count += 1
        elif len(active) == 1:
            mono_count += 1
        else:
            sorted_ious = np.sort(active)[::-1]
            if sorted_ious[0] > MONO_THRESHOLD_FACTOR * sorted_ious[1]:
                mono_count += 1
            else:
                poly_count += 1

    print(f"  Monosemantic: {mono_count} ({100*mono_count/N:.1f}%)")
    print(f"  Polysemantic: {poly_count} ({100*poly_count/N:.1f}%)")
    print(f"  Dead (no concept): {dead_count} ({100*dead_count/N:.1f}%)")
    print(f"  Mean concepts/neuron: {n_concepts_per_neuron.mean():.1f}")

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.hist(n_concepts_per_neuron, bins=range(0, int(n_concepts_per_neuron.max())+2),
            color='#8b5cf6', alpha=0.8, edgecolor='#4c1d95', linewidth=0.4)
    ax.set_xlabel('Number of concepts detected')
    ax.set_ylabel('Number of neurons')
    ax.set_title('Polysemanticity Distribution')
    fig.savefig(str(FIGURES_DIR / 'A4_polysemanticity.pdf'))
    plt.close()
    print(f"  Saved: A4_polysemanticity.pdf")


def A5_frequency_detectability(data):
    """Scatter: COCO frequency vs best IoU per concept + Spearman ρ."""
    print("\n--- A5: Frequency–Detectability ---")
    M = data['assoc_matrix']
    names = data['concept_names']
    cids = data['concept_ids']

    # Count images per concept from association data
    associations = data['associations']
    freq = []
    detectability = []
    labels = []

    for j, cid in enumerate(cids):
        scores = M[:, j]
        best_iou = scores.max()
        # Approximate frequency: number of neurons detecting it
        n_detect = (scores > TAU_ASSOC).sum()
        freq.append(n_detect)
        detectability.append(best_iou)
        labels.append(names[j])

    freq = np.array(freq)
    detectability = np.array(detectability)

    rho, pval = stats.spearmanr(freq, detectability)
    print(f"  Spearman ρ: {rho:.3f}, p = {pval:.2e}")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(freq, detectability, s=12, alpha=0.6, c='#3b82f6', edgecolors='#1e3a5f', linewidth=0.3)

    # Label extreme points
    for i in range(len(labels)):
        if detectability[i] > np.percentile(detectability, 90) or freq[i] > np.percentile(freq, 90):
            ax.annotate(labels[i], (freq[i], detectability[i]),
                        fontsize=6, alpha=0.7, ha='center', va='bottom')

    ax.set_xlabel('Detecting neurons (count)')
    ax.set_ylabel('Best IoU')
    ax.set_title(f'Frequency vs Detectability (ρ = {rho:.3f})')
    fig.savefig(str(FIGURES_DIR / 'A5_freq_detect.pdf'))
    plt.close()
    print(f"  Saved: A5_freq_detect.pdf")


# =============================================================
# PART B: Graph-Specific Analysis
# =============================================================

def B1_coactivation_concept_scatter(data):
    """
    Scatter: coactivation strength (ρ) vs shared concepts.
    The central figure testing the hypothesis.
    """
    print("\n--- B1: Coactivation–Concept Alignment ---")
    M = data['assoc_matrix']
    R = data['corr_matrix']
    N = data['N']

    # For all neuron pairs where ρ > 0 (from the graph)
    corr_vals = []
    shared_vals = []

    binary = (M > TAU_ASSOC).astype(int)

    for i in range(N):
        for j in range(i+1, N):
            rho = R[i, j]
            if rho > 0:
                shared = int(np.logical_and(binary[i], binary[j]).sum())
                corr_vals.append(rho)
                shared_vals.append(shared)

    corr_vals = np.array(corr_vals)
    shared_vals = np.array(shared_vals)

    spearman_rho, pval = stats.spearmanr(corr_vals, shared_vals)
    print(f"  Pairs analysed: {len(corr_vals):,}")
    print(f"  Spearman ρ: {spearman_rho:.3f}, p = {pval:.2e}")

    # Top 10% vs bottom 10%
    top_mask = corr_vals >= np.percentile(corr_vals, 90)
    bot_mask = corr_vals <= np.percentile(corr_vals, 10)
    top_mean = shared_vals[top_mask].mean()
    bot_mean = shared_vals[bot_mask].mean()
    ratio = top_mean / bot_mean if bot_mean > 0 else float('inf')
    tstat, tpval = stats.ttest_ind(shared_vals[top_mask], shared_vals[bot_mask])
    print(f"  Top 10% mean shared: {top_mean:.2f}")
    print(f"  Bottom 10% mean shared: {bot_mean:.2f}")
    print(f"  Ratio: {ratio:.2f}x (p = {tpval:.2e})")

    # Plot
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Hexbin for density
    hb = ax.hexbin(corr_vals, shared_vals, gridsize=40, cmap='Blues',
                   mincnt=1, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax, shrink=0.7, label='Pair count')

    # Binned means
    bins = np.linspace(corr_vals.min(), corr_vals.max(), 20)
    bin_idx = np.digitize(corr_vals, bins)
    bin_means = [shared_vals[bin_idx == b].mean() for b in range(1, len(bins))
                 if (bin_idx == b).sum() > 10]
    bin_centers = [(bins[b-1] + bins[b]) / 2 for b in range(1, len(bins))
                   if (bin_idx == b).sum() > 10]
    ax.plot(bin_centers, bin_means, 'o-', color='#ef4444', markersize=3,
            linewidth=1.2, label='Binned mean')

    ax.set_xlabel('Coactivation strength (ρ)')
    ax.set_ylabel('Shared concepts')
    ax.set_title(f'Coactivation–Concept Alignment (ρ_s = {spearman_rho:.3f})')
    ax.legend(fontsize=7, loc='upper left')
    fig.savefig(str(FIGURES_DIR / 'B1_coact_concept_scatter.pdf'))
    plt.close()
    print(f"  Saved: B1_coact_concept_scatter.pdf")


def B2_semantic_ensembles(data):
    """
    Semantic ensemble queries: neuron pairs that coactivate AND
    both detect the same concept.
    """
    print("\n--- B2: Semantic Ensembles ---")
    M = data['assoc_matrix']
    R = data['corr_matrix']
    N = data['N']
    names = data['concept_names']
    cids = data['concept_ids']

    binary = (M > TAU_ASSOC).astype(int)
    ensemble_counts = defaultdict(int)
    total_ensembles = 0

    for i in range(N):
        for j in range(i+1, N):
            if R[i, j] > TAU_COACT:
                # Both detect the same concept?
                shared = np.logical_and(binary[i], binary[j])
                for k in range(len(cids)):
                    if shared[k]:
                        ensemble_counts[names[k]] += 1
                        total_ensembles += 1

    print(f"  Total ensemble (pair, concept) tuples: {total_ensembles:,}")
    n_with = sum(1 for v in ensemble_counts.values() if v > 0)
    n_without = len(names) - n_with
    print(f"  Concepts with ≥1 ensemble: {n_with}")
    print(f"  Concepts with 0 ensembles: {n_without}")

    # Top 15
    print(f"\n  Top 15 concepts by ensemble count:")
    top15 = sorted(ensemble_counts.items(), key=lambda x: -x[1])[:15]
    for cname, count in top15:
        print(f"    {cname:20s}: {count}")

    # LaTeX table
    rows = []
    for cname, count in sorted(ensemble_counts.items(), key=lambda x: -x[1])[:15]:
        rows.append({'Concept': cname, 'Ensembles': count,
                     'Supercategory': supercategory(cname)})
    df = pd.DataFrame(rows)
    tex_path = FIGURES_DIR / 'B2_ensemble_table.tex'
    df.to_latex(str(tex_path), index=False,
                caption='Top 15 concepts by semantic ensemble count.',
                label='tab:ensembles')
    print(f"  Saved: B2_ensemble_table.tex")


def B3_case_studies(data):
    """
    Detailed case studies of notable neuron pairs.
    Finds: strongest pair, most shared concepts, best per-concept pair.
    """
    print("\n--- B3: Case Studies ---")
    M = data['assoc_matrix']
    R = data['corr_matrix']
    N = data['N']
    names = data['concept_names']

    binary = (M > TAU_ASSOC).astype(int)

    # Find strongest coactivating pair
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    R_upper = R * mask
    best_idx = np.unravel_index(R_upper.argmax(), R_upper.shape)
    i, j = best_idx
    shared = [names[k] for k in range(len(names)) if binary[i, k] and binary[j, k]]

    print(f"\n  Strongest coactivating pair:")
    print(f"    Neurons {i} and {j}: ρ = {R[i,j]:.3f}")
    print(f"    Shared concepts ({len(shared)}): {', '.join(shared[:8])}")

    # Find pair with most shared concepts
    max_shared = 0
    best_shared_pair = (0, 0)
    for ii in range(N):
        for jj in range(ii+1, N):
            if R[ii, jj] > TAU_COACT:
                ns = np.logical_and(binary[ii], binary[jj]).sum()
                if ns > max_shared:
                    max_shared = ns
                    best_shared_pair = (ii, jj)

    i2, j2 = best_shared_pair
    shared2 = [names[k] for k in range(len(names)) if binary[i2, k] and binary[j2, k]]
    print(f"\n  Most shared concepts:")
    print(f"    Neurons {i2} and {j2}: ρ = {R[i2,j2]:.3f}")
    print(f"    Shared concepts ({len(shared2)}): {', '.join(shared2[:10])}")

    # Best ensemble per concept (strongest ρ among pairs that both detect it)
    print(f"\n  Best ensemble per concept (top 10):")
    best_per_concept = {}
    for k, cname in enumerate(names):
        detectors = np.where(binary[:, k])[0]
        if len(detectors) < 2:
            continue
        best_rho = 0
        best_pair = (0, 0)
        for di in range(len(detectors)):
            for dj in range(di+1, len(detectors)):
                rho = R[detectors[di], detectors[dj]]
                if rho > best_rho:
                    best_rho = rho
                    best_pair = (detectors[di], detectors[dj])
        if best_rho > 0:
            best_per_concept[cname] = {
                'pair': best_pair,
                'rho': best_rho,
                'iou_a': M[best_pair[0], k],
                'iou_b': M[best_pair[1], k],
            }

    for cname, info in sorted(best_per_concept.items(),
                               key=lambda x: -x[1]['rho'])[:10]:
        p = info['pair']
        print(f"    {cname:16s}: ({p[0]:3d}, {p[1]:3d}) "
              f"ρ={info['rho']:.3f}  IoU=({info['iou_a']:.3f}, {info['iou_b']:.3f})")

    # LaTeX
    rows = []
    for cname, info in sorted(best_per_concept.items(), key=lambda x: -x[1]['rho'])[:15]:
        p = info['pair']
        rows.append({
            'Concept': cname,
            'Neuron A': p[0], 'Neuron B': p[1],
            '$\\rho$': round(info['rho'], 3),
            'IoU A': round(info['iou_a'], 3),
            'IoU B': round(info['iou_b'], 3),
        })
    df = pd.DataFrame(rows)
    tex_path = FIGURES_DIR / 'B3_case_studies.tex'
    df.to_latex(str(tex_path), index=False, float_format='%.3f', escape=False,
                caption='Best semantic ensembles per concept.',
                label='tab:case_studies')
    print(f"  Saved: B3_case_studies.tex")


def B4_community_analysis(data):
    """
    Hierarchical clustering on correlation matrix.
    Compare within-community vs between-community shared concepts.
    Generate community-concept heatmap.
    """
    print("\n--- B4: Community Analysis ---")
    R = data['corr_matrix']
    M = data['assoc_matrix']
    N = data['N']
    names = data['concept_names']
    binary = (M > TAU_ASSOC).astype(int)

    # Hierarchical clustering (Ward's method on distance = 1 - |ρ|)
    dist = 1 - np.abs(R)
    np.fill_diagonal(dist, 0)
    condensed = dist[np.triu_indices(N, k=1)]
    Z = linkage(condensed, method='ward')

    K_values = [5, 8, 10, 15, 20]
    print(f"  Testing K = {K_values}")

    for K in K_values:
        labels = fcluster(Z, t=K, criterion='maxclust')

        within_shared = []
        between_shared = []

        for i in range(N):
            for j in range(i+1, N):
                if R[i, j] > TAU_COACT:
                    ns = np.logical_and(binary[i], binary[j]).sum()
                    if labels[i] == labels[j]:
                        within_shared.append(ns)
                    else:
                        between_shared.append(ns)

        w_mean = np.mean(within_shared) if within_shared else 0
        b_mean = np.mean(between_shared) if between_shared else 0
        ratio = w_mean / b_mean if b_mean > 0 else float('inf')

        if within_shared and between_shared:
            _, pval = stats.ttest_ind(within_shared, between_shared)
        else:
            pval = 1.0

        print(f"    K={K:2d}: within={w_mean:.2f}, between={b_mean:.2f}, "
              f"ratio={ratio:.3f}x, p={pval:.2e}")

    # Generate heatmap for K=10
    K = 10
    labels = fcluster(Z, t=K, criterion='maxclust')

    # Community-concept profile: mean IoU per community per concept
    profile = np.zeros((K, len(names)))
    for c in range(1, K+1):
        members = np.where(labels == c)[0]
        if len(members) > 0:
            profile[c-1] = M[members].mean(axis=0)

    # Select most discriminative concepts (highest variance across communities)
    variance = profile.var(axis=0)
    top_concepts = np.argsort(variance)[::-1][:20]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(profile[:, top_concepts].T, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'C{c+1}\n(n={np.sum(labels==c+1)})' for c in range(K)], fontsize=7)
    ax.set_yticks(range(len(top_concepts)))
    ax.set_yticklabels([names[c] for c in top_concepts], fontsize=7)
    ax.set_xlabel('Community')
    ax.set_title('Community–Concept Semantic Profile (top 20 concepts)')
    fig.colorbar(im, ax=ax, shrink=0.7, label='Mean IoU')
    fig.savefig(str(FIGURES_DIR / 'B4_community_heatmap.pdf'))
    plt.close()
    print(f"  Saved: B4_community_heatmap.pdf")

    # LaTeX table
    if within_shared and between_shared:
        _, pval = stats.ttest_ind(within_shared, between_shared)
    rows = []
    for K_test in K_values:
        lab = fcluster(Z, t=K_test, criterion='maxclust')
        w, b = [], []
        for i in range(N):
            for j in range(i+1, N):
                if R[i, j] > TAU_COACT:
                    ns = np.logical_and(binary[i], binary[j]).sum()
                    if lab[i] == lab[j]:
                        w.append(ns)
                    else:
                        b.append(ns)
        wm = np.mean(w) if w else 0
        bm = np.mean(b) if b else 0
        r = wm / bm if bm > 0 else 0
        _, p = stats.ttest_ind(w, b) if w and b else (0, 1)
        rows.append({'K': K_test, 'Within': round(wm, 2), 'Between': round(bm, 2),
                     'Ratio': round(r, 3), 'p-value': f'{p:.2e}'})

    df = pd.DataFrame(rows)
    tex_path = FIGURES_DIR / 'B4_community_table.tex'
    df.to_latex(str(tex_path), index=False,
                caption='Within vs between community shared concepts.',
                label='tab:communities')
    print(f"  Saved: B4_community_table.tex")


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
    B3_case_studies(data)
    B4_community_analysis(data)


def main():
    parser = argparse.ArgumentParser(description='Step 4: Analysis & Figures')
    parser.add_argument('--part', choices=['A', 'B'], default=None,
                        help='Run only Part A or Part B (default: both)')
    args = parser.parse_args()

    ensure_dirs()
    data = load_data()

    if args.part == 'A':
        run_part_a(data)
    elif args.part == 'B':
        run_part_b(data)
    else:
        run_part_a(data)
        run_part_b(data)

    print("\n" + "=" * 60)
    print("  All analyses complete.")
    print(f"  Figures: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
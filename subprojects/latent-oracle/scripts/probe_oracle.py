"""Oracle Embedding Probe — Latent Space Clustering Analysis.

Self-contained script (no latent_oracle package imports). Loads embeddings.pt,
runs dimensionality reduction + clustering analysis on curated probe categories,
saves PNG plots + prints quantitative metrics.

Usage:
    python scripts/probe_oracle.py \
        --embeddings /workspace/latent-oracle/embeddings.pt \
        --output_dir /workspace/latent-oracle/probe_results
"""

import argparse
import os
import sys
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

# ── Probe categories: 8 tasks each, 32 total ──────────────────────────────────

PROBE_CATEGORIES = {
    "A: Movement/Gravity": [
        "3c9b0459", "6150a2bd", "a740d043", "25d8a9c8",
        "1f85a75f", "44d8ac46", "3906de3d", "0520fde7",
    ],
    "B: Symmetry/Reflection": [
        "6fa7a44f", "4c4377d9", "3618c87e", "ed36ccf7",
        "74dd1130", "d631b094", "5521c0d9", "67a3c6ac",
    ],
    "C: Color Substitution": [
        "0d3d703e", "1cf80156", "25ff71a9", "4258a5f9",
        "6855a6e4", "9565186b", "c9e6f938", "228f6490",
    ],
    "D: Bounding/Outlining": [
        "0ca9ddb6", "b94a9452", "1e0a9b12", "b91ae062",
        "3aa6fb7a", "a79310a0", "dc1df850", "7468f01a",
    ],
}

# Colorblind-friendly palette (Wong 2011)
CATEGORY_COLORS = {
    "A: Movement/Gravity": "#E69F00",       # orange
    "B: Symmetry/Reflection": "#56B4E9",    # sky blue
    "C: Color Substitution": "#009E73",      # bluish green
    "D: Bounding/Outlining": "#CC79A7",      # reddish purple
}


def load_canonical_embeddings(path):
    """Load embeddings.pt, filter to d4_idx=0, test_idx=0 → canonical set."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    emb = data["embeddings"]  # [N, dim] bf16
    task_ids = data["task_ids"]  # list[str]
    d4 = data["d4_indices"]  # [N]
    test = data["test_indices"]  # [N]

    # Filter to canonical: d4=0, test=0
    mask = (d4 == 0) & (test == 0)
    indices = mask.nonzero(as_tuple=True)[0]

    emb_canon = emb[indices].float()  # bf16 → float32
    ids_canon = [task_ids[i] for i in indices.tolist()]

    print(f"Loaded {len(task_ids)} total embeddings, "
          f"{len(ids_canon)} canonical (d4=0, test=0)")
    print(f"Embedding dim: {emb_canon.shape[1]}")
    return emb_canon.numpy(), ids_canon


def build_probe_set(embeddings, task_ids):
    """Extract probe embeddings + labels for the 32 curated tasks."""
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    probe_emb = []
    probe_labels = []
    probe_ids = []
    missing = []

    for cat_name, cat_ids in PROBE_CATEGORIES.items():
        for tid in cat_ids:
            if tid in id_to_idx:
                probe_emb.append(embeddings[id_to_idx[tid]])
                probe_labels.append(cat_name)
                probe_ids.append(tid)
            else:
                missing.append(tid)

    if missing:
        print(f"WARNING: {len(missing)} probe tasks not found: {missing}")

    X = np.array(probe_emb)
    print(f"Probe set: {X.shape[0]} tasks across {len(PROBE_CATEGORIES)} categories")
    return X, probe_labels, probe_ids


# ── Dimensionality reduction ──────────────────────────────────────────────────

def run_pca(X):
    return PCA(n_components=2, random_state=42).fit_transform(X)


def run_tsne(X):
    return TSNE(
        n_components=2, perplexity=10, random_state=42, init="pca",
        learning_rate="auto",
    ).fit_transform(X)


def run_umap(X):
    try:
        import umap
        return umap.UMAP(
            n_components=2, n_neighbors=10, min_dist=0.3, random_state=42,
        ).fit_transform(X)
    except ImportError:
        print("umap-learn not installed — skipping UMAP")
        return None


# ── Plotting ──────────────────────────────────────────────────────────────────

def scatter_plot(coords_2d, labels, title, save_path):
    """Color-coded scatter plot by category."""
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = list(PROBE_CATEGORIES.keys())

    for cat in categories:
        mask = [l == cat for l in labels]
        pts = coords_2d[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=CATEGORY_COLORS[cat], label=cat, s=80, alpha=0.8, edgecolors="k",
            linewidths=0.5,
        )

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def cluster_plot(coords_2d, cluster_labels, k, save_path):
    """K-means cluster scatter on full corpus."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab20", k)

    for c in range(k):
        mask = cluster_labels == c
        pts = coords_2d[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=[cmap(c)], s=15, alpha=0.6, label=f"Cluster {c}",
        )

    ax.set_title(f"K-Means k={k} (PCA 2D)", fontsize=14)
    if k <= 8:
        ax.legend(fontsize=7, loc="best", ncol=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Quantitative metrics ──────────────────────────────────────────────────────

def compute_silhouette(X, labels, label_name=""):
    """Silhouette score (requires ≥2 clusters with ≥2 members)."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if len(set(y)) < 2:
        return float("nan")
    score = silhouette_score(X, y)
    print(f"  Silhouette ({label_name}): {score:.4f}")
    return score


def linear_probe_loo(X, labels):
    """Logistic regression with leave-one-out CV.

    PCA to min(n_samples-1, 30) dims first — 32 samples can't support 4096 features.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_components = min(X.shape[0] - 1, 30)
    X_reduced = PCA(n_components=n_components, random_state=42).fit_transform(X)
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X_reduced):
        clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
        clf.fit(X_reduced[train_idx], y[train_idx])
        pred = clf.predict(X_reduced[test_idx])
        correct += int(pred[0] == y[test_idx][0])
    acc = correct / len(y)
    print(f"  Linear probe LOO-CV: {acc:.1%} (chance = {1/len(set(y)):.1%}, "
          f"PCA to {n_components} dims)")
    return acc


def l2_cluster_ratio(X, labels):
    """Inter-cluster / intra-cluster L2 distance ratio."""
    cats = list(PROBE_CATEGORIES.keys())
    cat_to_idx = defaultdict(list)
    for i, l in enumerate(labels):
        cat_to_idx[l].append(i)

    # Intra-cluster: mean pairwise L2 within each category
    intra_dists = []
    for cat in cats:
        idxs = cat_to_idx[cat]
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            intra_dists.append(np.linalg.norm(X[i] - X[j]))
    mean_intra = np.mean(intra_dists) if intra_dists else 1e-8

    # Inter-cluster: mean pairwise L2 between different categories
    inter_dists = []
    for c1, c2 in combinations(cats, 2):
        for i in cat_to_idx[c1]:
            for j in cat_to_idx[c2]:
                inter_dists.append(np.linalg.norm(X[i] - X[j]))
    mean_inter = np.mean(inter_dists) if inter_dists else 0.0

    ratio = mean_inter / mean_intra if mean_intra > 0 else 0.0
    print(f"  L2 inter/intra ratio: {ratio:.4f} "
          f"(inter={mean_inter:.4f}, intra={mean_intra:.4f})")
    return ratio


def cosine_analysis(X, labels):
    """Cosine similarity: intra vs inter, plus all 6 pairwise comparisons."""
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    Xn = X / norms

    cats = list(PROBE_CATEGORIES.keys())
    cat_to_idx = defaultdict(list)
    for i, l in enumerate(labels):
        cat_to_idx[l].append(i)

    # Intra-category cosine
    intra_sims = []
    for cat in cats:
        idxs = cat_to_idx[cat]
        for i, j in combinations(idxs, 2):
            intra_sims.append(np.dot(Xn[i], Xn[j]))

    # Inter-category cosine
    inter_sims = []
    for c1, c2 in combinations(cats, 2):
        for i in cat_to_idx[c1]:
            for j in cat_to_idx[c2]:
                inter_sims.append(np.dot(Xn[i], Xn[j]))

    mean_intra = np.mean(intra_sims) if intra_sims else 0.0
    mean_inter = np.mean(inter_sims) if inter_sims else 0.0

    print(f"  Cosine similarity — intra: {mean_intra:.4f}, inter: {mean_inter:.4f}, "
          f"gap: {mean_intra - mean_inter:.4f}")

    # Pairwise category comparisons
    print("  Pairwise cosine (between categories):")
    for c1, c2 in combinations(cats, 2):
        sims = []
        for i in cat_to_idx[c1]:
            for j in cat_to_idx[c2]:
                sims.append(np.dot(Xn[i], Xn[j]))
        short_c1 = c1.split(":")[0]
        short_c2 = c2.split(":")[0]
        print(f"    {short_c1}-{short_c2}: {np.mean(sims):.4f}")


# ── Full-corpus clustering ────────────────────────────────────────────────────

def full_corpus_clustering(embeddings, task_ids, output_dir):
    """K-means on all canonical embeddings for k=4,8,16."""
    print("\n=== Full-corpus clustering ===")
    pca_2d = PCA(n_components=2, random_state=42).fit_transform(embeddings)

    for k in [4, 8, 16]:
        km = KMeans(n_clusters=k, random_state=42, n_init=3)
        labels = km.fit_predict(embeddings)

        save_path = os.path.join(output_dir, f"cluster_kmeans_k{k}.png")
        cluster_plot(pca_2d, labels, k, save_path)

        # Sample task IDs per cluster
        print(f"  k={k} cluster samples:")
        for c in range(k):
            members = [task_ids[i] for i in range(len(labels)) if labels[i] == c]
            n = len(members)
            sample = members[:5]
            print(f"    Cluster {c} ({n} tasks): {sample}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Oracle Embedding Probe — Latent Space Clustering Analysis"
    )
    parser.add_argument(
        "--embeddings", type=str,
        default="/workspace/latent-oracle/embeddings.pt",
        help="Path to embeddings.pt",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/workspace/latent-oracle/probe_results",
        help="Directory for output PNGs and metrics",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load canonical embeddings
    embeddings, task_ids = load_canonical_embeddings(args.embeddings)

    # 2. Build probe set
    X_probe, probe_labels, probe_ids = build_probe_set(embeddings, task_ids)
    if X_probe.shape[0] < 8:
        print("ERROR: Too few probe tasks found. Check task IDs against embeddings.")
        sys.exit(1)

    # 3. Dimensionality reduction
    print("\n=== Dimensionality reduction ===")
    pca_2d = run_pca(X_probe)
    tsne_2d = run_tsne(X_probe)
    umap_2d = run_umap(X_probe)

    # 4. Scatter plots
    print("\n=== Scatter plots ===")
    scatter_plot(pca_2d, probe_labels, "Oracle Embeddings — PCA",
                 os.path.join(args.output_dir, "probe_pca.png"))
    scatter_plot(tsne_2d, probe_labels, "Oracle Embeddings — t-SNE (perp=10)",
                 os.path.join(args.output_dir, "probe_tsne.png"))
    if umap_2d is not None:
        scatter_plot(umap_2d, probe_labels, "Oracle Embeddings — UMAP",
                     os.path.join(args.output_dir, "probe_umap.png"))

    # 5. Quantitative metrics
    print("\n=== Quantitative metrics ===")
    compute_silhouette(X_probe, probe_labels, "raw 4096-dim")
    compute_silhouette(pca_2d, probe_labels, "PCA 2D")
    compute_silhouette(tsne_2d, probe_labels, "t-SNE 2D")
    if umap_2d is not None:
        compute_silhouette(umap_2d, probe_labels, "UMAP 2D")

    linear_probe_loo(X_probe, probe_labels)
    l2_cluster_ratio(X_probe, probe_labels)
    cosine_analysis(X_probe, probe_labels)

    # 6. Full-corpus clustering
    full_corpus_clustering(embeddings, task_ids, args.output_dir)

    print("\nDone. Results in:", args.output_dir)


if __name__ == "__main__":
    main()

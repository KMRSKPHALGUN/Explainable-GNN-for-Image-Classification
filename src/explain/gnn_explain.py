"""
Explainability utilities for the GAT image classifier.

Functions:
- extract_attention_node_importance(model, data, layer=1)
- compute_activation_strength(node_emb)
- combined_importance(attn_imp, act_imp)
- insertion_deletion_test(model, data, device, metric_fn, mode='deletion', steps=20)
- visualize_explanations(img_np, segments, centroids, edge_idx, attn_imp, act_imp, top_k=5, out_path=None)

Requires:
- torch, numpy
- matplotlib
- src.utils (plot_activation_graph, map_node_values_to_segments, plot_highlighted_superpixels)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Tuple, List
from torch_geometric.data import Data

from skimage.segmentation import mark_boundaries

# helper visual utils from your repo
from src.utils import map_node_values_to_segments, plot_activation_graph, plot_highlighted_superpixels

def extract_attention_node_importance(model, data: Data, device='cpu', layer: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a forward pass that returns attention weights, then aggregate edge-level attention to node-level importance.

    Returns:
      (node_importance, edge_index_numpy) where node_importance is shape [num_nodes]
    Notes:
      - Assumes your model returns attention from GAT layers when called with return_attention_weights=True inside model.forward
      - If your model currently doesn't return attentions, you will need to modify forward to return them (we used that pattern earlier).
    """
    model.eval()
    data = data.to(device)
    # the GAT model we used returns (logits, node_emb) by default.
    # Modify model forward to optionally return attention; here we'll attempt to call with a wrapper if available.
    # We'll assume your GAT model used earlier returns logits, node_emb, (edge_idx1, attn1), (edge_idx2, attn2)
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.batch)  # existing signature
    # try to unpack attentions if present
    if isinstance(out, tuple) and len(out) >= 4:
        # expected: (logits, node_emb, (edge_idx1, attn1), (edge_idx2, attn2))
        logits, node_emb, att_pair1, att_pair2 = out[0], out[1], out[2], out[3]
        if layer == 1:
            edge_idx, att = att_pair1
        else:
            edge_idx, att = att_pair2
        # att shape: [num_edges, num_heads]
        att_np = att.mean(dim=1).cpu().numpy()  # average over heads -> [num_edges]
        edge_idx_np = edge_idx.cpu().numpy()    # [2, E]
    else:
        raise RuntimeError("Model forward did not return attention pairs. Update model.forward to return attentions.")
    # aggregate incoming edge attention to node importance
    num_nodes = data.x.shape[0]
    node_imp = np.zeros(num_nodes, dtype=float)
    for i, (src, dst) in enumerate(edge_idx_np.T):
        node_imp[dst] += float(att_np[i])
    # normalize to [0,1]
    if node_imp.max() > 0:
        node_imp = node_imp / (node_imp.max() + 1e-12)
    return node_imp, edge_idx_np


def compute_activation_strength(node_emb: torch.Tensor) -> np.ndarray:
    """
    Compute L2-norm (or other norm) of node embeddings as activation strength.
    node_emb: torch.Tensor [N, D]
    returns np.ndarray [N] normalized to [0,1]
    """
    act = node_emb.detach().cpu().norm(p=2, dim=1).numpy()
    # robust normalization using 95th percentile
    upper = np.percentile(act, 95)
    if upper <= 0:
        upper = act.max() if act.max() > 0 else 1.0
    act = np.clip(act / upper, 0.0, 1.0)
    return act


def combined_importance(attn_imp: np.ndarray, act_imp: np.ndarray) -> np.ndarray:
    """
    Combine attention and activation importance into a single score (element-wise product).
    Normalize to [0,1].
    """
    comb = attn_imp * act_imp
    if comb.max() > 0:
        comb = comb / (comb.max() + 1e-12)
    return comb


def visualize_explanations(img_np: np.ndarray,
                           segments: np.ndarray,
                           centroids: List[Tuple[float, float]],
                           edge_idx: np.ndarray,
                           attn_imp: np.ndarray,
                           act_imp: np.ndarray,
                           top_k: int = 5,
                           out_path: str = None):
    """
    Creates a 2x2 figure:
      1) Original image with superpixel boundaries
      2) Attention importance heatmap (plus edges)
      3) Activation strength heatmap (plus edges)
      4) Combined importance heatmap with top-k highlighted nodes

    img_np: HxWx3 float in [0,1]
    segments: HxW segmentation map (labels starting at 0)
    centroids: list of (row, col)
    edge_idx: [2, E] numpy
    attn_imp: [N]
    act_imp: [N]
    """
    comb = combined_importance(attn_imp, act_imp)
    top_nodes = np.argsort(comb)[-top_k:]

    fig, axes = plt.subplots(2,2, figsize=(10,10))
    ax = axes[0,0]
    ax.imshow(mark_boundaries(img_np.copy(), segments))
    ax.set_title("Image + Superpixel Boundaries")
    ax.axis('off')

    # attention map
    ax = axes[0,1]
    act_map = map_node_values_to_segments(segments, attn_imp)
    ax.imshow(img_np)
    ax.imshow(act_map, cmap='plasma', alpha=0.6)
    # draw edges
    for (s,d) in edge_idx.T:
        y0,x0 = centroids[s]; y1,x1 = centroids[d]
        ax.plot([x0,x1],[y0,y1], color='white', alpha=0.3, linewidth=0.6)
    ax.scatter([c[1] for c in centroids], [c[0] for c in centroids],
               c=attn_imp, cmap='plasma', s=60, edgecolors='black')
    ax.set_title("Attention Importance")
    ax.axis('off')

    # activation map
    ax = axes[1,0]
    act_map2 = map_node_values_to_segments(segments, act_imp)
    ax.imshow(img_np)
    ax.imshow(act_map2, cmap='viridis', alpha=0.6)
    for (s,d) in edge_idx.T:
        y0,x0 = centroids[s]; y1,x1 = centroids[d]
        ax.plot([x0,x1],[y0,y1], color='white', alpha=0.3, linewidth=0.6)
    ax.scatter([c[1] for c in centroids], [c[0] for c in centroids],
               c=act_imp, cmap='viridis', s=60, edgecolors='black')
    ax.set_title("Activation Strength")
    ax.axis('off')

    # combined map with top_k highlighted
    ax = axes[1,1]
    comb_map = map_node_values_to_segments(segments, comb)
    ax.imshow(img_np)
    ax.imshow(comb_map, cmap='inferno', alpha=0.6)
    for (s,d) in edge_idx.T:
        y0,x0 = centroids[s]; y1,x1 = centroids[d]
        ax.plot([x0,x1],[y0,y1], color='white', alpha=0.25, linewidth=0.5)
    sc = ax.scatter([c[1] for c in centroids], [c[0] for c in centroids],
                    c=comb, cmap='inferno', s=60, edgecolors='black')
    for i in top_nodes:
        if i < len(centroids):
            y,x = centroids[i]
            ax.scatter(x, y, s=180, facecolors='none', edgecolors='cyan', linewidths=2)
    ax.set_title("Combined Importance (top nodes highlighted)")
    ax.axis('off')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.show()


def insertion_deletion_test(model, data: Data, device='cpu', metric_fn=None, mode='deletion', steps=20):
    """
    Simple insertion/deletion faithfulness test.

    mode: 'deletion' or 'insertion'
    metric_fn: function(pred_probs)->float, returns score (e.g. probability of true class)
    Procedure:
      - Rank superpixels by importance (combined importance inside)
      - For deletion: progressively remove top k superpixels (set to baseline color), measure metric_fn
      - For insertion: start with blank image and progressively add top k segments
    Returns:
      x_axis (fractions removed/added), y_axis (metric values)
    NOTE: this is image-level and requires ability to re-run model on modified images (we assume you can build graphs from modified images).
    """
    # This is left as a high-level helper; actual implementation depends on your pipeline for rebuilding graphs from modified images.
    raise NotImplementedError("Insertion/deletion requires image->graph rebuild utility. Use this as a template.")
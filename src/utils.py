# src/utils.py
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

def map_node_values_to_segments(segments, node_values):
    """Create per-pixel map by mapping each node id to node_values[node_id].
    If segments have labels not present in node_values, they stay zero.
    """
    out = np.zeros_like(segments, dtype=float)
    n_nodes = len(node_values)
    labels = np.unique(segments)
    for lab in labels:
        if lab < n_nodes:
            out[segments == lab] = node_values[lab]
    return out

def plot_highlighted_superpixels(img_np, segments, highlight_mask=None, title="Highlighted"):
    """
    img_np: HxWx3 numpy (0..1 float)
    highlight_mask: boolean mask same shape as segments marking pixels to tint
    """
    img = img_np.copy()
    if highlight_mask is not None:
        img[highlight_mask] = np.clip(img[highlight_mask] * 0.4 + np.array([1.0, 0.3, 0.3])*0.6, 0, 1)
    plt.figure(figsize=(4,4))
    plt.imshow(mark_boundaries(img, segments))
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_activation_graph(img_np, segments, centroids, edge_idx, node_vals, top_nodes=None, cmap='viridis'):
    plt.figure(figsize=(4,4))
    plt.imshow(img_np)
    # overlay heat
    act_map = map_node_values_to_segments(segments, node_vals)
    plt.imshow(act_map, cmap=cmap, alpha=0.6)
    # draw edges
    for (src, dst) in edge_idx.T:
        y0,x0 = centroids[src]
        y1,x1 = centroids[dst]
        plt.plot([x0, x1], [y0, y1], color='white', alpha=0.4, linewidth=0.6)
    sc = plt.scatter([c[1] for c in centroids], [c[0] for c in centroids],
                     c=node_vals, cmap=cmap, s=50, edgecolors='black')
    if top_nodes is not None:
        for i in top_nodes:
            if i < len(centroids):
                y,x = centroids[i]
                plt.scatter(x, y, s=160, facecolors='none', edgecolors='red', linewidths=2)
    plt.colorbar(sc, label='value')
    plt.axis('off')
    plt.show()
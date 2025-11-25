# src/explain/explain_inference.py

import torch
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import datasets, transforms

from src.models.gnn_classifier import GATImageClassifier
from torch_geometric.loader import DataLoader


def load_graph(graphs_root, idx):
    files = sorted(os.listdir(graphs_root))
    path = os.path.join(graphs_root, files[idx])
    return torch.load(path)


def normalize_attention(att, num_nodes):
    """Convert edge attentions into node importance values"""
    node_scores = torch.zeros(num_nodes)

    src, dst = att[0]
    weights = att[1]

    for i in range(len(weights)):
        node_scores[src[i]] += weights[i]
        node_scores[dst[i]] += weights[i]

    node_scores = node_scores / node_scores.max()
    return node_scores.cpu().numpy()


def visualize_superpixel_heatmap(image_tensor, segments, node_scores, save_path):
    img = image_tensor.permute(1, 2, 0).cpu().numpy()

    heatmap = np.zeros_like(img[:, :, 0], dtype=float)

    for sp in np.unique(segments):
        mask = segments == sp
        heatmap[mask] = node_scores[sp]

    heatmap_color = cm.jet(heatmap)[:, :, :3]  # RGB

    blended = 0.55 * img + 0.45 * heatmap_color

    plt.figure(figsize=(5,5))
    plt.imshow(blended)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--graph_idx", type=int, default=0)
    parser.add_argument("--graphs_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./results/explain")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load graph ----------
    graph = load_graph(args.graphs_root, args.graph_idx).to(device)
    in_dim = graph.x.shape[1]

    # ---------- Load model ----------
    model = GATImageClassifier(in_dim=in_dim, hidden=256, heads=4, num_classes=100)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # ---------- Forward pass ----------
    logits, node_emb, (att1, att2, att3) = model(graph.x, graph.edge_index, graph.batch)
    pred = logits.argmax(dim=1).item()

    print("Prediction:", pred)

    # ---------- Create heatmap ----------
    att = att3  # best layer for explainability
    node_scores = normalize_attention(att, num_nodes=graph.x.shape[0])

    out_path = os.path.join(args.out_dir, f"explain_{args.graph_idx}.png")
    visualize_superpixel_heatmap(graph.original_image, graph.segments, node_scores, out_path)

    print("Saved explanation to:", out_path)


if __name__ == "__main__":
    main()
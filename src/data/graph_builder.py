import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.util import img_as_float
from skimage.color import rgb2gray
import torchvision.transforms as T

def _augment_image_tensor(img_tensor, aug_transforms):
    # img_tensor: torch.Tensor [3,H,W] in [0,1]
    pil_trans = T.ToPILImage()
    to_tensor = T.ToTensor()
    img_pil = pil_trans(img_tensor)
    for t in aug_transforms:
        img_pil = t(img_pil)
    return to_tensor(img_pil)

def image_to_superpixel_graph(img_tensor,
                              feature_map_fn=None,
                              n_segments=100,
                              compactness=10,
                              knn_k=8,
                              use_spatial_edges=True,
                              pca_dim=64):
    """
    img_tensor: torch.Tensor [3, H, W] (0..1)
    feature_map_fn: callable that inputs Bx3xHxW -> BxCxhxw
    Returns PyG Data with x [N,d], edge_index [2,E], segments, centroids
    """

    img_np = img_tensor.permute(1,2,0).cpu().numpy()
    img_np_f = img_as_float(img_np)
    segments = slic(img_np_f, n_segments=n_segments, compactness=compactness, start_label=0)
    labels = np.unique(segments)
    n_nodes = labels.shape[0]

    # Node features using feature_map_fn if provided
    if feature_map_fn is not None:
        with torch.no_grad():
            fmap = feature_map_fn(img_tensor.unsqueeze(0))  # 1xCxhxw
        fmap = fmap.squeeze(0).cpu().numpy()  # C, hf, wf
        C, hf, wf = fmap.shape
        H, W = img_np.shape[:2]
        # upsample by nearest sampling
        ys = (np.linspace(0, hf-1, H)).astype(np.int32)
        xs = (np.linspace(0, wf-1, W)).astype(np.int32)
        up_fmap = fmap[:, ys][:, :, xs]  # C,H,W
        node_feats = np.zeros((n_nodes, C), dtype=np.float32)
        for i, lab in enumerate(labels):
            mask = (segments == lab)
            if mask.sum() == 0:
                node_feats[i] = 0.0
            else:
                vals = up_fmap[:, mask]  # C x Npix
                node_feats[i] = vals.mean(axis=1)
    else:
        H,W,_ = img_np.shape
        node_feats = np.zeros((n_nodes, 3), dtype=np.float32)
        for i, lab in enumerate(labels):
            mask = (segments == lab)
            node_feats[i] = img_np[mask].mean(axis=0)

    # optional PCA to reduce dimension to pca_dim
    if pca_dim is not None and node_feats.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        node_feats = pca.fit_transform(node_feats)

    # Build edges:
    edges = set()
    # spatial adjacency via pixel neighbors
    if use_spatial_edges:
        H, W = segments.shape
        for r in range(H-1):
            for c in range(W-1):
                a = segments[r, c]; b = segments[r, c+1]
                if a != b:
                    edges.add((a,b)); edges.add((b,a))
                a = segments[r, c]; b = segments[r+1, c]
                if a != b:
                    edges.add((a,b)); edges.add((b,a))

    # kNN in feature space using cosine similarity (neighbors in sklearn give distances; we'll use cosine)
    if knn_k > 0 and n_nodes > 1:
        # normalize for cosine
        feats_norm = node_feats / (np.linalg.norm(node_feats, axis=1, keepdims=True) + 1e-9)
        nbrs = NearestNeighbors(n_neighbors=min(knn_k+1, n_nodes), metric='cosine').fit(feats_norm)
        distances, indices = nbrs.kneighbors(feats_norm)
        for i in range(n_nodes):
            for j in indices[i, 1:]:  # skip self
                edges.add((i, int(j))); edges.add((int(j), i))

    if len(edges) == 0:
        for i in range(n_nodes-1):
            edges.add((i, i+1)); edges.add((i+1, i))

    edge_index = np.array(list(edges)).T.astype(np.int64)
    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.original_image = img_tensor.cpu()
    data.segments = segments
    props = regionprops(segments + 1)
    data.centroids = [p.centroid for p in props]
    return data
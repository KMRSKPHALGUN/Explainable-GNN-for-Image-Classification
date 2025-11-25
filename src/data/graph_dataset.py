import os
import torch
from torch_geometric.data import Dataset, Data

class GraphDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.graph_dir = os.path.join(root, split)
        self.files = sorted(os.listdir(self.graph_dir))
        self.transform = transform
        super().__init__()

    def len(self):
        return len(self.files)

    def get(self, idx):
        file_path = os.path.join(self.graph_dir, self.files[idx])
        data = torch.load(file_path, weights_only=False)

        # Ensure it's PyG Data
        assert isinstance(data, Data)

        return data
from torch_geometric.loader import DataLoader
from .graph_dataset import GraphDataset

def get_loaders(root="./graphs", batch_size=32):
    train_data = GraphDataset(root, split="train")
    test_data = GraphDataset(root, split="test")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
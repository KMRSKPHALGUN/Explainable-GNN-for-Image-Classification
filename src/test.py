# test.py
import torch
from torch_geometric.loader import DataLoader
from src.data.loaders import get_loaders
from src.models.gnn_classifier import GATImageClassifier

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            logits = out[0]

            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total

def main():
    graphs_root = "./graphs"   # or your old graph folder if needed
    model_path = "best_model_new_updated_arc.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test loader only
    _, test_loader = get_loaders(graphs_root, batch_size=32)

    # Infer input dimension
    sample = next(iter(test_loader))
    in_dim = sample.x.shape[1]

    # Build model
    model = GATImageClassifier(
        in_dim=in_dim,
        hidden=256,
        heads=4,
        num_classes=100
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate
    test_acc = evaluate(model, test_loader, device)
    print(f"\n Final Test Accuracy: {test_acc * 100:.2f}%\n")

if __name__ == "__main__":
    main()
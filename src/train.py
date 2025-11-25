import argparse, os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_adj
from src.data.loaders import get_loaders
from src.models.gnn_classifier import GATImageClassifier
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device,
                    noise_std=0.02, edge_dropout=0.15):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Train batch"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # ---------------------------
        # 1) Feature noise (regularization)
        # ---------------------------
        batch.x = batch.x + torch.randn_like(batch.x) * noise_std

        # ---------------------------
        # 2) Random edge dropout
        # ---------------------------
        edge_index, _ = dropout_adj(
            batch.edge_index,
            p=edge_dropout,
            force_undirected=True,
            num_nodes=batch.x.size(0)
        )

        # Forward pass
        out = model(batch.x, edge_index.to(device), batch.batch)
        logits = out[0]

        # Loss
        loss = criterion(logits, batch.y)
        loss.backward()

        # ---------------------------
        # 3) Gradient clipping
        # ---------------------------
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        running_loss += loss.item() * batch.y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)

        # no edge dropout during evaluation
        out = model(batch.x, batch.edge_index, batch.batch)
        logits = out[0]

        loss = criterion(logits, batch.y)
        running_loss += loss.item() * batch.y.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_root", type=str, default="./graphs_res50")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graphs
    train_loader, test_loader = get_loaders(args.graphs_root, batch_size=args.batch_size)

    # Infer input dimension
    sample = next(iter(train_loader))
    in_dim = sample.x.shape[1]

    # Build model
    model = GATImageClassifier(
        in_dim=in_dim,
        hidden=256,
        heads=4,
        num_classes=100
    ).to(device)

    # Label smoothing + AdamW + scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            noise_std=0.02,
            edge_dropout=0.15
        )

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"[{epoch}/{args.epochs}] "
              f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} "
              f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} Time={elapsed:.1f}s")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_new_updated_arc.pt")
            print("Saved best model.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    print("Training finished. Best Val Acc:", best_acc)


if __name__ == "__main__":
    main()
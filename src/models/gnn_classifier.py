import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool, GlobalAttention

class ResidualGATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, concat=True, dropout=dropout)
        self.act = nn.ELU()
        self.res_proj = (nn.Linear(in_dim, out_dim) if in_dim != out_dim else None)

    def forward(self, x, edge_index):
        out, att = self.gat(x, edge_index, return_attention_weights=True)
        out = self.act(out)
        if self.res_proj is not None:
            x = self.res_proj(x)
        return out + x, att  # return residual sum and attention info

class GATImageClassifier(nn.Module):
    def __init__(self, in_dim=128, hidden=256, heads=4, num_classes=100, dropout=0.3):
        super().__init__()
        self.block1 = ResidualGATBlock(in_dim, hidden, heads=heads, dropout=0.1)
        self.block2 = ResidualGATBlock(hidden, hidden, heads=heads, dropout=0.1)
        self.block3 = ResidualGATBlock(hidden, hidden, heads=heads, dropout=0.1)

        # Attention-based global pooling
        gate_nn = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
        self.pool = GlobalAttention(gate_nn)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),      # New
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), # New
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )


    def forward(self, x, edge_index, batch):
        x, att1 = self.block1(x, edge_index)
        x, att2 = self.block2(x, edge_index)
        x, att3 = self.block3(x, edge_index)

        graph_emb = self.pool(x, batch)
        logits = self.classifier(graph_emb)

        # Attentions returned: take last two as tuple shapes (edge_idx from GATConv not returned here)
        # For explainability we will call GATConv with return_attention_weights externally if needed.
        # But we also want to return node embeddings for activation maps.
        return logits, x, (None, att1[1] if isinstance(att1, tuple) else None), (None, att2[1] if isinstance(att2, tuple) else None)
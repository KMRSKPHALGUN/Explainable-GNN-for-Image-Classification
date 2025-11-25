import torch

g = torch.load("./graphs/train/0.pt", weights_only=False)

print("Nodes:", g.x.shape)
print("Edges:", g.edge_index.shape)
print("Label:", g.y)

print("\nFeature mean:", g.x.mean(dim=0)[:10])
print("Feature std:", g.x.std(dim=0)[:10])
print("Total std:", g.x.std().item())

# from src.models.feature_extractor import FeatureExtractor
# import torchvision.transforms as T
# from PIL import Image
# import torch

# feat = FeatureExtractor("resnet50")
# feat.eval()

# img = Image.open("./data/cifar-100-python/train/0.png")  # or any CIFAR image
# t = T.ToTensor()(img)

# out = feat(t.unsqueeze(0))
# print(out.mean(), out.std())

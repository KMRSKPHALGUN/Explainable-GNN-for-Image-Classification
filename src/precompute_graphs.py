import os
import torch
import argparse
from torchvision import datasets, transforms
from torch_geometric.data import Data
from src.models.feature_extractor import FeatureExtractor
from src.data.graph_builder import image_to_superpixel_graph, _augment_image_tensor
from torchvision import transforms as T
from tqdm import tqdm

def build_and_save_split(split_name, dataset, feature_extractor, out_dir, n_segments, augmentations):
    os.makedirs(out_dir, exist_ok=True)
    aug_transforms = [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop(size=32, scale=(0.8,1.0)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
    ]
    for idx in tqdm(range(len(dataset)), desc=f"Processing {split_name} set"):
        img, label = dataset[idx]  # tensor ToTensor()
        # Save original + augmented copies
        for a in range(augmentations):
            if a == 0:
                img_t = img
            else:
                img_t = _augment_image_tensor(img, aug_transforms)
            try:
                def fmap_fn(x):
                    return feature_extractor(x)
                data = image_to_superpixel_graph(img_t, fmap_fn, n_segments=n_segments, knn_k=8, pca_dim=64)
            except Exception as e:
                print(f"Skipped {idx} aug {a} due to {e}")
                continue
            data.y = torch.tensor([label], dtype=torch.long)
            # Save with idx and augmentation id
            save_path = os.path.join(out_dir, f"{idx}_{a}.pt")
            torch.save(data, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./precomputed_graphs_res50")
    parser.add_argument("--n_segments", type=int, default=80)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--augmentations", type=int, default=1, help="how many augmented variants to save per image (1=original only)")
    args = parser.parse_args()

    transform = transforms.ToTensor()
    train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

    feat = FeatureExtractor(args.backbone, pretrained=not args.no_pretrained)
    feat.eval()

    train_out = os.path.join(args.out_dir, "train")
    test_out  = os.path.join(args.out_dir, "test")

    build_and_save_split("train", train_ds, feat, train_out, args.n_segments, args.augmentations)
    build_and_save_split("test",  test_ds, feat, test_out,  args.n_segments, 1)

    print("Precomputation complete! All graphs saved successfully.")

if __name__ == "__main__":
    main()
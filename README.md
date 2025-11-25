<!-- # Quick Start (Demo Notebook)

Check out [`demo notebook/demo_graph_building.ipynb`](demo notebook/demo_graph_building.ipynb)
for the minimal working example that:

- Loads CIFAR-10 images
- Converts one image to a superpixel graph
- Runs a GAT model forward pass -->

# Explainable GNN for Image Classification (student project)

Paper baseline: /mnt/data/A_Novel_Graph-based_Framework_for_Explainable_Image_Classification_Features_That_Matter.pdf

## Quick start

1. create venv and install deps:
   pip install -r requirements.txt
   # install torch-geometric according to your system: https://pytorch-geometric.readthedocs.io

2. Run training (small prototype):
   python src/train.py configs/cifar_gat_res18.yaml

3. Demo notebook:
   open notebooks/demo_graph_building.ipynb (contains single-image forward + visualization)

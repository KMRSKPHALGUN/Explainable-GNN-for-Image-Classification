# import torch.nn as nn
# from torchvision.models import resnet50, ResNet50_Weights

# class FeatureExtractor(nn.Module):
#     """Return conv feature map from a ResNet backbone.
#     Output: tensor [B, C, Hf, Wf]
#     """
#     def __init__(self, name='resnet50', pretrained=True):
#         super().__init__()
#         if name != 'resnet50':
#             raise ValueError("This file supports resnet50 only for upgraded pipeline.")
#         # Use ResNet50, keep up to last conv (exclude avgpool and fc)
#         weights = ResNet50_Weights.DEFAULT if pretrained else None
#         res = resnet50(weights=weights)
#         self.backbone = nn.Sequential(*list(res.children())[:-2])  # output [B, 2048, Hf, Wf]

#     def forward(self, x):
#         return self.backbone(x)

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()

        # Select backbone
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            out_dim = 512
        elif backbone == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            out_dim = 512
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            out_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove classification head
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.out_dim = out_dim

    def forward(self, x):
        """
        Input: B x 3 x H x W
        Output: B x C x h x w
        """
        feats = self.backbone(x)
        return feats
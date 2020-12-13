import torch
import torch.nn as nn


import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        """
        Load the pretrained ResNet152 and replace fc
        """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.A = nn.Linear(resnet.fc.in_features, emb_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.A(features)
        return features

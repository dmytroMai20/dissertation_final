import torch
from torch import nn
import torch.nn.functional as F
from .util import EqualizedLinear

class MappingMLP(nn.Module):
    def __init__(self, features, num_layers, num_classes=1):
        super().__init__()

        self.label_embed = nn.Embedding(num_classes, features)

        layers =[]
        for i in range(num_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)
        
    def forward(self, z, labels=None):
        z = F.normalize(z, dim=1)
        #class_embedding = self.label_embed(labels)  
        #z = z + class_embedding 
        return self.net(z)
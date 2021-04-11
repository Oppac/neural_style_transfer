from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models

class Vgg19Nst(torch.nn.Module):
    def __init__(self, requires_grad=False, features=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']):
        super().__init__()
        pretrained_vgg19 = models.vgg19(pretrained=True).features

        self.features = features
        self.layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'avg_pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'avg_pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'avg_pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'avg_pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'avg_pool5',
        ]

        self.model = nn.Sequential(OrderedDict([(self.layers[i], pretrained_vgg19[i])
                                    for i in range(len(self.layers))]))

        for i, layer in enumerate(self.model):
            if isinstance(layer, torch.nn.MaxPool2d):
                self.model[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        for param in self.parameters():
            param.requires_grad = False

    def __str__(self):
        for i in range(len(self.layers)):
            name = self.layers[i]
            print(f"{name} : {getattr(self.model, name)}")
        return f"# of layers: {len(self.layers)}"

    def forward(self, x):
        layer_rep = {}
        for i in range(0, len(self.layers)):
            name = self.layers[i]
            layer = getattr(self.model, name)
            if name in self.features:
                layer_rep[name] = layer(x)
            x = layer(x)
        return layer_rep

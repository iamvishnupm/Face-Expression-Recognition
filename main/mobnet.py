# defining custom model to to add custom fully connected layers
# it is done by inheriting the model - nn.Module.

import torch
import torch.nn as nn
from torchvision import models


class CustomModel(nn.Module):

    def __init__(self, num_classes):
        super(CustomModel, self).__init__()

        self.mobnet = self.init_model()
        self.num_classes = num_classes

        self.fc1 = nn.Linear(1280, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.drop3 = nn.Dropout(0.5)

        self.out = nn.Linear(16, self.num_classes)

    def init_model(self):
        model = models.mobilenet_v2(weights="DEFAULT")
        # model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        model.classifier = nn.Identity()
        for layer in list(model.children())[0][:12]:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in list(model.children())[0][12:]:
            for param in layer.parameters():
                param.requires_grad = True

        for layer in list(model.children())[1:]:
            for param in layer.parameters():
                param.requires_grad = True
        return model

    def forward(self, x):

        x = self.mobnet(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.drop3(x)

        x = self.out(x)

        return x

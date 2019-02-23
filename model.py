

from torchvision import datasets, models, transforms
import torch.nn as nn


def buildModel():
    model = models.resnet18(pretrained=True)
    fc_in_size = model.fc.in_features
    model.fc = nn.Linear(fc_in_size, 4)

    return model

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def get_model(num_classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
import torch
import core.checkpoint as checkpoint
from core.config import cfg
from test.test_model import test_model
from model.SENet_model import SENet

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet101
from torchvision.models import resnet50

import torch.nn as nn
import torch.nn.functional as F

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        # Load the pre-trained ResNet101 model
        resnet = resnet50(weights='DEFAULT')
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Add a Global Average Pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x

def setup_pytorch_resnet50():
    """Sets up a PyTorch ResNet101 model."""
    print("=> creating PyTorch ResNet101 model with global average pooling")
    model = ResNet50FeatureExtractor()
    print(model)
    model = model.cuda() if torch.cuda.is_available() else model
    return model

def __main__():
    """Test the model."""
    if cfg.TEST.WEIGHTS == "":
        print("no test weights exist!!")
    else:
        model = setup_pytorch_resnet50()
        test_model(model, cfg.TEST.DATA_DIR, cfg.TEST.DATASET_LIST, cfg.TEST.SCALE_LIST)

if __name__ == "__main__":
    __main__()


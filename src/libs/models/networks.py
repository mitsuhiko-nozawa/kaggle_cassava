import torch
import torch.nn as nn
import torchvision
import timm

class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False, out_size=5):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x


class CustomEfficientNet(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=False, out_size=5):
        super().__init__()
        #self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x

class CustomSEResNext(nn.Module):
    def __init__(self, model_name='seresnext50_32x4d', pretrained=False, out_size=5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_size)
        #n_features = self.model.classifier.in_features
        #self.model.classifier = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x

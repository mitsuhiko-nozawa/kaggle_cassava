import torch
import torch.nn as nn

class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False, out_size=5):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x
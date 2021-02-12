import torch
import torch.nn as nn
import timm
#from vision_transformer_pytorch import VisionTransformer

class ResNext50_32x4d(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False, out_size=5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=False, out_size=5):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x

class seresnext50_32x4d(nn.Module):
    def __init__(self, model_name=None, pretrained=False, out_size=5):
        model_name = self.__class__.__name__
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_size)
        #n_features = self.model.classifier.in_features
        #self.model.classifier = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x


class resnet50d(nn.Module):
    def __init__(self, model_name=None, pretrained=False, out_size=5):
        model_name = self.__class__.__name__
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_size)

    def forward(self, x):
        x = self.model(x)
        return x

class efficientnet_b3(nn.Module):
    def __init__(self, model_name=None, pretrained=False, out_size=5):
        super().__init__()
        model_name = self.__class__.__name__
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x

class efficientnet_b4(nn.Module):
    def __init__(self, model_name=None, pretrained=False, out_size=5):
        super().__init__()
        model_name = self.__class__.__name__
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x

class efficientnet_b5(nn.Module):
    def __init__(self, model_name=None, pretrained=False, out_size=5):
        super().__init__()
        model_name = self.__class__.__name__
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x

class efficientnet_b6(nn.Module):
    def __init__(self, model_name=None, pretrained=False, out_size=5):
        super().__init__()
        model_name = self.__class__.__name__
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x

class efficientnet_b7(nn.Module):
    def __init__(self, model_name=None, pretrained=False, out_size=5):
        super().__init__()
        model_name = self.__class__.__name__
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, out_size)

    def forward(self, x):
        x = self.model(x)
        return x

#class ViT_B_16(nn.Module):
#    def __init__(self, model_name=None, pretrained=False, out_size=5):
#        super().__init__()
#        model_name = self.__class__.__name__
#        model_name = model_name[:3] + '-' + model_name[4:]
#        self.model = VisionTransformer.from_pretrained(model_name, num_classes=out_size) 
#
#    def forward(self, x):
#        x = self.model(x)
#        return x 
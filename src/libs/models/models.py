from .base_model import BaseModel
from .networks import *
from .criterions import *
from utils import seed_everything
from utils_torch import run_training, inference_fn
from torch.nn import CrossEntropyLoss

import os.path as osp
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

class BaseClassifierModel(BaseModel):
    def fit(self, trainloader, validloader):
        optimizer = eval(self.optimizer)(self.model.parameters(), **self.params["optimizer_params"])
        scheduler = eval(self.scheduler)(optimizer, **self.params["scheduler_params"])
        criterion = eval(self.params["criterion"])().to(self.device)

        self.val_preds = run_training(
            model=self.model,
            trainloader=trainloader,
            validloader=validloader,
            epochs=self.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=criterion,
            early_stopping_steps=self.early_stopping_steps,
            verbose=self.verbose,
            device=self.device,
            seed=self.seed,
            fold=self.fold,
            weight_path=self.weight_path
        )
        
    def predict(self, testloader):
        preds = inference_fn(model=self.model, dataloader=testloader, device=self.device)
        return preds

    def read_weight(self):
        fname = f"{self.seed}_{self.fold}.pt"
        self.model.load_state_dict(torch.load( osp.join(self.weight_path, fname) , map_location=self.device), self.device)

    def save_weight(self):
        pass
    
    def perse_params(self):
        self.ROOT = self.params["ROOT"]
        self.WORK_DIR = self.params["WORK_DIR"]
        self.weight_path = osp.join(self.WORK_DIR, "weight")

        self.device = torch.device(self.params["device"] if torch.cuda.is_available() else 'cpu')
        self.epochs = self.params["epochs"]
        self.early_stopping_steps = self.params["early_stopping_steps"]
        self.verbose = self.params["verbose"]
        self.seed = self.params["seed"]
        self.fold = self.params["fold"]
        self.pretrained = self.params["pretrained"]


        self.optimizer = self.params["optimizer"]
        self.scheduler = self.params["scheduler"]
    

class ResNext50_32x4d(BaseClassifierModel):
    def get_model(self, pretrained):
        model = CustomResNext(pretrained=pretrained, out_size=self.params["output_size"])
        model.to(self.device)
        return model

class EfficientNet(BaseClassifierModel):
    def get_model(self, pretrained):
        model = CustomEfficientNet(pretrained=pretrained, out_size=self.params["output_size"])
        model.to(self.device)
        return model

class seresnext50_32x4d(BaseClassifierModel):
    def get_model(self, pretrained):
        model = CustomSEResNext(pretrained=pretrained, out_size=self.params["output_size"])
        model.to(self.device)
        return model






from .base_model import BaseModel
from .networks import *
from .run_utils import run_training, inference_fn
from utils import seed_everything

import os.path as osp
import torch

class CassavaClassifierModel(BaseModel):
    def fit(self, trainloader, validloader):
        self.val_preds = run_training(
            model=self.model,
            trainloader=trainloader,
            validloader=validloader,
            epochs=self.epochs,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            scheduler=self.scheduler,
            scheduler_params=self.scheduler_params,
            loss_tr=self.loss_tr,
            loss_fn=self.loss_fn,
            early_stopping_steps=self.early_stopping_steps,
            verbose=self.verbose,
            device=self.device,
            seed=self.seed,
            fold=self.fold,
            weight_path=self.weight_path,
            do_cutmix=self.params["do_cutmix"],
            do_fmix=self.params["do_fmix"]
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

        self.loss_tr = self.params["loss_tr"]
        self.loss_fn = self.params["loss_fn"]

        self.optimizer = self.params["optimizer"]
        self.optimizer_params = self.params["optimizer_params"]
        self.scheduler = self.params["scheduler"]
        self.scheduler_params = self.params["scheduler_params"]

    def get_model(self, model_name):
        model = eval(model_name)(pretrained=self.pretrained, out_size=self.params["output_size"])
        model.to(self.device)
        return model
    







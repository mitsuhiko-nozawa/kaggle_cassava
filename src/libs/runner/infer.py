import sys, os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .manager import BaseManager
from utils import make_cv, seed_everything
from utils_torch import get_transforms
from dataset import TestDataset
from models import ResNext50_32x4d, EfficientNet

class Infer(BaseManager):
    def __init__(self, params):
        super(Infer, self).__init__(params)
        self.device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
        self.model = params["model"]

        
    def __call__(self):
        print("Inference")
        if self.get("infer_flag"):
            # test image id from sample_submission.csv
            test_df = pd.read_csv(osp.join(self.data_path, "sample_submission.csv"))
            test_dataset = TestDataset(test_df, self.data_path, get_transforms('valid', self.get("val_transform_params")))
            testloader = DataLoader(test_dataset, batch_size=self.get("batch_size"), num_workers=self.get("num_workers"), shuffle=False, pin_memory=True)
            oof_preds = []
            for seed in self.seeds:
                for fold in range(self.get("n_splits")):
                    self.params["seed"] = seed
                    self.params["fold"] = fold
                    self.params["pretrained"] = False
                    model = eval(self.model)(self.params)
                    model.read_weight()
                    preds = model.predict(testloader)
                    pd.DataFrame(preds, columns=[f"pred_{n}" for n in range(self.params["output_size"])]).to_csv(osp.join(self.preds_path, f"pred_{seed}_{fold}.csv"), index=False)
                    oof_preds.append(preds)
            oof_preds = np.mean(np.array(oof_preds), axis=0)
            oof_preds = pd.DataFrame(preds, columns=[f"pred_{n}" for n in range(self.params["output_size"])])
            oof_preds.to_csv(osp.join(self.preds_path, "pred.csv"), index=False)

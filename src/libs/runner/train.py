import sys, os
import os.path as osp
import pandas as pd
import torch

from .manager import BaseManager
from utils import make_cv, seed_everything
from utils_torch import get_transforms
from dataset import TrainDataset, get_dataloader
from models import *


class Train(BaseManager):
    def __init__(self, params):
        super(Train, self).__init__(params)
        self.device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
        self.model = params["model"]

        
    def __call__(self):
        # cvで画像を分ける
        print("Training")
        if self.get("recreate_cv"):
            make_cv(
                train_df=pd.read_csv(osp.join(self.ROOT, "input", self.raw_dirname, "train.csv")),
                cv_type=self.get("cv"),
                out_path=osp.join(self.ROOT, "input", "cv"),
                n_splits=self.get("n_splits"),
                seeds=self.get("seeds"),
            )
        if self.get("train_flag"):
            for seed in self.seeds: # train by seed
                seed_everything(seed)
                cv_df = pd.read_csv(osp.join(self.cv_path, f"{self.get('cv')}_{seed}.csv"))
                for fold in self.get("run_folds"):
                #for fold in range(self.get("n_splits")): # train by fold
                    train_df = cv_df[(cv_df["fold"] != fold) & (cv_df["fold"] != -1)]
                    val_df = cv_df[cv_df["fold"] == fold]
                    self.train(train_df, val_df, seed, fold)

    def train(self, train_df, val_df, seed, fold):
        if self.debug:
            train_df = train_df[:4]
            val_df = val_df[:4]
            self.params["epochs"] = 1
        train_dataset = TrainDataset(train_df, self.data_path, get_transforms('train', self.get("tr_transform_params")))
        val_dataset = TrainDataset(val_df, self.data_path, get_transforms('valid', self.get("val_transform_params")))
        trainloader, validloader = get_dataloader(train_dataset, val_dataset, self.get("batch_size"), self.get("num_workers"))
        self.params["seed"] = seed
        self.params["fold"] = fold
        self.params["pretrained"] = True
        
        model = eval(self.model)(self.params)
        model.fit(trainloader, validloader)
        # valid predict
        val_preds = model.val_preds
        val_preds = pd.DataFrame(val_preds, columns=[f"pred_{n}" for n in range(self.params["output_size"])])
        val_preds.to_csv(osp.join(self.val_preds_path, f"preds_{seed}_{fold}.csv"), index=False)

        #model.save_weight(osp.join(self.weight_path, f"{seed}_{fold}.pkl"))



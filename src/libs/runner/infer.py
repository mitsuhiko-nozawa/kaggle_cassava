import os
import os.path as osp
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .manager import BaseManager
from utils import seed_everything
from dataset import TestDataset, get_transforms
from models import CassavaClassifierModel

class Infer(BaseManager):
    def __call__(self):
        print("Inference")
        if self.get("infer_flag"):
            test_df = pd.DataFrame()
            test_df["image_id"] = list(os.listdir(osp.join(self.data_path, "test_images")))
            if test_df.shape[0] == 1:
                for i in range(7):
                    test_df = test_df.append(test_df)
                test_df = test_df.reset_index(drop=True)

            test_dataset = TestDataset(test_df, self.data_path, get_transforms('valid', self.get("val_transform_params")))
            testloader = DataLoader(test_dataset, batch_size=self.get("batch_size"), num_workers=self.get("num_workers"), shuffle=False, pin_memory=False)
            oof_preds = []
            for seed in self.seeds:
                for fold in range(self.get("n_splits")):
                    self.params["seed"] = seed
                    self.params["fold"] = fold
                    self.params["pretrained"] = False
                    model = CassavaClassifierModel(self.params)
                    model.read_weight()
                    preds = model.predict(testloader)
                    pd.DataFrame(preds, columns=[f"pred_{n}" for n in range(self.params["output_size"])]).to_csv(osp.join(self.preds_path, f"pred_{seed}_{fold}.csv"), index=False)
                    oof_preds.append(preds)
            oof_preds = np.mean(np.array(oof_preds), axis=0)
            oof_preds = pd.DataFrame(preds, columns=[f"pred_{n}" for n in range(self.params["output_size"])])
            oof_preds.to_csv(osp.join(self.preds_path, "pred.csv"), index=False)

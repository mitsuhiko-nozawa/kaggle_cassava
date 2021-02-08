import os
import os.path as osp
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

def make_cv(train_df, cv_type, out_path, n_splits, seeds):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for seed in seeds:
        df = train_df.copy()
        df["fold"] = -1
        kf = eval(cv_type)(shuffle=True, n_splits=n_splits, random_state=seed)
        for fold, (tr_ind, val_ind) in enumerate(kf.split(df, df["label"])):
            df.loc[val_ind, "fold"] = fold
        df.to_csv(osp.join(out_path, f"{cv_type}_{seed}.csv"))

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available(): 
        print("cuda available")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
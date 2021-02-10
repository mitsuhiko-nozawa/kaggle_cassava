import os
import os.path as osp
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_confusion_matrix(cm, classes,save_path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(osp.join(save_path, "confusion_matrix.png"))
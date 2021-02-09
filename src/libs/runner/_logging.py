import numpy as np
import pandas as pd
import os
import os.path as osp
from sklearn.metrics import accuracy_score
from .manager import BaseManager


class Logging(BaseManager):
    def __call__(self):
        print("Logging")
        if self.get("log_flag"):
            if self.get("calc_cv") and not self.debug: self.cv_score, self.cv_scores = self.calc_cv()
            if self.get("make_submission"): self.make_submission()
            if self.get("mlflow"):
                import mlflow
                mlflow.set_tracking_uri(osp.join(self.ROOT, "src",  "mlflow", "mlruns"))
                self.create_mlflow()

    def calc_cv(self):
        preds = []
        cv_scores = []
        train_df = pd.read_csv(osp.join(self.data_path, "train.csv"))
        
        for seed in self.seeds:
            # cvを1ファイルずつに変更
            cv_df = pd.read_csv(osp.join(self.ROOT, "src", "cvs", f"{self.cv_type}_{seed}.csv" ))
            #mask = train_y[cv_feat] != -1
            cv_df["pred"] = np.nan
            cols = [f"pred_{n}" for n in range(self.out_size)]
            for col in cols:
                cv_df[col] = np.nan

            for fold in range(self.n_splits):
                val_preds = pd.read_csv(osp.join(self.val_preds_path, f"preds_{seed}_{fold}.csv"))
                cv_df.loc[cv_df["fold"] == fold, cols] = val_preds[cols].values
                cv_df.loc[cv_df["fold"] == fold, "pred"] =  np.argmax(cv_df[cv_df["fold"] == fold][cols].values, axis=1)
                fold_cv_score = accuracy_score(cv_df.loc[cv_df["fold"] == fold, "label"].values, cv_df.loc[cv_df["fold"] == fold, "pred"].values)
                print(f"fold {fold} | cv : {fold_cv_score}")
            cv_df[cols].to_csv(osp.join(self.val_preds_path, f"oof_preds_{seed}.csv"), index=False) 
            
            cv_score = accuracy_score(cv_df["label"].values, cv_df["pred"].values)
            cv_scores.append(cv_score)
            print(f"seed {seed} | cv : {cv_score}")
            preds.append(cv_df[cols].values.copy()) # copy!!!!!
        preds = np.mean(np.array(preds), axis=0)
        preds = np.argmax(preds, axis=1)
        preds = pd.DataFrame(preds, columns=["pred"])
        preds.to_csv(osp.join(self.val_preds_path, "oof_preds.csv"), index=False)
        try:
            cv_score = accuracy_score(train_df["label"], preds["pred"])
        except:
            cv_score = np.mean(cv_scores)
            print("mean cv")
            
        print(f"final cv : {cv_score}")
        return cv_score, cv_scores

    def make_submission(self):
        preds = pd.read_csv( osp.join(self.preds_path, "pred.csv") )
        test_df = pd.DataFrame()
        test_df["image_id"] = list(os.listdir(osp.join(self.data_path, "test_images")))
        if test_df.shape[0] == 1:
            for i in range(7):
                test_df = test_df.append(test_df)
            test_df = test_df.reset_index(drop=True)
        test_df["label"] = np.argmax(preds.values, axis=1).copy()
        test_df.to_csv(osp.join(self.sub_path, self.get("submission_name")), index=False)



    def create_mlflow(self):
        with mlflow.start_run(run_name=self.get("exp_name")):
            mlflow.log_param("description", self.get("description"))
            mlflow.log_param("model", self.get("model"))
            mlflow.log_param("cv_scores", self.cv_scores)
            mlflow.log_metric("cv_score", self.cv_score)
            mlflow.log_param("image size", self.get("tr_transform_params")["size"])
            mlflow.log_param("seeds", self.get("seeds"))

            #try:
            #    mlflow.log_artifact(self.feature_importances_fname)
            #except:
            #    pass
            #mlflow.log_artifact(self.submission_fname)


    

from abc import ABCMeta, abstractmethod
import os
import os.path as osp


class BaseManager(metaclass=ABCMeta):
    def __init__(self, params):
        self.params = params
        self.ROOT = params["ROOT"]
        self.WORK_DIR = params["WORK_DIR"]
        self.raw_dirname = params["raw_dirname"] # cassava-leaf-disease-classification
        self.data_path = osp.join(self.ROOT, "input", self.raw_dirname)
        self.cv_path = osp.join(self.ROOT, "src", "cvs")
        self.val_preds_path = osp.join(self.WORK_DIR, "val_preds")
        self.preds_path = osp.join(self.WORK_DIR, "preds")
        self.sub_path = self.preds_path
        self.weight_path = osp.join(self.WORK_DIR, "weight")

        self.seeds = params["seeds"]
        self.debug = params["debug"]
        self.cv_type = params["cv"]
        self.out_size = params["output_size"]
        self.n_splits = params["n_splits"]
        self.device = params["device"]
        self.model = params["model"]
        self.classes = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]

        self.env = params["env"]
        if self.env == "kaggle":
            self.to_kaggleEnv()

        if not osp.exists(self.val_preds_path): os.mkdir(self.val_preds_path)
        if not osp.exists(self.weight_path): os.mkdir(self.weight_path)
        if not osp.exists(self.preds_path): os.mkdir(self.preds_path)

    def to_kaggleEnv(self):
        self.val_preds_path = osp.join("/", "kaggle", "working", "val_preds")
        self.preds_path = osp.join("/", "kaggle", "working", "preds")
        self.sub_path = osp.join("/", "kaggle", "working")
        self.data_path = osp.join("/", "kaggle", "input", self.raw_dirname)
        self.cv_path = osp.join(self.ROOT, "cvs")

    def get(self, key):
        try:
            return self.params[key]
        except:
            raise ValueError(f"No such value in params, {key}")
    @abstractmethod
    def __call__(self):
        raise NotImplementedError
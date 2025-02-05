import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import os.path as osp
from .transforms import get_transforms, get_resize_transforms

class TrainDataset(Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform
        self.data_path = data_path
        self.image_size = 512
        self.resize_transforms = get_resize_transforms()
        self.len_transforms = len([1 for t in self.transform])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = osp.join(self.data_path, "train_images", file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            if self.len_transforms <= 5 and image.shape[0] < 512 or image.shape[1] < 512:
                augmented = self.resize_transforms(image=image)
            else:
                augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image, label
    def change_transforms(self):
        print("change transforms")
        self.transforms = get_transforms(phase="valid", params={"size": self.image_size})

class TestDataset(Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.transform = transform
        self.data_path = data_path
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = osp.join(self.data_path, "test_images", file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

def get_dataloader(train_dataset, val_dataset, batch_size, num_workers):
    trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=False, drop_last=False)
    validloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False, drop_last=False)
    return trainloader, validloader
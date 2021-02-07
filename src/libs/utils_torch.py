import time
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os.path as osp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

def run_training(model, trainloader, validloader, epochs, optimizer, scheduler, loss_fn, early_stopping_steps, verbose, device, seed, fold, weight_path):
    early_step = 0
    best_loss = np.inf
    best_epoch = 0
    best_val_preds = -1
    scaler = GradScaler() ## 
    start = time.time()
    t = time.time() - start
    for epoch in range(epochs):
        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, device, scaler)
        #torch.cuda.empty_cache()
        valid_loss, val_preds = valid_fn(model, loss_fn, validloader, device)
        #torch.cuda.empty_cache()
        # scheduler step
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(valid_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        
        if epoch % verbose==0 or epoch==epoch_-1:
            t = time.time() - start
            print(f"EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}, time: {t}")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_val_preds = val_preds
            torch.save(model.state_dict(), osp.join( weight_path,  f"{seed}_{fold}.pt") )
            early_step = 0
            best_epoch = epoch
        
        elif early_stopping_steps != 0:
            early_step += 1
            if (early_step >= early_stopping_steps):
                t = time.time() - start
                print(f"early stopping in iteration {epoch},  : best itaration is {best_epoch}, valid loss is {best_loss}, time: {t}")
                return best_val_preds

    t = time.time() - start       
    print(f"training until max epoch {epochs},  : best itaration is {best_epoch}, valid loss is {best_loss}, time: {t}")
    return best_val_preds

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device, scaler):
    model.train()
    final_loss = 0
    s = time.time()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (images, labels) in pbar:
        images = images.to(device).float()
        labels = labels.to(device).long()
        
        with autocast():
            outputs = model(images) # これをautocastから外すと実験結果が固定されるけど、メモリが増える、512でやるときは必須
            loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            final_loss += loss.item()
        if (i+1) % 2 == 0 or ((i + 1) == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 
        if i % 50 == 0: 
            description = f"iteration {i} | time {time.time() - s:.4f} | avg loss {final_loss / (i+1):.16f}"
            pbar.set_description(description)
    
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    s = time.time()
    model.eval()
    final_loss = 0
    valid_preds = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    with torch.no_grad():
        for i, (images, labels) in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            final_loss += loss.item()
            valid_preds.append(outputs.softmax(1).detach().cpu().numpy())
            if i % 10 == 0: 
                description = f"iteration {i} | time {time.time() - s:.4f} | avg loss {final_loss / (i+1):.16f}"
                pbar.set_description(description)

        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    for images in dataloader:
        images = images.to(device)
        outputs = model(images)
        with torch.no_grad():
            outputs = model(images)
        preds.append(outputs.softmax(1).detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds


def get_transforms(data, params):    
    if data == 'train':
        return Compose([
            #Resize(params["size"], params["size"]),
            RandomResizedCrop(params["size"], params["size"]),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            CenterCrop(params["size"], params["size"], p=1.), #
            Resize(params["size"], params["size"]),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
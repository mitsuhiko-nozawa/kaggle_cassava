import time
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from .criterions import *


def run_training(model, trainloader, validloader, epochs, optimizer, optimizer_params, scheduler, \
    scheduler_params, loss_tr, loss_fn, early_stopping_steps, verbose, device, seed, fold, weight_path):

    loss_tr = eval(loss_tr)()
    loss_fn = eval(loss_fn)()
    optimizer = eval(optimizer)(model.parameters(), **optimizer_params)
    scheduler = eval(scheduler)(optimizer, **scheduler_params)

    early_step = 0
    best_loss = np.inf
    best_epoch = 0
    best_val_preds = -1
    scaler = GradScaler() ## 
    start = time.time()
    for epoch in range(epochs):
        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, device, scaler, epoch)
        valid_loss, val_preds = valid_fn(model, loss_fn, validloader, device, epoch)

        # scheduler step
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(valid_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
                
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

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device, scaler, epoch):
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
        if i % 50 == 0 or (i+1) == len(dataloader): 
            description = f"[train] epoch {epoch} | iteration {i} | time {time.time() - s:.4f} | avg loss {final_loss / (i+1):.6f}"
            pbar.set_description(description)
    
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device, epoch):
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
            if i % 10 == 0 or (i+1) == len(dataloader): 
                description = f"[valid] epoch {epoch} | iteration {i} | time {time.time() - s:.4f} | avg loss {final_loss / (i+1):.6f}"
                pbar.set_description(description)

        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for i, (images) in pbar:
            images = images.to(device)
            outputs = model(images)
            preds.append(outputs.softmax(1).detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds

import time
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from .criterions import *
from .mixing import *
from sklearn.metrics import accuracy_score


def run_training(model, trainloader, validloader, epochs, optimizer, optimizer_params, scheduler, \
    scheduler_params, loss_tr, loss_fn, early_stopping_steps, verbose, device, seed, fold, weight_path, do_cutmix, do_fmix, do_mixup, reduce_transforms):

    loss_tr = eval(loss_tr)()
    loss_fn = eval(loss_fn)()
    optimizer = eval(optimizer)(model.parameters(), **optimizer_params)
    scheduler = eval(scheduler)(optimizer, **scheduler_params)

    early_step = 0
    best_loss = np.inf
    best_acc = -1
    best_epoch = 0
    best_val_preds = -1
    scaler = GradScaler() ## 
    start = time.time()

    for epoch in range(epochs):
        #train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, device, scaler, epoch, do_cutmix, do_fmix, do_mixup)
        valid_loss, val_preds, valid_acc = valid_fn(model, loss_fn, validloader, device, epoch)

        # scheduler step
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(valid_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
                
        # if valid_loss < best_loss:
        if best_acc < valid_acc:
            best_loss = valid_loss
            best_acc = valid_acc
            best_val_preds = val_preds
            torch.save(model.state_dict(), osp.join( weight_path,  f"{seed}_{fold}.pt") )
            early_step = 0
            best_epoch = epoch
        
        elif early_stopping_steps != 0:
            early_step += 1
            if (early_step >= early_stopping_steps):
                t = time.time() - start
                print(f"early stopping in iteration {epoch},  : best itaration is {best_epoch} | valid loss {best_loss:.4f} | valid acc {best_acc:.4f}time: {t:.4f}")
                return best_val_preds

        # reduce transforms
        if reduce_transforms:
            do_cutmix = False
            do_fmix = False
            if early_step > 0:
                trainloader.dataset.change_transforms()

    t = time.time() - start       
    print(f"training until max epoch {epochs},  : best itaration is {best_epoch} | valid loss {best_loss:.4f} | valid acc {best_acc:.4f} time: {t:.4f}")
    return best_val_preds

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device, scaler, epoch, do_cutmix, do_fmix, do_mixup):
    model.train()
    final_loss = 0
    s = time.time()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (images, labels) in pbar:
        ### mix up 
        p = np.random.uniform(0, 1)
        if do_cutmix and do_fmix: 
            if p < 0.25: # cutmix
                images, labels = cutmix(images, labels, alpha=1.)
            elif p < 0.5: # fmix
                img_size = (images.size(2), images.size(3))
                images, labels = fmix(images, labels, alpha=1., decay_power=3., shape=img_size)
            else :
                eyes = torch.eye(5)
                labels = eyes[labels]

        elif do_cutmix and not do_fmix and p < 0.5: # cutmix
            images, labels = cutmix(images, labels, alpha=1.)
        
        elif do_fmix and not do_cutmix and p < 0.5: # fmix
            img_size = (images.size(2), images.size(3))
            images, labels = fmix(images, labels, alpha=1., decay_power=3., shape=img_size)
        
        elif do_mixup and p < 0.5:
            images, labels = mixup(images, labels, alpha=1.0)

        else :
            eyes = torch.eye(5)
            labels = eyes[labels]
        ########
        
        images = images.to(device).float()
        labels = labels.to(device).float()

        with autocast():
            outputs = model(images) # これをautocastから外すと実験結果が固定されるけど、メモリが増える、512でやるときは必須
            loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            final_loss += loss.item() 
            del loss; torch.cuda.empty_cache()
        if (i+1) % 2 == 0 or ((i + 1) == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 
        if i % 10 == 0 or (i+1) == len(dataloader): 
            description = f"[train] epoch {epoch} | iteration {i} | time {time.time() - s:.4f} | avg loss {final_loss / (i+1):.6f}"
            pbar.set_description(description)
        torch.cuda.empty_cache()
    
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device, epoch):
    s = time.time()
    model.eval()
    final_loss = 0
    valid_preds = []
    valid_labels = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    with torch.no_grad():
        for i, (images, labels) in pbar:
            ### to onehot
            eyes = torch.eye(5)
            labels = eyes[labels]
            ###
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            final_loss += loss.item()
            valid_preds.append(outputs.softmax(1).detach().cpu().numpy())
            valid_labels.append(labels.detach().cpu().numpy())
            if i % 10 == 0 or (i+1) == len(dataloader): 
                description = f"[valid] epoch {epoch} | iteration {i} | time {time.time() - s:.4f} | avg loss {final_loss / (i+1):.6f}"
                pbar.set_description(description)

        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    valid_labels = np.concatenate(valid_labels)
    valid_acc = accuracy_score(valid_labels.argmax(axis=1), valid_preds.argmax(axis=1))
    print(f"[valid] epoch {epoch} | acc {valid_acc:.4f}")
    
    return final_loss, valid_preds, valid_acc


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

import torch
import numpy as np
import cv2
import os
from collections import OrderedDict
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR/Img.shape[0])

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr

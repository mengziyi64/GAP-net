import scipy.io as sio
import os 
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch 
import logging
from ssim_torch import ssim

def generate_masks(mask_path, batch_size):
    nC = 28
    mask = scio.loadmat(mask_path)
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch,1)
    Phi_s_batch[Phi_s_batch==0] = 1
    return Phi_batch, Phi_s_batch


def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    max_ = 0
    for i in range(len(scene_list)):                          #10
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "cast" in img_dict:
            img = img_dict['cast']/65536.
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))

    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        #img = img/img.max()
        img[img<0]=0
        test_data[i,:,:,:] = img
        print(i, img.shape, img.max(), img.min())
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data


def torch_psnr(img, ref):      #input [nC,256,256]
    nC = img.shape[0]
    pixel_max = torch.max(ref)
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i,:,:] - ref[i,:,:]) ** 2)
        psnr += 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr/nC

def torch_ssim(img, ref):   #input [nC,256,256]
    return ssim(torch.unsqueeze(img,0), torch.unsqueeze(ref,0))

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def shift(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs
	

def shuffle_crop(train_data, batch_size):
    
    index = np.random.choice(range(len(train_data)), batch_size)
    processed_data = np.zeros((batch_size, 256, 256, 28), dtype=np.float32)
    
    for i in range(batch_size):
        h, w, _ = train_data[index[i]].shape
        x_index = np.random.randint(0, h - 256)
        y_index = np.random.randint(0, w - 256)
        gt_img = train_data[index[i]][x_index:x_index + 256, y_index:y_index + 256, :] 
        rot_angle = random.randint(1,4)
        gt_img = np.rot90(gt_img, rot_angle)
        processed_data[i, :, :, :] = gt_img
    gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
    return gt_batch


def gen_meas_torch(data_batch, Phi_batch, is_training=True):
    [batch_size, nC, H, W] = data_batch.shape
    step = 2
    if is_training is False:
        Phi_batch = (Phi_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()
    gt_batch = torch.zeros(batch_size, nC, H, W+step*(nC-1)).cuda()
    gt_batch[:,:,:,0:W] = data_batch
    gt_shift_batch = shift(gt_batch)
    meas = torch.sum(Phi_batch*gt_shift_batch, 1)
    return meas

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    
    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO) 
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
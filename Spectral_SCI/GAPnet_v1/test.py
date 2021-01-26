from models import GAP_net
from utils import *
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

mask_path = "../Data/mask/mask_rep1x1.mat"
test_path = "../Data/testing_data/" 
last_train =                       
model_save_filename = ''                 
test_data = LoadTest(test_path)
batch_size_test = len(test_data)
Phi_batch, Phi_s_batch = generate_masks(mask_path, batch_size_test)
nC, H, W = 28, 256, 256
model = GAP_net().cuda()

if last_train != 0:
    model = torch.load('./model/' + model_save_filename + '/model_epoch_{}.pth'.format(last_train))    

def test(epoch):
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    test_y = gen_meas_torch(test_gt, Phi_batch, is_training = False)
    begin = time.time()
    with torch.no_grad():
        model_out = model(test_y, Phi_batch, Phi_s_batch)
    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[-1][k,:,:,:], test_gt[k,:,:,:])
        ssim_val = torch_ssim(model_out[-1][k,:,:,:], test_gt[k,:,:,:])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out[-1].detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    print('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(epoch, psnr_mean, ssim_mean, (end - begin)))
    return (pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean)
    
     
def main():
    if model_save_filename == '':
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
    else:
        date_time = model_save_filename
    result_path = 'recon' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(last_train)
    name = result_path + '/' + 'Testing_{}_m1x1rep_{:.2f}_{:.3f}'.format(last_train, psnr_mean, ssim_mean) + '.mat'
    scio.savemat(name, {'truth':truth, 'pred': pred, 'psnr_list':psnr_all, 'ssim_list':ssim_all})
        
if __name__ == '__main__':
    main()  


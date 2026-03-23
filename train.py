# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
from dataset import HyperDatasetTrain, HyperDatasetValid, HyperDatasetTest, HyperDatasetTestFull
from net import CloudPan
import datetime
import itertools
import sys
import time
import cv2
from torch.autograd import Variable
import hdf5storage as hdf5
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import cc, record_loss, show, PSNR_SSIM_cal, cal_decomp_loss, save_logfile, AverageMeter_test_full, Loss_test

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''
network_name = 'PanTCR'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

init_lr = 1e-3*0.45
num_epochs = 700 
batch_size = 4

scale = 4
if_show = True     # whether save the psf and srf images

model_name = 'baseline'
dataset_name = ['PanTCR-GF2']
# Model
if(dataset_name=='WV3'):
    model = nn.DataParallel(CloudPan(8, 16)).to(device)
else:
    model = nn.DataParallel(CloudPan(4, 16)).to(device)

Hyper_train = HyperDatasetTrain(dataset_name)
Hyper_valid = HyperDatasetValid(dataset_name)
Hyper_test = HyperDatasetTest(dataset_name)

datalen = Hyper_train.__len__()
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999))   # optimizer 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

if not os.path.exists('result'):
    os.mkdir('result')
if not os.path.exists(os.path.join('result', (dataset_name))):
    os.mkdir(os.path.join('result', (dataset_name)))
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
if not os.path.exists(os.path.join('result', timestamp+'_'+network_name)):
    save_root = os.path.join('result', (dataset_name), timestamp+'_'+network_name)
    os.mkdir(save_root)
writer = SummaryWriter(log_dir=os.path.join(save_root, 'Tensorboard_SFITNET_fpn:'+dataset_name))
writer.add_text('Training mode: ', timestamp+'_'+network_name)

trainloader = DataLoader(Hyper_train,batch_size=batch_size,shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
validloader = DataLoader(Hyper_valid,batch_size=16,shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
testloader = DataLoader(Hyper_test,batch_size=1,shuffle=False, num_workers=0, pin_memory=False, drop_last=True)

test_dataset = HyperDatasetTestFull(dataset_name)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

loss_csv_train = open(os.path.join("models/pancloud_"+timestamp+'loss.csv'), 'a+')
record_loss(loss_csv_train, 'epoch', 'train_loss', 'valid_loss', 'test_loss', 'psnr', 'ssim')

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''
torch.backends.cudnn.benchmark = True
est_epoch = 0
best_psnr = 20
for epoch in range(num_epochs):
    ''' train '''
    epoch_train_loss, epoch_val_loss, epoch_test_loss, epoch_test_psnr, epoch_test_ssim = [], [], [], [], []
    # ============Epoch Train=============== #
    model.train()
    with tqdm(total=len(trainloader), miniters=1, desc='Training Epoch: [{}/{}]'.format(epoch, num_epochs)) as t:
        criterion = nn.L1Loss().cuda()
        for batch in trainloader:
            gt, lms, pan = Variable(batch[0], requires_grad=False).cuda(), Variable(batch[1]).cuda(), Variable(batch[2]).cuda()
            optimizer.zero_grad()  # fixed
            out = model(lms, pan)
            loss = criterion(out, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch
            loss.backward()  # fixed
            optimizer.step()  # fixed
            t.set_postfix_str("Batch loss: {:.4f}".format(loss.item()))
            t.update()
        scheduler.step()
        train_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
    
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(validloader+testloader), miniters=1, desc='Validing Epoch: [{}/{}]'.format(epoch, num_epochs)) as t:
            for batch in validloader:
                gt, lms, pan = Variable(batch[0], requires_grad=False).cuda(), Variable(batch[1]).cuda(), Variable(batch[2]).cuda()
                out = model(lms, pan)
                loss = criterion(out, gt)
                epoch_val_loss.append(loss.item())
                t.set_postfix_str("Valid loss: {:.4f}".format(loss.item()))
                t.update()

            for batch in testloader:
                gt, lms, pan = Variable(batch[0], requires_grad=False).cuda(), Variable(batch[1]).cuda(), Variable(batch[2]).cuda()
                out = model(lms, pan)
                loss = criterion(out, gt)
                psnr_,ssim_ = PSNR_SSIM_cal(gt, out)
                epoch_test_loss.append(loss.item())
                epoch_test_psnr.append(psnr_)
                epoch_test_ssim.append(ssim_)
                t.set_postfix_str("Test loss {:.4f}".format(loss.item()))
                t.update()

            v_loss = np.nanmean(np.array(epoch_val_loss))
            t_loss = np.nanmean(np.array(epoch_test_loss))
            t_psnr = np.nanmean(np.array(epoch_test_psnr))
            t_ssim = np.nanmean(np.array(epoch_test_ssim))
            # t.set_postfix_str("Avg_valid loss: {:.4f}, Avg_test loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}".format(v_loss,t_loss,t_psnr,t_ssim))
            # t.update()
    print('Train loss: %7f, Validate loss: %.7f, Test loss: %.7f, PSNR: %.4f, SSIM: %.4f'%(train_loss, v_loss, t_loss, t_psnr, t_ssim))
    save_logfile(save_root, epoch, train_loss, v_loss, t_loss, t_psnr, t_ssim)

    writer.add_scalar('train_loss', train_loss, epoch)  # write to tensorboard to check
    writer.add_scalar('valid_loss', v_loss, epoch)  # write to tensorboard to check
    writer.add_scalar('test_loss', t_loss, epoch)  # write to tensorboard to check
    writer.add_scalar('psnr_loss', t_psnr, epoch)  # write to tensorboard to check
    writer.add_scalar('ssim_loss', t_ssim, epoch)  # write to tensorboard to check
    scheduler.step()  
    if optimizer.param_groups[0]['lr'] <= 1e-6:
        optimizer.param_groups[0]['lr'] = 1e-6
    checkpoint = {
        'Model': model.state_dict(),
    }
    if not os.path.exists(os.path.join(save_root, "pth")):
        os.mkdir(os.path.join(save_root, "pth"))
    torch.save(checkpoint, os.path.join(save_root, "pth", model_name+"_epoch:"+str(epoch)+".pth"))
    
    if t_psnr>best_psnr:
        best_psnr = t_psnr
        torch.save(checkpoint, save_root.replace(str(timestamp), "best.pth"))

    


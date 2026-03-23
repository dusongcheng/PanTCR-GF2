import h5py
import torch
import sys
from net import CloudPan
from dataset import HyperDatasetTrain, HyperDatasetValid, HyperDatasetTest
import numpy as np
import scipy.io as sio
import os
import hdf5storage as hdf5
from utils import AverageMeter_valid, Loss_valid
from tqdm import tqdm
import datetime
import time


def load_model(model_arch, model_path, model_var='Model'):
    model_param = torch.load(model_path, weights_only=True)[model_var]
    model_dict = {}
    for k1, k2 in zip(model_arch.state_dict(), model_param):
        model_dict[k1] = model_param[k2]
    model_arch.load_state_dict(model_dict)
    return model_arch.cuda()

def validate(val_loader, model, criterion, save, save_path):
    model.eval()
    losses = AverageMeter_valid()
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    inference_time = 0
    i = 0
    for gt, lms, pan in tqdm(val_loader):
        with torch.no_grad():
            gt, lms, pan = gt.cuda(), lms.cuda(), pan.cuda()
            # gt, lms, pan = gt[:,:,:64,:64], lms[:,:,:64,:64], pan[:,:,:64,:64]
            # if i<23:
            #     lms = lms/1.1
            model.eval()
            start_time = time.time()
            HR_HSI = model(lms, pan)
            print('Inferen time: %6f'%(time.time() - start_time))
            inference_time = inference_time + time.time() - start_time
            print('Mean L1 loss: %6f'%(torch.mean(torch.abs(gt-HR_HSI)).data.cpu().numpy()))
        if save:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            # if not os.path.exists(os.path.join(save_path, timestamp)):
            #     os.mkdir(os.path.join(save_path, timestamp))
            save_img_path = os.path.join(save_path, str(i)+'.mat')
            out = HR_HSI.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float32)
            gt_ = gt.clone().data.permute(0, 2, 3, 1).cpu().detach().numpy()[0, :, :, :].astype(np.float32)
            lms = lms.clone().data.permute(0, 2, 3, 1).cpu().detach().numpy()[0, :, :, :].astype(np.float32)
            print(save_img_path)
            hdf5.write(data=out, path='rec', filename=save_img_path, matlab_compatible=True)
            hdf5.write(data=gt_, path='gt', filename=save_img_path, matlab_compatible=True)
            hdf5.write(data=lms, path='lms', filename=save_img_path, matlab_compatible=True)
            i = i+1
        
        loss = criterion(gt[0,:].cpu().numpy(), np.clip(HR_HSI[0,:].cpu().numpy(),0,1))
        losses.update(loss.data)
    return losses.avg

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_name = './model.pth'
    dataset_name = 'PanTCR-GF2'
    model = CloudPan(4, 16).cuda()
    print('Dataset: %s, Parameters number of model is %d'%(dataset_name, (sum(param.numel() for param in model.parameters()))))
    model = load_model(model, model_name)
    test_dataset = HyperDatasetTest(dataset_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    print('Network name: %s;' %model_name)
    criterion_valid = Loss_valid(4).cuda()
    loss = validate(test_loader, model, criterion_valid)
    print('psnr:        rmse:       ssim:       sam:        ergas:      UIQI:')
    print("%5f,   %5f,   %5f,   %5f,   %5f,   %5f"%(loss[0][2], loss[0][1], loss[0][0], loss[0][4], loss[0][3], loss[0][5]))

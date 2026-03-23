import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os
import hdf5storage as hdf5
import scipy.io as scio
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

base_root = './dataset'

class HyperDatasetTrain(udata.Dataset):
    def __init__(self, dataset_name):
        super(HyperDatasetTrain, self).__init__()
        file_path = os.path.join(base_root, dataset_name, 'train.h5')
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        self.gt = np.array(gt1, dtype=np.float32)
        print(self.gt.shape)
        lms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        self.lms = np.array(lms1, dtype=np.float32)
        pan1 = data['pan'][...]  # Nx1xHxW
        self.pan = np.array(pan1, dtype=np.float32)
        self.gt, self.lms, self.pan = np.transpose(self.gt, [0,2,3,1]), np.transpose(self.lms, [0,2,3,1]), np.transpose(self.pan, [0,2,3,1])

    def __getitem__(self, index):
        gt_ = self.gt[index, :, :, :]
        lms_ = self.lms[index, :, :, :]
        pan_ = self.pan[index, :, :, :]
        
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        for j in range(rotTimes):
            gt_     = np.rot90(gt_)
            lms_    = np.rot90(lms_)
            pan_    = np.rot90(pan_)

        # Random vertical Flip   
        for j in range(vFlip):
            gt_     = np.flip(gt_,axis=1)
            lms_    = np.flip(lms_,axis=1)
            pan_    = np.flip(pan_,axis=1)
    
        # Random Horizontal Flip
        for j in range(hFlip):
            gt_     = np.flip(gt_,axis=0)
            lms_    = np.flip(lms_,axis=0)
            pan_    = np.flip(pan_,axis=0)

        gt_     = torch.Tensor(np.transpose(gt_  ,(2,0,1)).copy())
        lms_    = torch.Tensor(np.transpose(lms_ ,(2,0,1)).copy())
        pan_    = torch.Tensor(np.transpose(pan_ ,(2,0,1)).copy())
        return gt_, lms_, pan_
    
    def __len__(self):
        return self.gt.shape[0]
        return 32
    

class HyperDatasetValid(udata.Dataset):
    def __init__(self, dataset_name):
        super(HyperDatasetValid, self).__init__()
        file_path = os.path.join(base_root, dataset_name, 'valid.h5')
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        self.gt = np.array(gt1, dtype=np.float32)
        print(self.gt.shape)
        lms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        self.lms = np.array(lms1, dtype=np.float32)
        pan1 = data['pan'][...]  # Nx1xHxW
        self.pan = np.array(pan1, dtype=np.float32)
        self.gt, self.lms, self.pan = torch.from_numpy(self.gt), torch.from_numpy(self.lms), torch.from_numpy(self.pan)

    #####必要函数
    def __getitem__(self, index):
        b,c,h,w = self.gt.shape
        gt_ = self.gt[index, :, :, :].float()
        lms_ = self.lms[index, :, :, :].float()
        pan_ = self.pan[index, :, :, :].float()
        return gt_, lms_, pan_
    
    def __len__(self):
        return self.gt.shape[0]
        return 32


class HyperDatasetTest(udata.Dataset):
    def __init__(self, dataset_name):
        super(HyperDatasetTest, self).__init__()
        file_path = os.path.join(base_root, dataset_name, 'test.h5')
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        self.gt = np.array(gt1, dtype=np.float32)
        # self.gt = np.delete(self.gt, )
        print(self.gt.shape)
        lms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        self.lms = np.array(lms1, dtype=np.float32)
        pan1 = data['pan'][...]  # Nx1xHxW
        self.pan = np.array(pan1, dtype=np.float32)
        self.gt, self.lms, self.pan = torch.from_numpy(self.gt), torch.from_numpy(self.lms), torch.from_numpy(self.pan)

    #####必要函数
    def __getitem__(self, index):
        gt_ = self.gt[index, :, :, :].float()
        lms_ = self.lms[index, :, :, :].float()
        pan_ = self.pan[index, :, :, :].float()
        return gt_, lms_, pan_
    
    def __len__(self):
        return self.gt.shape[0]
        return 32

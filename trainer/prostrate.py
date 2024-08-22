import os
import glob
import numpy as np
import torch
import pathlib
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
from monai.transforms import (
    Orientationd, AddChanneld, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
)
from monai.data import Dataset
import h5py
import threading
import torchvision.transforms as transforms


class SliceData(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """


    def __init__(self, root,transforms = None, mode='train', train_test_split = 0.8):

        files = list(pathlib.Path(root).iterdir())
        files = sorted([str(i) for i in files])
        imgs=[]
        self.xfms = transforms
        if mode == "train":
            for filename in files[:int(train_test_split * len(files))]:

                if filename[-3:] == '.h5':

                    imgs.append(filename)
                    
        elif mode == 'test' or 'validation':
            for filename in files[int(train_test_split * len(files)):]:
                if filename[-3:] == '.h5':
                    imgs.append(filename)


        self.examples = []


        for fname in imgs:
            with h5py.File(fname,'r') as hf:
                fsvol = hf['T2']
                num_slices = fsvol.shape[-1]
                self.examples += [(fname, slice) for slice in range(num_slices)]
            

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice = self.examples[i] 
        final_dic ={}
        
        with h5py.File(fname, 'r') as data:
            t2 = torch.from_numpy(data['T2'][0,:,:,:].astype(np.float64))
            adc = torch.from_numpy(data['ADC'][0,:,:,:].astype(np.float64))
            # dwi0 = torch.from_numpy(data['DWI0'][0,:,:,:].astype(np.float64))
            pd = torch.from_numpy(data['PD'][0,:,:,:].astype(np.float64))
            dce_01 = torch.from_numpy(data['DCE_01'][0,:,:,:].astype(np.float64))
            dce_02 = torch.from_numpy(data['DCE_02'][0,:,:,:].astype(np.float64))
            dce_03 = torch.from_numpy(data['DCE_03'][0,:,:,:].astype(np.float64))
            
        dict_ = {"T2": t2,"ADC":adc,
                 "PD": pd, "DCE_01":dce_01,
                "DCE_02": dce_02, "DCE_03":dce_03}
        
        trans_dic = dict_ # self.xmfs(dict_)

        t2 = trans_dic['T2'][None,:,:,slice]
        adc = trans_dic['ADC'][None,:,:,slice]
        # dwi0 = trans_dic['DWI0'][None,:,:,slice]
        pd = trans_dic['PD'][None,:,:,slice]
        dce_01 = trans_dic['DCE_01'][None,:,:,slice]
        dce_02 = trans_dic['DCE_02'][None,:,:,slice]
        dce_03 = trans_dic['DCE_03'][None,:,:,slice]
        
        dce_02_crop = transforms.CenterCrop(60)(dce_02)
        dce_03_crop = transforms.CenterCrop(60)(dce_03)
        
#         data__ = {"T2": torch.from_numpy(t2),"ADC":torch.from_numpy(adc),
#                 "PD": torch.from_numpy(pd), "DCE_01":torch.from_numpy(dce_01),
#                 "DCE_02":torch.from_numpy(dce_02), "DCE_03":torch.from_numpy(dce_03)}
        
        data_lst = {'A':torch.concatenate((t2,adc,pd,dce_01),axis=0),'B':dce_02, 'DX': i } #, 'C':torch.concatenate((dce_02_crop, dce_03_crop),axis=0)}
        
        
        return data_lst
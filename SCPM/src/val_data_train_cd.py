# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import numpy as np
import torch
import os


# --- Validation/test dataset --- #
class ValData_train(data.Dataset):
    def __init__(self,dataset_name,crop_size, val_data_dir):
        super().__init__()

        self.val_data_dir = val_data_dir
        haze_names = sorted(os.listdir(self.val_data_dir + '/T_Simulated/' ))
        gt_names = sorted(os.listdir(self.val_data_dir + '/T_Cmask/' ))
        self.haze_names = haze_names
        self.gt_names = gt_names
        
        # self.data_list=val_list  
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]
        
    def get_images(self, index):
        haze_name = self.haze_names[index] 
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + '/T_Simulated/' + haze_name).convert("RGB")
        gt_img = Image.open(self.val_data_dir + '/T_Cmask/' + gt_name).convert("L")
          
        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Resize((self.size_w, self.size_h))])
        transform_gt = Compose([ToTensor(),Resize((self.size_w, self.size_h))])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)
         
        return haze, gt, self.haze_names[index]

    def __getitem__(self, index):
        res = self.get_images(index)
        return res
    def __len__(self): 
        return len(self.haze_names)

 
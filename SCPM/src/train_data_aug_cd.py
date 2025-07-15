

# --- Imports --- #
import torch.utils.data as data
from PIL import Image,ImageFile
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import imghdr
import random
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self,dataset_name, crop_size, train_data_dir):
        super().__init__()
        self.train_data_dir = train_data_dir
        haze_names = sorted(os.listdir(self.train_data_dir + '/T_Simulated/' )) #cloud images
        gt_names = sorted(os.listdir(self.train_data_dir + '/T_Cmask/' )) # cloud thickness maps

             
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.dataset_name = dataset_name
        self.crop_size = crop_size
        # self.train_data_dir = train_data_dir
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        haze_name = self.haze_names[index]
        
        gt_name = self.gt_names[index]

        # filename1=self.train_data_dir + 'haze/' + haze_name

        #f = open('check_error.txt','w+')
        #check = imghdr.what(filename1)
        #if check != None:
            #print(filename1)
            #f.write(filename1)
 
            #f.write('\n')
 
            #error_images.append(filename1)
 
        haze = Image.open(self.train_data_dir + '/T_Simulated/' + haze_name)
        clear = Image.open(self.train_data_dir + '/T_Cmask/' + gt_name)
        width, height = haze.size
        i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size_w, self.size_h))
        haze = FF.crop(haze, i, j, h, w)
        clear = FF.crop(clear, i, j, h, w)

        haze,gt=self.augData(haze.convert("RGB") ,clear.convert("L") )

        return haze, gt, self.haze_names[index]
    
    def augData(self,data,target):
        #if self.train:
        if 1: 
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __getitem__(self, index):
        res = self.get_images(index)
        return res 
    def __len__(self):
        return len(self.haze_names)


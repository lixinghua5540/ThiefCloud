import os
import random
import torch
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset
from PIL import Image


class MyTrainDataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=512):
        super(MyTrainDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        targetImagePath = os.path.join(self.targetPath, self.inputImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        inputImage = ttf.to_tensor(inputImage)
        targetImage = ttf.to_tensor(targetImage)

        hh, ww = targetImage.shape[1], targetImage.shape[2]

        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)
        aug = random.randint(0, 8)
        #
        input_ = inputImage[:, rr:rr+ps, cc:cc+ps]
        target = targetImage[:, rr:rr+ps, cc:cc+ps]

        if aug == 1:
            input_, target = input_.flip(1), target.flip(1)
        elif aug == 2:
            input_, target = input_.flip(2), target.flip(2)
        elif aug == 3:
            input_, target = torch.rot90(input_, dims=(1, 2)), torch.rot90(target, dims=(1, 2))
        elif aug == 4:
            input_, target = torch.rot90(input_, dims=(1, 2), k=2), torch.rot90(target, dims=(1, 2), k=2)
        elif aug == 5:
            input_, target = torch.rot90(input_, dims=(1, 2), k=3), torch.rot90(target, dims=(1, 2), k=3)
        elif aug == 6:
            input_, target = torch.rot90(input_.flip(1), dims=(1, 2)), torch.rot90(target.flip(1), dims=(1, 2))
        elif aug == 7:
            input_, target = torch.rot90(input_.flip(2), dims=(1, 2)), torch.rot90(target.flip(2), dims=(1, 2))

        return input_, target


class MyValueDataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=64):
        super(MyValueDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        targetImagePath = os.path.join(self.targetPath, self.inputImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        inputImage = ttf.center_crop(inputImage, [ps, ps])
        targetImage = ttf.center_crop(targetImage, [ps, ps])

        input_ = ttf.to_tensor(inputImage)
        target = ttf.to_tensor(targetImage)

        return input_, target


class MyTestDataSet(Dataset):
    def __init__(self, inputPathTest, targetPathTest):
        super(MyTestDataSet, self).__init__()

        self.inputPath = inputPathTest
        self.inputImages = sorted(os.listdir(inputPathTest))
        # self.inputImages.sort(key=lambda x: int(x.split('.')[0]))

        self.targetPath = targetPathTest
        self.targetImages = sorted(os.listdir(targetPathTest))
        # self.targetImages.sort(key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):
        index = index % len(self.inputImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        input_ = ttf.to_tensor(inputImage)
        target = ttf.to_tensor(targetImage)

        return input_, target, self.inputImages[index]
    

class MyTrain2DataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain, inputPathNeg,patch_size=512):
        super(MyTrain2DataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

        self.NegPath = inputPathNeg
        self.NegImages = os.listdir(inputPathNeg)

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        targetImagePath = os.path.join(self.targetPath, self.inputImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        NegImagePath = os.path.join(self.NegPath, self.inputImages[index])
        NegImage = Image.open(NegImagePath).convert('RGB')

        inputImage = ttf.to_tensor(inputImage)
        targetImage = ttf.to_tensor(targetImage)
        NegImage = ttf.to_tensor(NegImage)

        hh, ww = targetImage.shape[1], targetImage.shape[2]

        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)
        aug = random.randint(0, 8)
        #
        input_ = inputImage[:, rr:rr+ps, cc:cc+ps]
        target = targetImage[:, rr:rr+ps, cc:cc+ps]
        neg = NegImage[:, rr:rr+ps, cc:cc+ps]

        if aug == 1:
            input_, target, neg = input_.flip(1), target.flip(1), neg.flip(1)
        elif aug == 2:
            input_, target, neg = input_.flip(2), target.flip(2), neg.flip(2)
        elif aug == 3:
            input_, target, neg = torch.rot90(input_, dims=(1, 2)), torch.rot90(target, dims=(1, 2)), torch.rot90(neg, dims=(1, 2))
        elif aug == 4:
            input_, target, neg = torch.rot90(input_, dims=(1, 2), k=2), torch.rot90(target, dims=(1, 2), k=2), torch.rot90(neg, dims=(1, 2), k=2)
        elif aug == 5:
            input_, target, neg = torch.rot90(input_, dims=(1, 2), k=3), torch.rot90(target, dims=(1, 2), k=3), torch.rot90(neg, dims=(1, 2), k=3)
        elif aug == 6:
            input_, target, neg = torch.rot90(input_.flip(1), dims=(1, 2)), torch.rot90(target.flip(1), dims=(1, 2)), torch.rot90(neg.flip(1), dims=(1, 2))
        elif aug == 7:
            input_, target, neg = torch.rot90(input_.flip(2), dims=(1, 2)), torch.rot90(target.flip(2), dims=(1, 2)), torch.rot90(neg.flip(2), dims=(1, 2))

        return input_, target, neg

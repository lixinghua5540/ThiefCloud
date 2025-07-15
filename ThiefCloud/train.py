import os
import sys
import torch
import argparse
from tqdm import tqdm
from model import ThiefCloud
import torch.optim as optim
import torch.nn.functional as F
from perceptual import LossNetwork
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils import torchPSNR, setseed
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import MyTrainDataSet, MyValueDataSet, MyTestDataSet
from utils import expand2square
from pytorch_msssim import ssim
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def train(args):

    cudnn.benchmark = True
    setseed(args.seed)

    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model if args.cuda else vgg_model
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model).cuda()
    loss_network.eval()
    # model
    model_restoration = ThiefCloud().cuda() if args.cuda else ThiefCloud()

    pa = sum(p.numel() for p in model_restoration.parameters())
    print(rf"pa:{pa}")
    # optimizer
    optimizer = optim.Adam(model_restoration.parameters(), lr=args.lr1)
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch1, eta_min=args.lr_min)
    # training dataset
    path_train_input, path_train_target = args.train_data + '/haze/', args.train_data + '/gt/'
    datasetTrain = MyTrainDataSet(path_train_input, path_train_target, patch_size=args.patch_size_train)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=args.batch_size_train1, shuffle=True,
                             drop_last=True, num_workers=args.num_works, pin_memory=True)
    # validation dataset
    path_val_input, path_val_target = args.val_data + '/haze/', args.val_data + '/gt/'
    datasetValue = MyValueDataSet(path_val_input, path_val_target, patch_size=args.patch_size_val)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=args.batch_size_val1, shuffle=True,
                             drop_last=True, num_workers=args.num_works, pin_memory=True)
    
    scaler = GradScaler()
    best_psnr = 0
    best_epoch = 0
    for epoch in range(args.epoch1):
        model_restoration.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        # train
        for index, (x, y) in enumerate(iters, 0):

            model_restoration.zero_grad()
            optimizer.zero_grad()

            input_train = Variable(x).cuda() if args.cuda else Variable(x)
            target_train = Variable(y).cuda() if args.cuda else Variable(y)

            with autocast(args.autocast):
                if args.only_last:
                    restored_train = model_restoration(input_train, only_last=args.only_last)
                    loss = F.mse_loss(restored_train, target_train) + args.loss_weight1 * loss_network(restored_train, target_train)
                else:
                    restored_train, fake_image_x4, fake_image_x2 = model_restoration(input_train, only_last=args.only_last)
                    loss_2 = 2.4*F.mse_loss(restored_train, target_train) + 0.3*F.mse_loss(fake_image_x4, target_train) + 0.3*F.mse_loss(fake_image_x2, target_train)
                    fake_image_x4 = fake_image_x4.to(dtype=torch.float32).cuda()
                    fake_image_x2 = fake_image_x2.to(dtype=torch.float32).cuda()
                    loss_perpetual = loss_network(restored_train, target_train) + loss_network(fake_image_x4, target_train) + loss_network(fake_image_x2, target_train)
                    loss = loss_2 + args.loss_weight1 * loss_perpetual

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epochLoss += loss.item()
            iters.set_description('Train_1 !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, args.epoch1, loss.item()))

        # validation
        if epoch % args.val_frequency == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for index, (x, y) in enumerate(valueLoader, 0):
                input_val, target_val = (x.cuda(), y.cuda()) if args.cuda else (x, y)
                with torch.no_grad():
                    if args.only_last:
                        restored_val = model_restoration(input_val, only_last=args.only_last)
                    else:
                        restored_val, _, _ = model_restoration(input_val, only_last=args.only_last)
                for restored_val, target_val in zip(restored_val.clamp_(-1, 1), target_val):
                    psnr_val_rgb.append(torchPSNR(restored_val, target_val))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb >= best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save(model_restoration.state_dict(), args.pre_state)
            print("----------------------------------------------------------------------------------------------")
            print("Validation Finished, Current PSNR: {:.4f}, Best PSNR: {:.4f}.".format(psnr_val_rgb, best_psnr))
            print("----------------------------------------------------------------------------------------------")
        scheduler.step()
    print("Training Process Finished ! Best PSNR : {:.4f}, EPOCH : {:.4f}".format(best_psnr, best_epoch))



def train_op(args):

    cudnn.benchmark = True
    setseed(args.seed)

    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model if args.cuda else vgg_model
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model).cuda()
    loss_network.eval()
    # model
    model_restoration = ThiefCloud(swich=True).cuda() if args.cuda else ThiefCloud()
    # optimizer
    optimizer = optim.Adam(model_restoration.parameters(), lr=args.lr2)
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch2, eta_min=args.lr_min)
    # training dataset
    path_train_input, path_train_target = args.train_data + '/haze/', args.train_data + '/gt/'
    datasetTrain = MyTrainDataSet(path_train_input, path_train_target, patch_size=args.patch_size_train)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=args.batch_size_train2, shuffle=True,
                             drop_last=True, num_workers=args.num_works, pin_memory=True)
    # validation dataset
    path_val_input, path_val_target = args.val_data + '/haze/', args.val_data + '/gt/'
    datasetValue = MyValueDataSet(path_val_input, path_val_target, patch_size=args.patch_size_val)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=args.batch_size_val2, shuffle=True,
                             drop_last=True, num_workers=args.num_works, pin_memory=True)
    
    # load pre model
    if os.path.exists(args.pre_state):
        if args.cuda:
            model_restoration.load_state_dict(torch.load(args.pre_state))
        else:
            model_restoration.load_state_dict(torch.load(args.pre_state, map_location=torch.device('cpu')))

    scaler = GradScaler()
    best_psnr = 0
    best_epoch = 0
    for epoch in range(args.epoch2):
        model_restoration.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        # train
        for index, (x, y) in enumerate(iters, 0):

            model_restoration.zero_grad()
            optimizer.zero_grad()

            input_train = Variable(x).cuda() if args.cuda else Variable(x)
            target_train = Variable(y).cuda() if args.cuda else Variable(y)


            with autocast(args.autocast):
                if args.only_last:
                    restored_train = model_restoration(input_train, only_last=args.only_last)
                    loss = F.mse_loss(restored_train, target_train) + args.loss_weight1 * loss_network(restored_train, target_train)

                else:
                    restored_train, fake_image_x4, fake_image_x2 = model_restoration(input_train, only_last=args.only_last)
                    loss_2 = 2.4*F.mse_loss(restored_train, target_train) + 0.3*F.mse_loss(fake_image_x4, target_train) + 0.3*F.mse_loss(fake_image_x2, target_train)
                    fake_image_x4 = fake_image_x4.to(dtype=torch.float32).cuda()
                    fake_image_x2 = fake_image_x2.to(dtype=torch.float32).cuda()
                    loss_perpetual = loss_network(restored_train, target_train) + loss_network(fake_image_x4, target_train) + loss_network(fake_image_x2, target_train)
                    loss = loss_2 + args.loss_weight1 * loss_perpetual

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epochLoss += loss.item()
            iters.set_description('Train_2 !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, args.epoch2, loss.item()))
        # validation
        if epoch % args.val_frequency == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for index, (x, y) in enumerate(valueLoader, 0):
                input_val, target_val = (x.cuda(), y.cuda()) if args.cuda else (x, y)
                with torch.no_grad():
                    if args.only_last:
                        restored_val = model_restoration(input_val, only_last=args.only_last)
                    else:
                        restored_val, _, _ = model_restoration(input_val, only_last=args.only_last)
                for restored_val, target_val in zip(restored_val.clamp_(-1, 1), target_val):
                    psnr_val_rgb.append(torchPSNR(restored_val, target_val))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb >= best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save(model_restoration.state_dict(), args.save_state)
            print("----------------------------------------------------------------------------------------------")
            print("Validation Finished, Current PSNR: {:.4f}, Best PSNR: {:.4f}.".format(psnr_val_rgb, best_psnr))
            print("----------------------------------------------------------------------------------------------")
        scheduler.step()
    print("Training Process Finished ! Best PSNR : {:.4f}, EPOCH : {:.4f}".format(best_psnr, best_epoch))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--epoch1', type=int, default=200)
    parser.add_argument('--epoch2', type=int, default=200)
    parser.add_argument('--batch_size_train1', type=int, default=10)
    parser.add_argument('--batch_size_train2', type=int, default=5)
    parser.add_argument('--batch_size_val1', type=int, default=10)
    parser.add_argument('--batch_size_val2', type=int, default=1)
    parser.add_argument('--patch_size_train', type=int, default=256)
    parser.add_argument('--patch_size_val', type=int, default=256)
    parser.add_argument('--lr1', type=float, default=5e-4)
    parser.add_argument('--lr2', type=float, default=3e-4)
    parser.add_argument('--lr_min', type=float, default=5e-7)
    parser.add_argument('--train_data', type=str, default='/home/zaq/data/RICE/train')
    parser.add_argument('--neg_save', type=str, default='/home/zaq/data/RICE/train/neg')
    parser.add_argument('--val_data', type=str, default='/home/zaq/data/RICE/test')
    parser.add_argument('--resume_state', type=str, default='./RICE_resume.pth')
    parser.add_argument('--pre_state', type=str, default='./RICE__pre.pth')
    parser.add_argument('--save_state', type=str, default='./RICE_best.pth')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--val_frequency', type=int, default=1)
    parser.add_argument('--loss_weight1', type=float, default=0.04)
    parser.add_argument('--only_last', type=bool, default=False)
    parser.add_argument('--autocast', type=bool, default=True)
    parser.add_argument('--num_works', type=int, default=4)
    parser.add_argument('--control', type=bool, default=True)
    args = parser.parse_args()

    if args.control:
        train(args)

    train_op(args)






import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import sys
sys.path[0]="/home/zaq/Dehazing/SCPM"
import utils.scheduler

from tqdm import tqdm
import network
import utils
import random
import argparse
import numpy as np
from torch.utils import data
# from .datasets import gf1Segmentation
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
import time
from src.train_data_aug_cd import TrainData 
from src.val_data_train_cd import ValData_train
from PIL import Image,ImageFile
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def get_argparser():
    parser = argparse.ArgumentParser()

    # Save position
    parser.add_argument("--save_dir", type=str, default='/home/zaq/Dehazing/CDModel/Result/SateHaze20000',
                        help="path to Dataset")

    # Test options
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--ckpt", 
                        default='/home/zaq/Dehazing/CDModel/checkpointsCDNet_V1/latest_CDNet_V1_SateHaze1k_sim2.pth',
                        type=str, help="restore from checkpoint")

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/home/zaq/data/SateHaze1k/test/T_Simulated',
                        help="path to Dataset")
    
    parser.add_argument("--dataset", type=str, default='gf1', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: 2)")
    parser.add_argument("--in_channels", type=int, default=4,
                        help="num input channels (default: None)")
    parser.add_argument("--feature_scale", type=int, default=2,
                        help="feature_scale (default: 2)")

    # DCNet Options
    parser.add_argument("--model", type=str, default='DCNet_L1',
                        choices=['DCNet_L1','DCNet_L12','DCNet_L123','self_contrast',
                                 'FCN','UNet','SegNet','cloudUNet','cloudSegnet'], help='model name')

    # Train Options
    # parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=1000000,
                        help="epoch number (default: 100k)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=250)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='1,4',
                        help="GPU ID")

    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=1,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=10,
                        help="epoch interval for eval (default: 5000)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=16,
                        help='number of samples for visualization (default: 8)')
    return parser
    parser = argparse.ArgumentParser()

    # Save position
    parser.add_argument("--save_dir", type=str, default='./checkpoints/',
                        help="path to Dataset")

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./GF1_datasets/cropsize_321/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='gf1', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=1,
                        help="num classes (default: None)")
    parser.add_argument("--in_channels", type=int, default=3,
                        help="num input channels (default: None)")
    parser.add_argument("--feature_scale", type=int, default=4,
                        help="feature_scale (default: 2)")

    # DCNet Options
    parser.add_argument("--model", type=str, default='DCNet_L1',
                        choices=['DCNet_L1','DCNet_L12','DCNet_L123','self_contrast',
                                 'FCN','UNet','SegNet','cloudUNet','cloudSegnet'], help='model name')

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--dataset_name", action='store_true', default="T_Cloud")
    parser.add_argument('-ts', '--train-size',nargs='+', help='size of train dataset',default=[256,256], type=int)
    parser.add_argument('-vs', '--valid-size',nargs='+', help='size of valid dataset',default=[256,256], type=int)
    parser.add_argument('-t', '--train-dir', help='training set path', default='/home/zaq/data/T_Cloud/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='/home/zaq/data/T_Cloud/val')

    parser.add_argument("--total_itrs", type=int, default=100000,
                        help="epoch number (default: 100k)")
    parser.add_argument("--batch_size", type=int, default=6,
                        help='batch size (default: 4)')
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=250)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='1,4',
                        help="GPU ID")

    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--ckpt", default=None,help="restore from checkpoint")

    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=5000,
                        help="epoch interval for eval (default: 5000)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


# def get_dataset(opts):
    # """ Dataset And Augmentation
    # """
    # # if opts.dataset == 'gf1':
    # #     train_dst = gf1Segmentation(root=opts.data_root,image_set='train_test', transform=None)
    # #     val_dst = gf1Segmentation(root=opts.data_root,image_set='test_test', transform=None)

    # return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            outputs = model(images)
            preds = outputs.detach().squeeze().cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

        score = metrics.get_results()
    return score, ret_samples



def main():
    opts = get_argparser().parse_args()

    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
    print('Save position is %s\n'%(opts.save_dir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s\n" % (device))

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    data_fold = opts.data_root
    # data_gt_fold = "/home/zaq/data/T_Cloud/val/T_Cmask"
    haze_names = sorted(os.listdir(data_fold))
    # gt_names = sorted(os.listdir(data_gt_fold))

    model = network.CDNet_V1(in_channel=3, out_channel=1, dim=24, feature_scale=4, is_batchnorm=True, is_deconv=False)

    checkpoint = torch.load(opts.ckpt, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint["model_state"].items() if (k in model_dict)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    print("Model restored from %s" % opts.ckpt)


    for i in range(len(haze_names)):
        haze_img = Image.open(data_fold + '/' + haze_names[i]).convert("RGB")
        # clear_img = Image.open(data_gt_fold + '/' + gt_names[i]).convert("L")

        transform_haze = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        # gt = transform_haze(clear_img)

        haze = haze.to(device, dtype=torch.float32).unsqueeze(0)
        # gt = gt.to(device, dtype=torch.float32).unsqueeze(0)
        # val_loader = data.DataLoader(haze, batch_size=1, shuffle=False, num_workers=6) 
        outputs = model(haze)

        preds = torch.squeeze(outputs).detach().cpu().numpy()
        # gt = torch.squeeze(gt).detach().cpu().numpy()
        # dis = gt - preds
        preds[preds<0] = 0
        preds = (preds* 255).astype(np.uint8)
        save_file = opts.save_dir + '/' + haze_names[i]
        image = Image.fromarray(preds, mode='L')  # 'L' 模式表示灰度图像
        # 保存为 PNG 文件
        image.save(save_file)


        # preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        print('test:' + haze_names[i] + '\n')



    
            
    

if __name__ == '__main__':
    main()

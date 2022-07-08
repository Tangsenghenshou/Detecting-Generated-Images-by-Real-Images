"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import numpy as np
import os
import argparse
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import scipy.io as sio
from networks.denoising_rgb import DenoiseNet
from dataloaders.data_rgb import get_validation_data
import utils
import cv2
from skimage import img_as_ubyte





parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='the source picture dic',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='the LNP pic dic',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/denoising/sidd_rgb.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)
count = 0


test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
print(len(test_dataset))
model_restoration = DenoiseNet()
utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration=nn.DataParallel(model_restoration)
model_restoration.eval()
with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        # print(data_test[0].shape)
        rgb_noisy = data_test[0].cuda()
        filenames = data_test[1]
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored,0,1)

        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_noisy)):
                denoised_img = img_as_ubyte(rgb_restored[batch])
                cv2.imwrite(args.result_dir + filenames[batch][:-4] + '.png', denoised_img * 255)

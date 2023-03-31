#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/11 18:04
# @Author : XQP
# @File : ser.py

import torch.nn.functional as F
# from senet.baseline import resnet20
from senet.newse152 import se_resnet152
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob
import scipy.io as sio
import torch.hub
import models
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import json
import numpy as np
import openpyxl
import torch
import models
import torch.optim as optim
from PIL import Image
import models
import torch
import torch.optim as optim
import models
from data_loader import get_test_loader, get_train_loader
# from configfa import get_config
from utils import accuracy, AverageMeter, loader_model, loader_model1
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# path1 = "./ckpt/save/400/multi_se5_NO1_e400_fa_ckpt_DML_2_lr001.pth.tar"
path2 = "./ckpt/save/600_aug/multi_se_resnet50_se_resnext101_NO2_e600_aug_fa_model_best_DML.pth.tar"
# path3 = "./ckpt/save/400/multi_se5_NO3_e400_fa_ckpt_DML_2_lr001.pth.tar"
# path4 = "./ckpt/save/400/multi_se5_NO4_e400_fa_ckpt_DML_2_lr001.pth.tar"
# path5 = "./ckpt/save/400/multi_se5_NO5_e400_fa_ckpt_DML_2_lr001.pth.tar"
# gpu_id = "0,1,2"
# kwargs = {'map_location': lambda storage, loc: storage.cuda(gpu_id)}
# # seres50, _, __ = load_GPUS(9, "se_resnet50", path1)
# seres101, a_, b_ = load_GPUS(9, "se_resnext101", path2)


# seres50, _, __ = loader_model(9, "se_resnet50", path1)
seresxt101, a_, b_ = loader_model1(2, "se_resnext101", path2)
# seres152, c_, d_ = loader_model(9, "se_resnet152", path3)
# seres101, e_, f_ = loader_model(9, "se_resnet101", path4)
# seresxt50, g_, h_ = loader_model(9, "se_resnext50", path5)
from senet.newsext101 import se_resnext101_32x4d
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = se_resnext101_32x4d(pretrained=None, num_classes=2)
        checkpoint = torch.load(path2)
        self.model.load_state_dict(checkpoint['model_state'])
        print(self.model)

        def save_output(module, input, output):
            self.buffer = output
            # print(output)
        self.model.avg_pool.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

features_dir = './augfafea'
def main():
    model = Net()
    # model.load_state_dict(torch.load("seresnet50-60a8950a85b2b.pkl"))
    model = model.cuda()
    model.eval()

    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG','png']
    features = []
    files_list = []
    imgs_path = open("./SARS-Cov-all.txt").read().splitlines()
    # x = os.walk(data_dir)
    # for path, d, filelist in x:
    #     for filename in filelist:
    #         file_glob = os.path.join(path, filename)
    #         files_list.extend(glob.glob(file_glob))
    #
    # print(files_list)
    for i, img in enumerate(imgs_path):
        print("%d %s" % (i, img))
    print("")
    use_gpu = torch.cuda.is_available()
    # for x_path in files_list:
    #     print("x_path" + x_path)
    #     file_name = x_path.split('/')[-1]
    #     fx_path = os.path.join(features_dir, file_name + '.txt')
    # print(fx_path)
    # extractor(x_path, fx_path, model, use_gpu)

    # def extractor(img_path, saved_path, net, use_gpu):
    for i, im in enumerate(imgs_path):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
        )

        img = Image.open(im)
        img = img.convert("RGB")
        img = transform(img)
        # img = transform(img)

        print(im)
        print(img.shape)

        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        print(x.shape)

        if use_gpu:
            x = x.cuda()
            model = model.cuda()
        y = model(x).cpu()
        y = torch.squeeze(y)
        y = y.data.numpy()
        print(y.shape)
        # np.savetxt(saved_path, y, delimiter=',')
        feature = np.reshape(y, [1, -1])
        features.append(feature)
    features = np.array(features)
    dic = {'seresnetxt101': features}
    sio.savemat(features_dir + '/seresnetxt101_sars' + '.mat', dic)


if __name__ == '__main__':

    main()



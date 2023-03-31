import models
import torch
import torch.optim as optim
import models
from data_loader import get_test_loader, get_train_loader
from config import get_config
from utils import accuracy, AverageMeter, loader_model
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# resnet50 = models.get_resnet50(2,False,False,1)
# googlenet = models.get_googlenet(2,False,False,1)

class Net(nn.Module):
    def __init__(self, seresnet50, seresnext101):
        super(Net, self).__init__()
        self.conv1 = list(seresnet50.children())[0]
        self.bn1 = list(seresnet50.children())[1]
        self.relu = list(seresnet50.children())[2]
        self.maxpool = list(seresnet50.children())[3]
        self.layer1 = list(seresnet50.children())[4]
        self.layer2 = list(seresnet50.children())[5]
        self.layer3 = list(seresnet50.children())[6]
        self.layer4 = list(seresnet50.children())[7]
        self.avgpool = list(seresnet50.children())[8]

        self.layer0x = list(seresnext101.children())[0]
        self.layer1x = list(seresnext101.children())[1]
        self.layer2x = list(seresnext101.children())[2]
        self.layer3x = list(seresnext101.children())[3]
        self.layer4x = list(seresnext101.children())[4]
        self.avg_poolx = list(seresnext101.children())[5]

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(4096, 2048)
        # self.fc2 = nn.Linear(2048, 1024)
        # self.dropout1 = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(1024, 8)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # print('*'*50)
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        # print('-' * 50)
        # print(x.shape)

        y = self.layer0x(input)
        y = self.layer1x(y)
        y = self.layer2x(y)
        y = self.layer3x(y)
        y = self.layer4x(y)
        y = self.avg_poolx(y)

        y = y.view(y.size(0), -1)
        # print('-' * 50)
        # print(y.shape)
        # 池化层拼接+fc至2048再至9:multi_ronghe_ma_400_model_best_1_2048_9
        output = torch.cat((x, y), 1)
        output = self.dropout(output)
        output = self.fc1(output)
        # output = self.dropout1(output)
        # output = self.fc2(output)

        return output






config, unprased = get_config()
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
test_dataset = get_test_loader(config.predictdata_dir,config.batch_size,config.input_size,config.num_workers)





def getModel():
    path1 = "./ckpt/save/500/multi_se_resnet50_se_resnext101_NO1_e500_fa_model_best_DML.pth.tar"
    path2 = "./ckpt/save/500/multi_se_resnet50_se_resnext101_NO2_e500_fa_model_best_DML.pth.tar"
    seresnet50, _, __ = loader_model(2, "se_resnet50", path1)
    seresnext101, a_, b_ = loader_model(2, "se_resnext101", path2)


    for index, p in enumerate(seresnet50.parameters()):
        p.requires_grad = False

    for index, p in enumerate(seresnext101.parameters()):
        p.requires_grad = False

    model = Net(seresnet50, seresnext101)
    return model

# def se_resnetronghe(num_classes=2, pretrained=False):
#     """Constructs a ResNet-50 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = Net(seresnet50, seresnext101)
#     # model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
#     if pretrained:
#         # model.load_state_dict(load_state_dict_from_url(
#         #     'https://download.pytorch.org/models/resnet50-19c8e357.pth'))
#
#         # model.load_state_dict(model_zoo.load_url(load_state_dict_from_url['resnet50']))
#         model.load_state_dict(load_state_dict_from_url(
#             "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
#     return model

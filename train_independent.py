import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from utils import accuracy, AverageMeter, getWorkBook, loader_model, f1_ronghe, precision_ronghe, recall_ronghe, auc_ronghe, f1, precision, recall, auc
from VGG16 import *
# from test import getModel
from ronghe import getModel
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import time
import shutil

from tqdm import tqdm
#from utils import accuracy, AverageMeter
import models
from tensorboard_logger import configure, log_value


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the MobileNet Model.
 bmnmnbnmmn;'
    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)

        self.num_classes = config.num_classes

        # training params
        self.epochs = config.epochs
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.weight_decay = config.weight_decay
        self.gamma = config.gamma
        self.step_size = config.step_size

        # misc params
        self.use_gpu = config.use_gpu
        self.feature_extract = config.feature_extract
        self.use_pretrained = config.use_pretrained
        self.paramseed = config.paramseed
        self.loss_ce = nn.CrossEntropyLoss()
        self.best_valid_accs = 0.
        self.best_valid_f1s = 0.
        self.best_valid_precisions = 0.
        self.best_valid_recalls = 0.
        self.best_valid_aucs = 0.
        self.model_name = config.save_name
        self.ckpt_dir = './ronghe/500'
        self.logs_dir = config.logs_dir
        self.mywb = getWorkBook()

        self.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # modelpath = '/media/cvnlp/3b670053-8188-42b6-a0aa-7390926a3303/home/cvnlp/LiChuanxiu/实验/googlenet/multi/400/googlenet_multi_e50_lr001_400X_model_best.pth.tar'
        # self.model = getModel()
        # self.model = models.get_googlenet(self.num_classes, self.feature_extract,self.use_pretrained, self.paramseed)
        # m,_,__ = loader_model(8,'googlenet', modelpath)
        # self.model = m
        self.model = getModel()

        if torch.cuda.is_available() and self.config.use_gpu:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model)

        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)
        # self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma, last_epoch=-1)

        print('[*] Number of parameters of one model: {:,}'.format(

            sum([p.data.nelement() for p in self.model.parameters()])))

    def train(self):


        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.epochs):


            self.scheduler.step(epoch)

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]['lr'], )
            )

            # train for 1 epoch
            train_losses, train_accs, train_f1s, train_precisions, train_recalls, train_aucs = self.train_one_epoch(epoch)
            # evaluate on validation set
            valid_losses, valid_accs, valid_f1s, valid_precisions, valid_recalls, valid_aucs = self.validate(epoch)


            is_best = valid_accs.avg > self.best_valid_accs
            msg1 = "model_: train loss: {:.3f} - train acc: {:.3f} - train f1: {:.3f} - train precision: {:.3f} - train recall: {:.3f} - train auc: {:.3f}"
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val f1: {:.3f} - val precision: {:.3f} - val recall: {:.3f} - val auc: {:.3f}"
            if is_best:
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_losses.avg, train_accs.avg, train_f1s.avg, train_precisions.avg, train_recalls.avg, train_aucs.avg, valid_losses.avg, valid_accs.avg, valid_f1s.avg, valid_precisions.avg, valid_recalls.avg, valid_aucs.avg))
            self.record_loss_acc(train_losses.avg, train_accs.avg, train_f1s.avg, train_precisions.avg, train_recalls.avg, train_aucs.avg, valid_losses.avg, valid_accs.avg, valid_f1s.avg, valid_precisions.avg, valid_recalls.avg, valid_aucs.avg)
            self.best_valid_accs = max(valid_accs.avg, self.best_valid_accs)
            self.best_valid_f1s = max(valid_f1s.avg, self.best_valid_f1s)
            self.best_valid_precisions = max(valid_precisions.avg, self.best_valid_precisions)
            self.best_valid_recalls = max(valid_recalls.avg, self.best_valid_recalls)
            self.best_valid_aucs = max(valid_aucs.avg, self.best_valid_aucs)
            self.save_checkpoint(epoch,
                                 {'epoch': epoch + 1,
                                  'model_state': self.model.state_dict(),
                                  'optim_state': self.optimizer.state_dict(),
                                  'best_valid_acc': self.best_valid_accs,
                                  'best_valid_f1': self.best_valid_f1s,
                                  'best_valid_precision': self.best_valid_precisions,
                                  'best_valid_recall': self.best_valid_recalls,
                                  'best_valid_auc': self.best_valid_aucs,
                                  }, is_best
                                 )
            dir = "ronghe/xls/multi_ronghe_se50_sext101-500_with_contrastive.xlsx"
            self.mywb.save(dir)





    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        f1s = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        aucs = AverageMeter()
        self.model.train()

        tic = time.time()


        with tqdm(total=self.num_train) as pbar:
            for i, (images, labels) in enumerate(self.train_loader):
                if self.use_gpu:
                    images, labels = images.cuda(), labels.cuda()
                images, labels = Variable(images), Variable(labels)

                outputs = self.model(images)
                # print("outputs:=============================================================================")
                # print(outputs)
                # print("outputs.argmax(dim=1)================================================================")
                # print(outputs.argmax(dim=1))
                # print("labels===============================================================================")
                # print(labels)
                loss = self.loss_ce(outputs, labels)

                prec = accuracy(outputs, labels)
                # print("prec:================================================================================")
                # print(prec)
                f1_score = 0
                try:
                    f1_score = f1(outputs, labels)
                except ValueError:
                    pass
                # print("f1_score:================================================================================")
                # print(f1_score)
                precision_score = 0
                try:
                    precision_score = precision(outputs, labels)
                except ValueError:
                    pass
                # print("precision===============================================================================")
                # print(precision_score)
                recall_score = 0
                try:
                    recall_score = recall(outputs, labels)
                except ValueError:
                    pass
                # print("recall=================================================================================")
                # print(recall_score)
                auc_score = 0
                try:
                    auc_score = auc_ronghe(outputs, labels)
                except ValueError:
                    pass
                # print("auc=====================================================================================")
                # print(auc_score)

                losses.update(loss.item(), images.size()[0])
                accs.update(prec, images.size()[0])
                f1s.update(f1_score, images.size()[0])
                precisions.update(precision_score, images.size()[0])
                recalls.update(recall_score, images.size()[0])
                aucs.update(auc_score, images.size()[0])

                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - model1_loss: {} - model1_acc: {:.6f} - model1_f1: {:.6f} - model1_precision: {:.6f} - model1_recall: {:.6f} - model1_auc: {:.6f}".format(
                            (toc - tic), losses.avg, accs.avg, f1s.avg, precisions.avg, recalls.avg, aucs.avg
                        )
                    )
                )
                self.batch_size = images.shape[0]

                pbar.update(self.batch_size)

            return losses, accs, f1s, precisions, recalls, aucs


    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()
        f1s = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        aucs = AverageMeter()
        self.model.eval()


        for i, (images, labels) in enumerate(self.valid_loader):
            if self.use_gpu:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            outputs = self.model(images)
            loss = self.loss_ce(outputs, labels)

            prec = accuracy(outputs, labels)
            f1_score = f1_ronghe(outputs, labels)
            precision_score = precision_ronghe(outputs, labels)
            recall_score = recall_ronghe(outputs, labels)
            auc_score = 0
            try:
                auc_score = auc_ronghe(outputs, labels)
            except ValueError:
                pass

            losses.update(loss.item(), images.size()[0])
            accs.update(prec, images.size()[0])
            f1s.update(f1_score, images.size()[0])
            precisions.update(precision_score, images.size()[0])
            recalls.update(recall_score, images.size()[0])
            aucs.update(auc_score, images.size()[0])

        return losses, accs, f1s, precisions, recalls, aucs


    def save_checkpoint(self, i, state, is_best):
        # i = i+50
        # filename = 'googlenet_multi_e50_lr0001_400X_ckpt_'+str(i)+'.pth.tar'
        # ckpt_path = os.path.join(self.ckpt_dir, filename)
        # torch.save(state, ckpt_path)
        #
        # if is_best:
        #     filename = 'googlenet_extention_multi_e50_lr0001_400X_model_best.pth.tar'
        #     shutil.copyfile(
        #         ckpt_path, os.path.join(self.ckpt_dir, filename)
        #     )

          if is_best:
              filename = 'multi_ronghe_se50_sext101_model_best.pth.tar'
              ckpt_path = os.path.join(self.ckpt_dir, filename)
              torch.save(state, ckpt_path)

    def record_loss_acc(self,epoch_trainloss, epoch_trainacc, epoch_trainf1, epoch_trainprecision, epoch_trainrecall, epoch_trainauc, epoch_testloss, epoch_testacc, epoch_testf1, epoch_testprecision, epoch_testrecall, epoch_testauc):
        self.mywb["epoch_trainloss"].append([epoch_trainloss])
        self.mywb["epoch_trainacc"].append([epoch_trainacc])
        self.mywb["epoch_trainf1"].append([epoch_trainf1])
        self.mywb["epoch_trainprecision"].append([epoch_trainprecision])
        self.mywb["epoch_trainrecall"].append([epoch_trainrecall])
        self.mywb["epoch_trainauc"].append([epoch_trainauc])

        self.mywb["epoch_testloss"].append([epoch_testloss])
        self.mywb["epoch_testacc"].append([epoch_testacc])
        self.mywb["epoch_testf1"].append([epoch_testf1])
        self.mywb["epoch_testprecision"].append([epoch_testprecision])
        self.mywb["epoch_testrecall"].append([epoch_testrecall])
        self.mywb["epoch_testauc"].append([epoch_testauc])


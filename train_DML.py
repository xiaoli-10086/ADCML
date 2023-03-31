import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import models
import Contrastive

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import time
import shutil

from tqdm import tqdm
from utils import accuracy, AverageMeter, getWorkBook_DML, f1, precision, recall, auc, roc, at, at_loss
# from tensorboard_logger import configure, log_value



class Trainer(object):


    def __init__(self, config, data_loader):

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
        self.ckpt_dir = './ckpt/save/500'
        self.logs_dir = config.logs_dir
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.resume = config.resume

        self.feature_extract = config.feature_extract
        self.use_pretrained = config.use_pretrained
        self.paramseed = config.paramseed

        self.print_freq = config.print_freq
        self.model_name = config.save_name
        self.model_num = config.model_num
        self.models = []
        self.optimizers = []
        self.schedulers = []
        self.mywb = getWorkBook_DML()


        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.best_valid_accs = [0.] * self.model_num
        self.best_valid_f1s = [0.] * self.model_num
        self.best_valid_precisions = [0.] * self.model_num
        self.best_valid_recalls = [0.] * self.model_num
        self.best_valid_aucs = [0.] * self.model_num

        self.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



        for i in range(self.model_num):
            # build models
            if i == 0:
                model = models.get_se_resnet50(self.num_classes, self.feature_extract, self.use_pretrained, self.paramseed)
            if i == 1:
                model = models.get_se_resnetxt101(self.num_classes, self.feature_extract, self.use_pretrained, self.paramseed)
            # model.cuda()
            # if torch.cuda.device_count() > 1:
            #     model = nn.s(model)
            # model.to(self.device)
            # if self.use_gpu:
            #     model.cuda()
            if torch.cuda.is_available() and self.config.use_gpu:
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = model.cuda()
                model = torch.nn.DataParallel(model)

            self.models.append(model)

            # initialize optimizer and scheduler
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum,
                                       weight_decay=self.weight_decay)

            self.optimizers.append(optimizer)

            # set learning rate decay
            scheduler = optim.lr_scheduler.StepLR(self.optimizers[i], step_size=40, gamma=self.gamma, last_epoch=-1)
            self.schedulers.append(scheduler)

        print('[*] Number of parameters of one model: {:,}'.format(
            sum([p.data.nelement() for p in self.models[0].parameters()])))

    def train(self):


        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.epochs):

            for scheduler in self.schedulers:
                scheduler.step(epoch)

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.optimizers[0].param_groups[0]['lr'], )
            )

            # train for 1 epoch
            train_losses, train_accs, train_f1s, train_precisions, train_recalls, train_aucs = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_losses, valid_accs, valid_f1s, valid_precisions, valid_recalls, valid_aucs = self.validate(epoch)

            for i in range(self.model_num):
                is_best = valid_accs[i].avg > self.best_valid_accs[i]
                msg1 = "model_{:d}: train loss: {:.3f} - train acc: {:.3f} - train f1: {:.3f} - train precision: {:.3f} - train recall: {:.3f} - train auc: {:.3f}"
                msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val f1: {:.3f} - val precision: {:.3f} - val recall: {:.3f} - val auc: {:.3f}"
                if is_best:
                    # self.counter = 0
                    msg2 += " [*]"
                msg = msg1 + msg2
                print(msg.format(i + 1, train_losses[i].avg, train_accs[i].avg, train_f1s[i].avg, train_precisions[i].avg, train_recalls[i].avg, train_aucs[i].avg, valid_losses[i].avg, valid_accs[i].avg, valid_f1s[i].avg, valid_precisions[i].avg, valid_recalls[i].avg, valid_aucs[i].avg))

                # check for improvement
                # if not is_best:
                # self.counter += 1
                # if self.counter > self.train_patience:
                # print("[!] No improvement in a while, stopping training.")
                # return
                self.best_valid_accs[i] = max(valid_accs[i].avg, self.best_valid_accs[i])
                self.best_valid_f1s[i] = max(valid_f1s[i].avg, self.best_valid_accs[i])
                self.best_valid_precisions[i] = max(valid_precisions[i].avg, self.best_valid_accs[i])
                self.best_valid_recalls[i] = max(valid_recalls[i].avg, self.best_valid_accs[i])
                self.best_valid_aucs[i] = max(valid_aucs[i].avg, self.best_valid_accs[i])
                self.save_checkpoint(i,
                                     {'epoch': epoch + 1,
                                      'model_state': self.models[i].module.state_dict() if isinstance(self.models[i], nn.DataParallel) else self.models[i].state_dict(),
                                      'optim_state': self.optimizers[i].state_dict(),
                                      'best_valid_acc': self.best_valid_accs[i],
                                      'best_valid_f1': self.best_valid_f1s[i],
                                      'best_valid_precision': self.best_valid_precisions[i],
                                      'best_valid_recall': self.best_valid_recalls[i],
                                      'best_valid_auc': self.best_valid_aucs[i],
                                      }, is_best
                                     )
            self.record_loss_acc(train_losses, train_accs, train_f1s, train_precisions, train_recalls, train_aucs, valid_losses, valid_accs, valid_f1s, valid_precisions, valid_recalls, valid_aucs)
            dir = "./ckpt/multi_se_resnet50_DML_se_resnext101_e500_with_contrastive.xlsx"
            self.mywb.save(dir)

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = []
        accs = []
        f1s = []
        precisions = []
        recalls = []
        aucs = []

        for i in range(self.model_num):
            self.models[i].train()
            losses.append(AverageMeter())
            accs.append(AverageMeter())
            f1s.append(AverageMeter())
            precisions.append(AverageMeter())
            recalls.append(AverageMeter())
            aucs.append(AverageMeter())

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (images, labels) in enumerate(self.train_loader):
                if self.use_gpu:
                    # images, labels = images.to(self.device1), labels.to(self.device1)
                    images, labels = images.cuda(), labels.cuda()
                images, labels = Variable(images), Variable(labels)

                # print("images:===============================================")
                # print(images)
                # print("labels:===============================================")
                # print(labels)

                # forward pass
                outputs = []
                for model in self.models:
                    outputs.append(model(images))



                for i in range(self.model_num):
                    ce_loss = self.loss_ce(outputs[i], labels)
                    kl_loss = 0
                    contrastive_loss = 0
                    for j in range(self.model_num):
                        if i != j:
                            kl_loss += self.loss_kl(F.log_softmax(outputs[i], dim=1),
                                                    F.softmax(Variable(outputs[j]), dim=1))
                            contrastive_loss = Contrastive.ContrastiveLoss().forward(output1=outputs[i], output2=outputs[j], label=labels)


                            # print("contrastive_loss:==========================================================")
                            # print(contrastive_loss)
                            #
                            # print("kl:==========================================================================")
                            # print(kl_loss)
                    Attation_loss = at_loss(images.float().reshape(1, -1), labels.float().reshape(1, -1))
                    # loss = ce_loss + kl_loss / (self.model_num - 1) + contrastive_loss / 2
                    loss = ce_loss + kl_loss / (self.model_num - 1) + 0.1 * Attation_loss + contrastive_loss
                    # loss = ce_loss + kl_loss / (self.model_num - 1)
                    # print("loss===========================================================")
                    # print(loss)

                    # print("model_num:=======================================")
                    # print(i)
                    # print("outputs train_one_epoch[{}]:===========================================".format(i))
                    # print(outputs[i])
                    # print("ce_loss:=========================================")
                    # print(ce_loss)
                    # print("kl_loss:=========================================")
                    # print(kl_loss)


                    # measure accuracy and record loss
                    prec = accuracy(outputs[i], labels)
                    # print("prec:============================================")
                    # print(prec)

                    f1_score = f1(outputs[i], labels)
                    # print("f1_score:========================================")
                    # print(f1_score)

                    precision_score = precision(outputs[i], labels)
                    # print("precision_score:=================================")
                    # print(precision_score)

                    recall_score = recall(outputs[i], labels)
                    # print("recall_score:====================================")
                    # print(recall_score)

                    auc_score = 0
                    try:
                        auc_score = auc(outputs[i], labels)
                    except ValueError:
                        pass
                    # print("auc_score:=======================================")
                    # print(auc_score)
                    # print("roctrainone{}===============================================".format(i))
                    # fpr, tpr, threshold = roc(outputs[i], labels)
                    # print("fpr:{}".format(fpr))
                    # print("tpr:{}".format(tpr))
                    # print("threshold:{}".format(threshold))


                    losses[i].update(loss.item(), images.size()[0])
                    accs[i].update(prec, images.size()[0])
                    f1s[i].update(f1_score, images.size()[0])
                    precisions[i].update(precision_score, images.size()[0])
                    recalls[i].update(recall_score, images.size()[0])
                    aucs[i].update(auc_score, images.size()[0])


                    # compute gradients and update SGD
                    self.optimizers[i].zero_grad()
                    loss.backward(retain_graph=True)
                    # loss.backward()

                    self.optimizers[i].step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - model1_loss: {:.3f} - model1_acc: {:.3f} - model1_f1: {:.3f} - model1_precision: {:.3f} - model1_recall: {:.3f} - model1_auc: {:.3f}".format(
                            (toc - tic), losses[0].avg, accs[0].avg, f1s[0].avg, precisions[0].avg, recalls[0].avg, aucs[0].avg
                        )
                    )
                )
                self.batch_size = images.shape[0]

                pbar.update(self.batch_size)

            return losses, accs, f1s, precisions, recalls, aucs

    def validate(self, epoch):


        losses = []
        accs = []
        f1s = []
        precisions = []
        recalls = []
        aucs = []
        for i in range(self.model_num):

            self.models[i].eval()
            losses.append(AverageMeter())
            accs.append(AverageMeter())
            f1s.append(AverageMeter())
            precisions.append(AverageMeter())
            recalls.append(AverageMeter())
            aucs.append(AverageMeter())

        for i, (images, labels) in enumerate(self.valid_loader):
            if self.use_gpu:
                # images, labels = images.to(self.device1), labels.to(self.device1)
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            # forward pass
            outputs = []
            for model in self.models:
                outputs.append(model(images))
            for i in range(self.model_num):
                ce_loss = self.loss_ce(outputs[i], labels)
                kl_loss = 0
                for j in range(self.model_num):
                    if i != j:
                        kl_loss += self.loss_kl(F.log_softmax(outputs[i], dim=1),
                                                F.softmax(Variable(outputs[j]), dim=1))
                        contrastive_loss = Contrastive.ContrastiveLoss().forward(output1=outputs[i], output2=outputs[j],
                                                                                 label=labels)
                Attation_loss = at_loss(images.float().reshape(1, -1), labels.float().reshape(1, -1))
                loss = ce_loss + kl_loss / (self.model_num - 1) + contrastive_loss + 0.1 * Attation_loss
                # loss = ce_loss + kl_loss / (self.model_num - 1)

                # print("outputs validate[{}]:===========================================".format(i))
                # print(outputs[i])

                # measure accuracy and record loss
                prec = accuracy(outputs[i], labels)
                # print("prec:============================================")
                # print(prec)

                f1_score = f1(outputs[i], labels)
                # print("f1_score:========================================")
                # print(f1_score)

                precision_score = precision(outputs[i], labels)
                # print("precision_score:=================================")
                # print(precision_score)

                recall_score = recall(outputs[i], labels)
                # print("recall_score:====================================")
                # print(recall_score)

                auc_score = 0
                try:
                    auc_score = auc(outputs[i], labels)
                except ValueError:
                    pass
                # print("auc_score:=======================================")
                # print(auc_score)

                # print("roctrainone{}===============================================".format(i))
                # fpr, tpr, threshold = roc(outputs[i], labels)
                # print("fpr:{}".format(fpr))
                # print("tpr:{}".format(tpr))
                # print("threshold:{}".format(threshold))


                losses[i].update(loss.item(), images.size()[0])
                accs[i].update(prec, images.size()[0])
                f1s[i].update(f1_score, images.size()[0])
                precisions[i].update(precision_score, images.size()[0])
                recalls[i].update(recall_score, images.size()[0])
                aucs[i].update(auc_score, images.size()[0])


        return losses, accs, f1s, precisions, recalls, aucs



    def save_checkpoint(self, i, state, is_best):


        filename = 'multi_se_resnet50_se_resnext101_NO' + str(i + 1) + '_e500_fa_ckpt_DML.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = 'multi_se_resnet50_se_resnext101_NO' + str(i + 1) + '_e500_fa_model_best_DML.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def record_loss_acc(self,trainloss, trainacc, trainf1, trainprecision, trainrecall, trainauc, testloss, testacc, testf1, testprecision, testrecall, testauc):
        self.mywb["model1_trainloss"].append([trainloss[0].avg])
        self.mywb["model1_trainacc"].append([trainacc[0].avg])
        self.mywb["model1_trainf1"].append([trainf1[0].avg])
        self.mywb["model1_trainprecision"].append([trainprecision[0].avg])
        self.mywb["model1_trainrecall"].append([trainrecall[0].avg])
        self.mywb["model1_trainauc"].append([trainauc[0].avg])
        self.mywb["model1_testloss"].append([testloss[0].avg])
        self.mywb["model1_testacc"].append([testacc[0].avg])
        self.mywb["model1_testf1"].append([testf1[0].avg])
        self.mywb["model1_testprecision"].append([testprecision[0].avg])
        self.mywb["model1_testrecall"].append([testrecall[0].avg])
        self.mywb["model1_testauc"].append([testauc[0].avg])

        self.mywb["model2_trainloss"].append([trainloss[1].avg])
        self.mywb["model2_trainacc"].append([trainacc[1].avg])
        self.mywb["model2_trainf1"].append([trainf1[1].avg])
        self.mywb["model2_trainprecision"].append([trainprecision[1].avg])
        self.mywb["model2_trainrecall"].append([trainrecall[1].avg])
        self.mywb["model2_trainauc"].append([trainauc[1].avg])
        self.mywb["model2_testloss"].append([testloss[1].avg])
        self.mywb["model2_testacc"].append([testacc[1].avg])
        self.mywb["model2_testf1"].append([testf1[1].avg])
        self.mywb["model2_testprecision"].append([testprecision[1].avg])
        self.mywb["model2_testrecall"].append([testrecall[1].avg])
        self.mywb["model2_testauc"].append([testauc[1].avg])




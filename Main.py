# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.optim as optim
import prettytable as pt

import models.Network_main as MultiviewNet
from networks import NegativeLogLikelihood
from dataset import GBMDataset
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger
import scipy.io as scio
import argparse


parser = argparse.ArgumentParser()
# dataset_dir   AWA/Features   Caltech101-20   Caltech101-all   NUSWIDEOBJ   Reuters  Flower CIFAR
# data_name   AWA  Caltech20  Caltechall   NUSWIDEOBJ   Reuters  Flower CIFAR
# num_class   50    20   102   31    6     102  10
# num_view    6     6     6    5    5    4  3


parser.add_argument('--basemodel', default='multiviewNet', help='multiviewNet')
parser.add_argument('--num_view', type=int, default=3, help='the number of views')
parser.add_argument('--fea_out', type=int, default=256, help='the dimension of the first linear layer')
parser.add_argument('--fea_com', type=int, default=128, help='the dimension of the combination layer')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

opt = parser.parse_args()
opt.cuda = True

train_dataset = scio.loadmat('GBMList_train')
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__())
train_iter = iter(train_loader)
idx, traindata, target = train_iter.next()
view_list = []
for v in range(len(traindata)):
    temp_size = traindata[v].size()
    view_list.append(temp_size[1])

def train(ini_file):
    ''' Performs training according to .ini file

    :param ini_file: (String) the path of .ini file
    :return best_c_index: the best c-index
    '''
    # reads configuration from .ini file
    config = read_config(ini_file)
    # builds network|criterion|optimizer based on configuration
    model = MultiviewNet.define_MultiViewNet(which_model=opt.basemodel, norm='batch', init_type='normal',
                                         use_gpu=opt.cuda, num_classes=opt.num_classes,
                                         view_list=view_list,
                                         fea_out=opt.fea_out, fea_com=opt.fea_com)
    criterion = NegativeLogLikelihood(config['network']).to(device)
    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])
    # constructs data loaders based on configuration
    train_dataset = scio.loadmat('GBMList_train')
    test_dataset = scio.loadmat('GBMList_test')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())
    # training
    best_c_index = 0
    flag = 0
    for epoch in range(1, config['train']['epochs']+1):
        # adjusts learning rate
        lr = adjust_learning_rate(optimizer, epoch,
                                  config['train']['learning_rate'],
                                  config['train']['lr_decay_rate'])
        # train step
        model.train()
        for X, y, e in train_loader:
            # makes predictions
            risk_pred = model(X)
            train_loss = criterion(risk_pred, y, e, model)
            train_c = c_index(-risk_pred, y, e)
            # updates parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # valid step
        model.eval()
        for X, y, e in test_loader:
            # makes predictions
            with torch.no_grad():
                risk_pred = model(X)
                valid_loss = criterion(risk_pred, y, e, model)
                valid_c = c_index(-risk_pred, y, e)
                if best_c_index < valid_c:
                    best_c_index = valid_c
                    flag = 0
                    # saves the best model
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, os.path.join(models_dir, ini_file.split('\\')[-1]+'.pth'))
                else:
                    flag += 1
                    if flag >= patience:
                        return best_c_index
        # notes that, train loader and valid loader both have one batch!!!
        print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}'.format(
            epoch, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)
    return best_c_index


import os, sys
sys.path.append('/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/')

import torch
import torch.nn as nn
from torch import optim
from torch import backends
import numpy as np

from model.ft_model import Delf_res50_ft
from dataloader import get_ft_dataloader
from utils.compute_accuracy import cal_accuracy
from train_cfg import config

import datetime
import logging

backends.cudnn.fastest = True
backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(cfg):
    current_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
    log_save_path = os.path.join(cfg.log_root, cfg.stage)

    # create new folder or file
    os.makedirs(os.path.join(cfg.ckpt_root, cfg.stage), exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    f = open(log_save_path+'/{}-train.log'.format(current_time), mode='w', encoding='utf-8')
    f.close()

    # logging config
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.INFO,
                        filename=log_save_path+'/{}-train.log'.format(current_time))
    
    # datasets
    print('datasets loading...')
    train_loader, val_loader = get_ft_dataloader(train_folder_root=cfg.train_data_root, 
                                                 val_folder_root=cfg.val_data_root,
                                                 batch_size=cfg.batch_size, 
                                                 num_workers=cfg.num_workers)
    print('datasets loaded.')

    # model config
    model = Delf_res50_ft(pretrained=True, num_classes=cfg.ncls)
    # model.load_state_dict(torch.load("/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/repository/finetune/epoch_4_loss_2.32170.pth"))
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(cfg.ft_ckpt))
    # optimizer config
    if cfg.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # loss config
    criterion = nn.CrossEntropyLoss()

    # lr scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_stepsize, gamma=cfg.lr_decay)

    train_iters_all = len(train_loader)
    print_interval = 4
    top1_sum = 0
    top3_sum = 0
    top5_sum = 0
    for epoch in range(cfg.ft_epochs):
        model.train()
        epoch_loss = []
        for iter, (inputs, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            top1, top3, top5 = cal_accuracy(outputs, labels)
            top1_sum += top1
            top3_sum += top3
            top5_sum += top5
            if (iter+1) % print_interval == 0:
                top1_avg = top1_sum / print_interval
                top3_avg = top3_sum / print_interval
                top5_avg = top5_sum / print_interval
                top1_sum = 0
                top3_sum = 0
                top5_sum = 0
                message = 'epoch: {}/{}, iters: {}/{}, top-1: {:.4f}, top-3: {:.4f}, top-5: {:.4f}, loss: {:.4f}.'\
                        .format(epoch+1, cfg.ft_epochs, iter+1, train_iters_all, top1_avg, top3_avg, top5_avg, loss.item())
                logging.info(message)
                print(message)
        
        lr_scheduler.step()
        # model save
        model_saving_path = os.path.join(cfg.ckpt_root, cfg.stage, 'epoch_{}_loss_{:.5f}.pth'.format(epoch+1, np.array(epoch_loss).mean()))
        torch.save(model.state_dict(), model_saving_path)


if __name__ == '__main__':
    cfg = config()
    train(cfg)




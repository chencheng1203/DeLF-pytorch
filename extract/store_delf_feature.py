import os, sys
sys.path.append('/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/extract')
import torch 
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import h5py
import pickle
import copy
import logging
import datetime
import warnings
warnings.filterwarnings('ignore')

from delf_feature import get_delf_feature
from pca import DelfPCA
from delf_feature import getDelfFeatureFromMultiScale
from folder import ImageFolder
from extract_cfg import extract_config


def store_delf_feature(cfg):
    # logging
    current_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
    log_save_path = os.path.join(cfg.log_root, cfg.stage)
    os.makedirs(log_save_path, exist_ok=True)
    f = open(log_save_path+'/{}-train-{}.log'.format(current_time, cfg.stage), mode='w', encoding='utf-8')
    f.close()
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.INFO,
                        filename=log_save_path+'/{}-train.log'.format(current_time))

    print('loading dataset...')
    dataset = ImageFolder(root=cfg.index_img, transform=transforms.ToTensor())
    dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)
    print('dataset load done.')
    # model construct
    print('model construct...')
    model = get_delf_feature(cfg.kp_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    print('model load done.')

    # pca trained params
    h5file = h5py.File(os.path.join(cfg.pca_saved, 'pca.h5'), 'r')
    pca_mean = copy.deepcopy(h5file['.']['pca_mean'].value)
    pca_var = copy.deepcopy(h5file['.']['pca_vars'].value)
    pca_matrix = copy.deepcopy(h5file['.']['pca_matrix'].value)
    # delf_pca = DelfPCA(pca_n_components=cfg.pca_dims, whitening=True, pca_saved_path=cfg.pca_saved)

    delf_features = []
    print('delf attention feature extracting...')
    pbar = tqdm(dataloader)
    for index, (inputs, _, filename) in enumerate(pbar):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        try:
            delf_feature = getDelfFeatureFromMultiScale(stage='delf', inputs=inputs, model=model, filename=filename, \
                                                pca_matrix=pca_matrix, pca_mean = pca_mean, pca_var=pca_var, pca_dims=cfg.pca_dims, rf=cfg.delf_rf, \
                                                stride=cfg.delf_stride, padding=cfg.delf_padding,topk=cfg.topk, scales=cfg.scales, \
                                                iou_thresh=cfg.iou_thres, attn_thres=cfg.atte_thres)
        except Exception as e:
            print(e)
            delf_feature = None
        if delf_feature != None:
            delf_features.append(delf_feature)
        msg = "image name: {}".format(filename[0])
        pbar.set_description(desc=msg)
        logging.info(msg)
    print('delf features get done.')
    with open(os.path.join(cfg.delf_saved, 'index.delf'), 'wb') as delf_file:
        pickle.dump(delf_features, delf_file, protocol=2)
    print('saved DeLF feature at {}'.format(os.path.join(cfg.delf_saved, 'index.delf')))


def get_final_results(cfg, inputs):
    # model construct
    print('model construct...')
    model = get_delf_feature(cfg.kp_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    print('model load done.')

    # pca trained params
    h5file = h5py.File(os.path.join(cfg.pca_saved, 'pca.h5'), 'r')
    pca_mean = copy.deepcopy(h5file['.']['pca_mean'].value)
    pca_var = copy.deepcopy(h5file['.']['pca_vars'].value)
    pca_matrix = copy.deepcopy(h5file['.']['pca_matrix'].value)
    # delf_pca = DelfPCA(pca_n_components=cfg.pca_dims, whitening=True, pca_saved_path=cfg.pca_saved)

    print('delf feature extracting...')
    delf_feature = getDelfFeatureFromMultiScale(stage='delf', inputs=inputs, model=model, \
                                                pca_matrix=pca_matrix, pca_mean = pca_mean, pca_var=pca_var, pca_dims=cfg.pca_dims, rf=cfg.delf_rf, \
                                                stride=cfg.delf_stride, padding=cfg.delf_padding,topk=cfg.topk, scales=cfg.scales, \
                                                iou_thresh=cfg.iou_thres, attn_thres=cfg.atte_thres)
    print('delf feature extracted.')
    return delf_feature

if __name__ == '__main__':
    cfg = extract_config()
    store_delf_feature(cfg)
        




    






    
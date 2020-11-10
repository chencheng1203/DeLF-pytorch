import os, sys
import torch 
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import logging
import datetime
import warnings
warnings.filterwarnings('ignore')

from delf_feature import get_delf_feature
from pca import DelfPCA
from delf_feature import getDelfFeatureFromMultiScale
from folder import ImageFolder
from extract_cfg import extract_config

def train_pca(cfg):

    # logging
    current_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
    log_save_path = os.path.join(cfg.log_root, cfg.stage)
    os.makedirs(log_save_path, exist_ok=True)
    f = open(log_save_path+'/{}-train-pca.log'.format(current_time), mode='w', encoding='utf-8')
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

    delf_pca = DelfPCA(pca_n_components=cfg.pca_dims, pca_saved_path=cfg.pca_saved)

    feature_maps = []
    print('delf attention feature extracting...')
    features_num = 0
    pbar = tqdm(dataloader)
    for (inputs, _, filename) in pbar:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        try:
            output = getDelfFeatureFromMultiScale(stage='pca', inputs=inputs, model=model, filename=filename, \
                                                pca_matrix=None, pca_var=None, pca_dims=cfg.pca_dims, rf=cfg.delf_rf, \
                                                stride=cfg.delf_stride, padding=cfg.delf_padding,topk=cfg.topk, scales=cfg.scales, \
                                                iou_thresh=cfg.iou_thres, attn_thres=cfg.atte_thres)
        except Exception as e:
            print(e)

            
        descriptor_np_list = output['descriptor_np_list']
        pca_feature = [descriptor_np_list[i,:] for i in range(descriptor_np_list.shape[0])]  # 将ndarray变成列表
        feature_maps.extend(pca_feature)

        curr_fea_nums = descriptor_np_list.shape[0]
        features_num += curr_fea_nums
        msg = "curr feature nums: {}".format(features_num)
        pbar.set_description(desc=msg)
        logging.info(filename[0] + "-feature nums: {}".format(curr_fea_nums))
    
    print('features get done. Total feature nums: {}'.format(features_num))
    print('start train pca...')
    delf_pca(feature_maps)
    print('pca trained done. saved at {}'.format(cfg.pca_saved + 'pca.h5'))


if __name__ == '__main__':
    cfg = extract_config()
    train_pca(cfg)
        




    






    
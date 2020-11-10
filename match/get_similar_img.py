'''
--------------构建delf对应的检索系统---------------
步骤：
1、初始化并载入模型、载入PCA参数、载入特征库数据
2、载入query图像，图像提特征，执行PCA降维
3、降维特征进入特征库查询，返回查询top热度索引值
4、利用KD树进行成对匹配，利用ransac算法实现几何匹配
5、求得inliner对应的得分，找到top1结果

作者：马军福
日期：2020-08-19

'''
import sys
sys.path.append('../')

import torch.nn
from torch.autograd import Variable
import pickle
from collections import Counter
import faiss
import os
import argparse
from PIL import Image
import h5py
import torch
import torchvision.transforms as transforms
import helper.delf_helper as delf_helper
from train.delf import Delf_V1
from io import BytesIO
from helper import matcher
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform
import cv2


def __build_delf_config__(data):
    parser = argparse.ArgumentParser('delf-config')
    parser.add_argument('--stage', type=str, default='inference')
    parser.add_argument('--use_random_gamma_rescale', type=str, default=False)
    parser.add_argument('--arch', type=str, default=data['ARCH'])
    parser.add_argument('--load_from', type=str, default=data['LOAD_FROM'])
    parser.add_argument('--target_layer', type=str, default=data['TARGET_LAYER'])
    delf_config, _ = parser.parse_known_args()

    state = {k: v for k, v in delf_config._get_kwargs()}
    print(state)
    return delf_config

'''
构建一个DelfInferece类，构建对应的模型对象、PCA参数、特征库对象
'''
class DelfInferece():
    def __init__(self,feeder_config):
        # environment setting.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(feeder_config.get('GPU_ID'))

        # parameters.
        self.iou_thres = feeder_config.get('IOU_THRES')
        self.attn_thres = feeder_config.get('ATTN_THRES')
        self.top_k = feeder_config.get('TOP_K')
        self.target_layer = feeder_config.get('TARGET_LAYER')
        self.scale_list = feeder_config.get('SCALE_LIST')
        self.workers = feeder_config.get('WORKERS')
        self.fea_data_path = feeder_config.get('FEATURE_DATA_PATH')

        # load pytorch model
        print('load DeLF pytorch model...')
        delf_config = __build_delf_config__(feeder_config)
        self.model = Delf_V1(
            ncls=None,
            load_from=delf_config.load_from,
            arch=delf_config.arch,
            stage=delf_config.stage,
            target_layer=delf_config.target_layer,
            use_random_gamma_rescale=False)
        self.model.eval()

        if torch.cuda.is_available():
            self.model =  self.model.cuda()

        # load pca matrix
        print('load PCA parameters...')
        h5file = h5py.File(feeder_config.get('PCA_PARAMETERS_PATH'), 'r')
        self.pca_mean = h5file['.']['pca_mean'].value
        self.pca_vars = h5file['.']['pca_vars'].value
        self.pca_matrix = h5file['.']['pca_matrix'].value
        self.pca_dims = feeder_config.get('PCA_DIMS')
        self.use_pca = feeder_config.get('USE_PCA')

        # !!! make sure to use stride=16 for target_layer=='layer3'.
        if self.target_layer in ['layer3']:
            self.fmap_depth = 1024
            self.rf = 291.0
            self.stride = 16.0
            self.padding = 145.0
        else:
            raise ValueError('Unsupported target_layer: {}'.format(self.target_layer))
        #load features database
        self.n_subq = 8  # number of sub-quantizers
        self.n_centroids = 32  # number of centroids for each sub-vector
        self.n_bits = 5  # number of bits for each sub-vector
        self.n_probe = 3  # number of voronoi cell to explore
        self.coarse_quantizer = faiss.IndexFlatL2(self.pca_dims)
        self.pq = faiss.IndexIVFPQ(self.coarse_quantizer, self.pca_dims, self.n_centroids, self.n_subq, self.n_bits)
        self.pq.nprobe = self.n_probe

        print('PQ complete')


    def __transform__(self, image):
        transform = transforms.ToTensor()
        return transform(image)

    def __print_result__(self, data):
        print('----------------------------------------------------------')
        print('filename: ', data['filename'])
        print("location_np_list shape: ", data['location_np_list'].shape)
        print("descriptor_np_list shape: ", data['descriptor_np_list'].shape)
        print("feature_scale_np_list shape: ", data['feature_scale_np_list'].shape)
        print("attention_score_np_list shape: ", data['attention_score_np_list'].shape)
        print("attention_np_list shape: ", data['attention_np_list'].shape)
        print('----------------------------------------------------------')

    def __get_result__(self,path,image):
        # load tensor image
        if torch.cuda.is_available():
            image = (self.__transform__(image)).cuda()

        x = image.unsqueeze(0)
        # extract feature.
        data = delf_helper.GetDelfFeatureFromMultiScale(
            x = x,
            model = self.model,
            filename = path,
            pca_mean = self.pca_mean,
            pca_vars = self.pca_vars,
            pca_matrix = self.pca_matrix,
            pca_dims = self.pca_dims,
            rf = self.rf,
            stride = self.stride,
            padding = self.padding,
            top_k = self.top_k,
            scale_list = self.scale_list,
            iou_thres = self.iou_thres,
            attn_thres = self.attn_thres,
            use_pca = self.use_pca,
            workers = self.workers)
        return data 
    
    #Get similar image  pairs
    def get_sim_img(self,query_des2imglist):
        all_searched_des = []
        for des_i in query_des2imglist.keys():
            all_searched_des.extend(query_des2imglist[des_i])

        imgFreq = Counter(all_searched_des).most_common()
        index, freq = list(zip(*imgFreq))
        result = {'index': index, 'freq':freq}
        return result
        

    def img_query_db(self,img_path):
        img_data = np.array(Image.open(img_path))
        img_data = Image.fromarray(img_data, mode='RGB')
        img_result = self.__get_result__(img_path,img_data)

        #features database search
        k=60
        cur_img_desc_data = img_result['descriptor_np_list']
        cur_img_loc_data = img_result['location_np_list']
        _,query_des2deslist = self.pq.search(cur_img_desc_data,k)
        query_des2imgList = {}

        #
        for img_i,des_list in enumerate(query_des2deslist):
            query_des2imgList[img_i] = [self.img_from_des[des_i] for des_i in des_list]

        result = self.get_sim_img(query_des2imgList)
        hot_img_list = result['index']
        res_db_desc_set = []
        res_db_loc_set = []
        res_db_name_set = []

        for name in hot_img_list:
            cur_db_name_idxs = self.des_from_img[name]
            cur_db_desc_data =self.des_np_set[cur_db_name_idxs]
            cur_db_loc_data = self.loc_np_set[cur_db_name_idxs]
            res_db_name_set.append(name)
            res_db_desc_set.append(cur_db_desc_data)
            res_db_loc_set.append(cur_db_loc_data)

        return cur_img_desc_data,cur_img_loc_data,res_db_name_set,res_db_desc_set,res_db_loc_set




def construct_retrieval_system(img_path):
    '''
    #初始化索引类（载入PCA参数、载入模型、初始化faiss对象）
    :return:
    '''
    feeder_config = {
        'GPU_ID': 1,
        'IOU_THRES': 0.98,
        'ATTN_THRES': 0.37,
        'TOP_K': 500,
        'PCA_PARAMETERS_PATH': '/data01/majunfu/code/image_retrival_delf/train/repo/pca/pca.h5',
        'PCA_DIMS': 40,
        'USE_PCA': True,
        'SCALE_LIST': [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4147, 2.0],
        'LOAD_FROM': '/data01/majunfu/code/image_retrival_delf/train/repo/landmark/keypoint/ckpt/bestshot.pth.tar',
        'ARCH': 'resnet50',
        'EXPR': 'dummy',
        'TARGET_LAYER': 'layer3',
        'FEATURE_DATA_PATH':'/data01/majunfu/code/image_retrival_delf/train/repo/db_data/oxf5k_index.delf'
    }

    delf_obj = DelfInferece(feeder_config)
    delf_obj.load_fea_data()
    cur_img_desc_data,cur_img_loc_data,res_db_name_set,res_db_desc_set,res_db_loc_set = delf_obj.img_query_db(img_path)

    query_img = cv2.imread(img_path)
    root_path = '/data02/majunfu/scan_data/fea_db/first_fea_data/data'
    save_show_path = '/data01/majunfu/code/image_retrival_delf/results'

    for i,img_name in enumerate(res_db_name_set):
        cur_db_img = cv2.imread(os.path.join(root_path,img_name))
        cur_db_loc_data = res_db_loc_set[i]
        cur_db_desc_data = res_db_desc_set[i]
        cur_inliers, cur_locations_1_to_use, cur_locations_2_to_use = delf_obj.get_ransac_image(cur_img_loc_data, cur_img_desc_data, cur_db_loc_data, cur_db_desc_data)
        print(cur_inliers)
        cur_save_path = os.path.join(save_show_path,'{:08}'.format(i)+'.jpg')
        delf_obj.draw_result_map(query_img,cur_db_img,cur_inliers, cur_locations_1_to_use, cur_locations_2_to_use,cur_save_path)



if __name__ == '__main__':
    img_path = '00011263.jpg'
    construct_retrieval_system(img_path)
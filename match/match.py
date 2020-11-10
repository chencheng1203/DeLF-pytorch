import os, sys

from numpy.lib.function_base import extract
sys.path.append('/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/')

import pickle
import cv2
import faiss
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform


from extract.store_delf_feature import get_final_results
from extract.extract_cfg import extract_config
from match_cfg import match_config

def get_inliers(query_loc, query_des, index_loc, index_des, dis_thresh):  
    """
    steps:
    1. apply KDTree for feature selected
    2. apply RANSAC for getting inliers
    inputs: query image features and corresponding location of features
            index image features and corresponding location of features
    return: index of inliners, seleted location of query image and index image
    """
    num_features_query = query_loc.shape[0]
    num_features_index = index_loc.shape[0]

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(query_des)
    # get feature index of index_des with distance < dis_thresh
    distances, indices = d1_tree.query(index_des, distance_upper_bound=dis_thresh)
    # Select feature locations for putative matches.
    index_locations_to_use = np.array([
        index_loc[i,] for i in range(num_features_index)
        if indices[i] != num_features_query])
    query_locations_to_use = np.array([
        query_loc[indices[i],] for i in range(num_features_index)
        if indices[i] != num_features_query])
    # Perform geometric verification using RANSAC.
    model_robust, inliers = ransac(
        (query_locations_to_use, index_locations_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)
    return inliers, query_locations_to_use, index_locations_to_use


def make_index_table(descriptors_list):   
    """get image index for every feature
    """
    des_from_img = {}
    img_from_des = {}
    cnt = 0
    for i_img, des_list in enumerate(descriptors_list):
        i_des_range = range(cnt, cnt+len(des_list))
        des_from_img[i_img] = list(i_des_range)
        for i_des in i_des_range:
            img_from_des[i_des] = i_img

        cnt+=len(des_list)
    return des_from_img, img_from_des


def get_sim_img(query_des2imglist):
    """
    map feature index to image index
    """
    all_searched_des = []
    for des_i in query_des2imglist.keys():
        all_searched_des.extend(query_des2imglist[des_i])

    imgFreq = Counter(all_searched_des).most_common()
    index, freq = list(zip(*imgFreq))
    result = {'index': index, 'freq':freq}
    return result


def get_stored_delf_info(delf_saved_path):
    """
    we saved delf extraction info in pkl file, which include:
    filename: which index image
    location_np_list: the location corresponding to the features
    descriptor_np_list: delf features
    """
    with open(delf_saved_path, 'rb') as f:
        delf_store = pickle.load(f)
    location_lists = []
    des_lists = []
    name_list = []
    for record in delf_store:
        location_lists.append(record['location_np_list'])
        des_lists.append(record['descriptor_np_list'])
        name_list.append(record['filename'][0])
    
    des_concat_np = np.concatenate(np.asarray(des_lists), axis=0)
    des_from_img, img_from_des = make_index_table(des_lists)

    return des_concat_np, location_lists, name_list, des_from_img, img_from_des


def build_retival_sys(match_cfg, extract_cfg):
    print('delf stored info loading...')
    des_concat_np, location_lists, name_list, _, img_from_des\
        = get_stored_delf_info(match_cfg.delf_saved_path)
    print('delf stored info loaded.')

    print('index system building...')
    pca_dims = match_cfg.pca_dims
    n_subq = 8  # number of sub-quantizers
    n_centroids = 32  # number of centroids for each sub-vector
    n_bits = 5  # number of bits for each sub-vector
    n_probe = 3  # number of voronoi cell to explore
    coarse_quantizer = faiss.IndexFlatL2(pca_dims)
    pq = faiss.IndexIVFPQ(coarse_quantizer, pca_dims, n_centroids, n_subq, n_bits)
    pq.nprobe = n_probe
    pq.train(des_concat_np)
    pq.add(des_concat_np)
    print('index system builded.')

    if isinstance(match_cfg.query_img_path, list):
        retrival_map_list = {}
        for img_path in match_cfg.query_img_path:
            test_img = cv2.imread(img_path)
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            if np.max(test_img) > 1:
                test_img = test_img / 255.
            inputs = torch.from_numpy(test_img.transpose(2, 0, 1)).unsqueeze(dim=0).float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            query_outputs = get_final_results(extract_cfg, inputs)
            query_loc = query_outputs['location_np_list']
            query_des = query_outputs['descriptor_np_list']
            _,query_des2deslist = pq.search(query_des, match_cfg.topk_features)
            
            query_des2imgList = {}
            for img_i,des_list in enumerate(query_des2deslist):
                query_des2imgList[img_i] = [img_from_des[des_i] for des_i in des_list]
            retrivaled_img_index = get_sim_img(query_des2imgList)['index'][:match_cfg.topk_retrival_imgs]
            
            retrivaled_img_path = []
            for img_index in retrivaled_img_index:
                abst_path = os.path.join(match_cfg.indx_img_root, name_list[img_index])
                retrivaled_img_path.append(abst_path)
            retrival_map_list[img_path] = retrivaled_img_path

        if not match_cfg.is_visulize:
            return retrival_map_list
        else:
            query_img_paths = retrival_map_list.keys()
            for query_img_path in query_img_paths:
                plt.figure(dpi=200)
                query_img = plt.imread(query_img_path)
                plt.subplot(1, match_cfg.topk_retrival_imgs+1, 1)
                plt.imshow(query_img)
                plt.title('query image')
                plt.xticks([])
                plt.yticks([])
                for i, retrival_img_path in enumerate(retrival_map_list[query_img_path]):
                    index_img = plt.imread(retrival_img_path)
                    plt.subplot(1, match_cfg.topk_retrival_imgs+1, i+2)
                    plt.imshow(index_img)
                    plt.xticks([])
                    plt.yticks([])
                plt.show()
            return retrival_map_list
    else:
        test_img = cv2.imread(match_cfg.query_img_path)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        if np.max(test_img) > 1:
            test_img = test_img / 255.
        inputs = torch.from_numpy(test_img.transpose(2, 0, 1)).unsqueeze(dim=0).float()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        query_outputs = get_final_results(extract_cfg, inputs)
        query_loc = query_outputs['location_np_list']
        query_des = query_outputs['descriptor_np_list']
        _,query_des2deslist = pq.search(query_des, match_cfg.topk_features)
        
        query_des2imgList = {}
        for img_i,des_list in enumerate(query_des2deslist):
            query_des2imgList[img_i] = [img_from_des[des_i] for des_i in des_list]
        retrivaled_img_index = get_sim_img(query_des2imgList)['index'][:match_cfg.topk_retrival_imgs]
        
        retrivaled_img_path = []
        for img_index in retrivaled_img_index:
            abst_path = os.path.join(match_cfg.indx_img_root, name_list[img_index])
            retrivaled_img_path.append(abst_path)
        
        if not match_cfg.is_visulize:
            return retrivaled_img_path
        else:
            query_img_path = match_cfg.query_img_path
            plt.figure(dpi=200)
            query_img = plt.imread(query_img_path)
            plt.subplot(1, match_cfg.topk_retrival_imgs+1, 1)
            plt.imshow(query_img)
            plt.title('query image')
            plt.xticks([])
            plt.yticks([])
            for i, retrival_img_path in enumerate(retrivaled_img_path):
                index_img = plt.imread(retrival_img_path)
                plt.subplot(1, match_cfg.topk_retrival_imgs+1, i+2)
                plt.imshow(index_img)
                plt.xticks([])
                plt.yticks([])
            plt.show()
            return retrivaled_img_path


def keypoints_match_visulize(match_cfg, extract_cfg):
    print('delf stored info loading...')
    des_concat_np, location_lists, name_list, _, img_from_des\
        = get_stored_delf_info(match_cfg.delf_saved_path)
    print('delf stored info loaded.')

    print('index system building...')
    pca_dims = match_cfg.pca_dims
    n_subq = 8  # number of sub-quantizers
    n_centroids = 32  # number of centroids for each sub-vector
    n_bits = 5  # number of bits for each sub-vector
    n_probe = 3  # number of voronoi cell to explore
    coarse_quantizer = faiss.IndexFlatL2(pca_dims)
    pq = faiss.IndexIVFPQ(coarse_quantizer, pca_dims, n_centroids, n_subq, n_bits)
    pq.nprobe = n_probe
    pq.train(des_concat_np)
    pq.add(des_concat_np)
    print('index system builded.')

    query_img = cv2.imread(match_cfg.query_img_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    if np.max(query_img) > 1:
        query_img = query_img / 255.
    inputs = torch.from_numpy(query_img.transpose(2, 0, 1)).unsqueeze(dim=0).float()
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    query_outputs = get_final_results(extract_cfg, inputs)
    query_loc = query_outputs['location_np_list']
    query_des = query_outputs['descriptor_np_list']
    _,query_des2deslist = pq.search(query_des, match_cfg.topk_features)
    
    query_des2imgList = {}
    for img_i,des_list in enumerate(query_des2deslist):
        query_des2imgList[img_i] = [img_from_des[des_i] for des_i in des_list]
    retrivaled_img_index = get_sim_img(query_des2imgList)['index'][0]
    retrivaled_img_path = os.path.join(match_cfg.indx_img_root, name_list[retrivaled_img_index])

    # get choosen index image feature info
    index_img = cv2.imread(retrivaled_img_path)
    index_img = cv2.cvtColor(index_img, cv2.COLOR_BGR2RGB)
    if np.max(index_img) > 1:
        index_img = index_img / 255.
    index_inputs = torch.from_numpy(index_img.transpose(2, 0, 1)).unsqueeze(dim=0).float()
    if torch.cuda.is_available():
        index_inputs = index_inputs.cuda()
    index_outputs = get_final_results(extract_cfg, index_inputs)
    index_loc = index_outputs['location_np_list']
    index_des = index_outputs['descriptor_np_list']
    
    inliers, query_loc, index_loc = get_inliers(query_loc,\
                                                query_des,\
                                                index_loc,\
                                                index_des, dis_thresh=5)
    inlier_idxs = np.nonzero(inliers)[0]
    inlier_matches = []
    for idx in inlier_idxs:
        inlier_matches.append(cv2.DMatch(idx, idx, 0))
    kp1 =[]
    for point in query_loc:
        kp = cv2.KeyPoint(point[1], point[0], _size=1)
        kp1.append(kp)

    kp2 =[]
    for point in index_loc:
        kp = cv2.KeyPoint(point[1], point[0], _size=1)
        kp2.append(kp)
    
    query_img = plt.imread(match_cfg.query_img_path)
    index__img = plt.imread(retrivaled_img_path)
    ransac_img = cv2.drawMatches(query_img, kp1, index__img, kp2, inlier_matches, None, flags=0)
    plt.figure(dpi=200)
    plt.imshow(ransac_img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':
    match_cfg = match_config()
    extract_cfg = extract_config()
    keypoints_match_visulize(match_cfg, extract_cfg)
    # build_retival_sys(match_cfg, extract_cfg)





            
            









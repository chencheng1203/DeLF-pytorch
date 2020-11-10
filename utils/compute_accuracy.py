import torch
import numpy as np

def cal_accuracy(outputs, labels):
    batch_size = outputs.shape[0]
    _, top_5_index = outputs.topk(5, dim=1)
    _, top_3_index = outputs.topk(3, dim=1)
    _, top_1_indx = outputs.topk(1, dim=1)

    top1_counts = 0
    top3_counts = 0
    top5_counts = 0
    for index, label in enumerate(labels):
        if label in top_1_indx[index]:
            top1_counts += 1
            top3_counts += 1
            top5_counts += 1
            continue
        if label in top_3_index[index]:
            top3_counts += 1
            top5_counts += 1
            continue
        if label in top_5_index[index]:
            top5_counts += 1
            continue
    return top1_counts / batch_size, top3_counts / batch_size, top5_counts / batch_size

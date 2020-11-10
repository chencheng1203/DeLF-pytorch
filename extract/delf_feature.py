import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import random
import torch.nn.functional as F

from pca import ApplyPcaAndWhitening
from extract_utils import nms

class Attention2d(nn.Module):

    def __init__(self, in_c, act_fn='relu'):
        super(Attention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 512, 1, 1)
        if act_fn.lower() in ['leakyrelu']:
            self.act1 = nn.LeakyReLU()
        else:
            self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1)
        self.act2 = nn.Softplus(beta=1, threshold=20)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        return x

class AttentionWeighted(nn.Module):

    def __init__(self):
        super(AttentionWeighted, self).__init__()
    def forward(self, inputs):
        x, scores = inputs
        assert x.size(2) == scores.size(2) and x.size(3) == scores.size(3),\
                'err: h, w of tensors x({}) and weights({}) must be the same.'\
                .format(x.size, scores.size)
        y = x * scores  # [N, c, h, w]
        y = y.view(-1, x.size(1), x.size(2) * x.size(3))  # [N, c, h * w]
        return torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)  # [N, c, 1, 1]

class logits(nn.Module):
    
    def __init__(self, in_c, ncls):
        super(logits, self).__init__()
        self.conv = nn.Conv2d(in_c, ncls, 1)
    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # ResNet-B
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Delf_feature(nn.Module):

    def __init__(self, block, layers, traget_layers='layer3', l2_norm=True):
        self.traget_layers = traget_layers
        self.l2_norm = l2_norm
        self.inplanes = 64
        super(Delf_feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        if self.traget_layers == 'layer4':
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.atte = Attention2d(self.__get_chs__(self.traget_layers))
        self.atteWeighted = AttentionWeighted()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def __get_chs__(self, layers):
        assert layers in ['layer2', 'layer3', 'layer4'], "[Error] layer must in resnet layer out"
        ch_map = {'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        return ch_map[layers]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.traget_layers == 'layer4':
            x = self.layer4(x)
        ret_fea = x
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=1)
        atte_scores = self.atte(x)
        # weighted_fea = self.atteWeighted([x, atte_scores])
        return {'features': ret_fea, 'atte_scores': atte_scores}


def get_delf_feature(keypoint_state_dict_path=None):
    """construct original feature extraction"""
    assert keypoint_state_dict_path != None, "[Error] finetune state-dict must be required"
    model = Delf_feature(Bottleneck, [3, 4, 6, 3])
    keypoint_state_dict = torch.load(keypoint_state_dict_path)
    model_state_dict = model.state_dict()

    print('loading keypoints state-dict...')
    update_state_dict = {}
    for k, v in model_state_dict.items():
        update_state_dict[k] = keypoint_state_dict[k]
    model.load_state_dict(update_state_dict)
    print('keypints state-dict loading done!')
    model.eval()
    return model


##########################
### Field Receptive about
##########################
def GenerateCoordinates(h, w):
    # x = torch.floor(torch.arange(0, w*h) / w)
    x = torch.arange(0, w * h) / w
    y = torch.arange(0, w).repeat(h)
    coord = torch.stack([x,y], dim=1)
    return coord

def CalculateReceptiveBoxes(height, width, rf, stride, padding):
    coordinates = GenerateCoordinates(h=height,
                                      w=width)
    # create [ymin, xmin, ymax, xmax]
    point_boxes = torch.cat([coordinates, coordinates], dim=1)
    bias = torch.FloatTensor([-padding, -padding, -padding + rf - 1, -padding + rf - 1])
    rf_boxes = stride * point_boxes + bias
    return rf_boxes 

def CalculateKeypointCenters(rf_boxes):
    if torch.cuda.is_available():
        index1 = torch.LongTensor([0, 1]).cuda()
        index2 = torch.LongTensor([2, 3]).cuda()
        rf_boxes = rf_boxes.cuda()
    else:
        index1 = torch.LongTensor([0, 1])
        index2 = torch.LongTensor([2, 3])
    xymin = torch.index_select(rf_boxes, dim=1, index=index1)
    xymax = torch.index_select(rf_boxes, dim=1, index=index2)
    return (xymax + xymin) / 2.0


##########################
###      Processing
##########################
def DelfFeaturePostProcessing(boxes, descriptors, pca_mean, pca_vars, pca_matrix, pca_dims, stage):
    ''' Delf feature post-processing.
    (1) apply L2 Normalization.
    (2) apply PCA and Whitening.
    (3) apply L2 Normalization once again.
    Args:
        descriptors: (w x h, fmap_depth) descriptor Tensor.
    Retturn:
        descriptors: (w x h, pca_dims) desciptor Tensor.
    '''
    locations = CalculateKeypointCenters(boxes)

    # L2 Normalization.
    descriptors = descriptors.squeeze()
    l2_norm = descriptors.norm(p=2, dim=1, keepdim=True)        # (1, w x h)
    descriptors = descriptors.div(l2_norm.expand_as(descriptors))  # (N, w x h)

    if stage == 'delf':
        # apply PCA and Whitening.
        descriptors = ApplyPcaAndWhitening(descriptors, pca_matrix, pca_mean, pca_vars, pca_dims, True)
    return descriptors


def GetDelfFeatureFromSingleScale(stage, x, model, scale, pca_mean, pca_vars, pca_matrix,pca_dims,
                                    rf, stride, padding, attn_thres):

    new_h = int(round(x.size(2)*scale))
    new_w = int(round(x.size(3)*scale))
    scaled_x = F.interpolate(x, size=(new_h, new_w), mode='bilinear')
    results= model(scaled_x)
    scaled_features, scaled_scores = results['features'], results['atte_scores']

    # save original size attention (used for attention visualization.)
    selected_original_scale_attn = None
    if scale == 1.0:
        selected_original_scale_attn = torch.clamp(scaled_scores*255, 0, 255) # [1 x 1 x h x w]
    rf_boxes = CalculateReceptiveBoxes(
        height=scaled_features.size(2),
        width=scaled_features.size(3),
        rf=rf,
        stride=stride,
        padding=padding)
    # re-projection back to original image space.
    rf_boxes = rf_boxes / scale
    scaled_scores = scaled_scores.view(-1)  # 获取每个特征的得分（Flatten的方式）
    scaled_features = scaled_features.view(scaled_features.size(1), -1).t()

    scaled_features = DelfFeaturePostProcessing(rf_boxes, scaled_features, pca_mean, pca_vars, pca_matrix, pca_dims, stage)
    
    # use attention score to select feature.
    indices = None
    while(indices is None or len(indices) == 0):
        indices = torch.gt(scaled_scores, attn_thres).nonzero().squeeze()
        attn_thres = attn_thres * 0.5   # use lower threshold if no indexes are found.
        if attn_thres < 0.001:
            break;
    try:
        if torch.cuda.is_available():
            rf_boxes = rf_boxes.cuda()
        selected_boxes = torch.index_select(rf_boxes, dim=0, index=indices)
        selected_features = torch.index_select(scaled_features, dim=0, index=indices)
        selected_scores = torch.index_select(scaled_scores, dim=0, index=indices)
        selected_scales = torch.ones_like(selected_scores) * scale
    except Exception as e:
        selected_boxes = None
        selected_features = None
        selected_scores = None
        selected_scales = None
        print(e)
        pass;

    return selected_boxes, selected_features, selected_scales, selected_scores, selected_original_scale_attn


def getDelfFeatureFromMultiScale(stage='pca', inputs=None, model=None, filename=None,\
                                 pca_mean=None, pca_var=None, pca_matrix=None, pca_dims=None,\
                                 rf=None, stride=None, padding=None, topk=None, scales=None,\
                                 iou_thresh=None, attn_thres=None):
    output_boxes = []
    output_features = []
    output_scores = []
    output_scales = []
    output_original_scale_attn = None

    features = []
    for scale in scales:
        future = GetDelfFeatureFromSingleScale(stage, inputs, model, scale, \
                                               pca_mean, pca_var, pca_matrix, pca_dims, \
                                               rf, stride, padding, attn_thres)
        features.append(future)
    
    for feature in features:
        (selected_boxes, selected_features, selected_scales, selected_scores, selected_original_scale_attn) = feature
        output_boxes.append(selected_boxes) if selected_boxes is not None else output_boxes
        output_features.append(selected_features) if selected_features is not None else output_features
        output_scales.append(selected_scales) if selected_scales is not None else output_scales
        output_scores.append(selected_scores) if selected_scores is not None else output_scores
        if selected_original_scale_attn is not None:
            output_original_scale_attn = selected_original_scale_attn
    
    # if scale == 1.0 is not included in scale list, just show noisy attention image.
    if output_original_scale_attn is None:
        output_original_scale_attn = inputs.clone().uniform()

    output_boxes = torch.cat(output_boxes, dim=0)
    output_features = torch.cat(output_features, dim=0)
    output_scales = torch.cat(output_scales, dim=0)
    output_scores = torch.cat(output_scores, dim=0)

    keep_indices, count = nms(boxes = output_boxes,
                              scores = output_scores,
                              overlap = iou_thresh,
                              top_k = topk)
    
    keep_indices = keep_indices[:topk]  # 选择前topk个特征
    output_boxes = torch.index_select(output_boxes, dim=0, index=keep_indices)
    output_features = torch.index_select(output_features, dim=0, index=keep_indices)
    output_scales = torch.index_select(output_scales, dim=0, index=keep_indices)
    output_scores = torch.index_select(output_scores, dim=0, index=keep_indices)
    output_locations = CalculateKeypointCenters(output_boxes)

    data = {
        'filename':filename,
        'location_np_list':output_locations.cpu().detach().numpy(),
        'descriptor_np_list':output_features.cpu().detach().numpy(),
        'feature_scale_np_list':output_scales.cpu().detach().numpy(),
        'attention_score_np_list':output_scores.cpu().detach().numpy(),
        'attention_np_list':output_original_scale_attn.cpu().detach().numpy()
    }

    # free Memory
    del output_locations
    del output_features
    del output_scales
    del output_scores
    del output_original_scale_attn

    return data


if __name__ == '__main__':
    kp_path = "/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/repository/finetune/epoch_4_loss_2.32170.pth"
    res = get_delf_feature(keypoint_state_dict_path=kp_path)
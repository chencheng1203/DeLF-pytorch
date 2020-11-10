import torch
import torch.nn as nn
import math
import random
import torch.nn.functional as F


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


class Delf_atte(nn.Module):

    def __init__(self, block, layers, traget_layers='layer3',\
                 random_resize=True, l2_norm=True, num_classes=1000):
        self.traget_layers = traget_layers    
        self.random_resize = random_resize
        self.l2_norm = l2_norm
        self.inplanes = 64
        super(Delf_atte, self).__init__()
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

        self.logits = logits(self.__get_chs__(self.traget_layers), num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
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
    
    def __gamma_rescale__(self, x, min_scale=0.3535, max_scale=1.0):
        '''max_scale > 1.0 may cause training failure.
        '''
        h, w = x.size(2), x.size(3)
        assert w == h, 'input must be square image.'
        gamma = random.uniform(min_scale, max_scale)
        new_h, new_w = int(h*gamma), int(w*gamma)
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear')
        return x

    def forward(self, x):
        if self.random_resize:
            x = self.__gamma_rescale__(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.traget_layers == 'layer4':
            x = self.layer4(x)
        if self.l2_norm:
            atte_x = F.normalize(x, p=2, dim=1)
        else:
            atte_x = x
        atte_scores = self.atte(x)
        weighted_fea = self.atteWeighted([atte_x, atte_scores])
        out = self.logits(weighted_fea)

        return out


def Delf_res50_atte(num_classes=1000, ft_state_dict_path=None):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    assert ft_state_dict_path != None, "[Error] finetune state-dict must be required"
    model = Delf_atte(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    ft_state_dict = torch.load(ft_state_dict_path)
    model_state_dict = model.state_dict()
    print('loading finetune state-dict...')
    update_state_dict = {}
    for k, v in model_state_dict.items():
        if k in ft_state_dict:
            update_state_dict[k] = ft_state_dict[k]
        else:
            update_state_dict[k] = v
    model.load_state_dict(update_state_dict)
    print('finetune state-dict loading done!')
    print('freezing finetune layers...')
    param_len = len(list(model.parameters()))
    for index, (name, param) in enumerate(model.named_parameters()):
        if index < param_len - 6:
            param.requires_grad = False
        else:
            param.requires_grad = True
    print('finetune layers frozen done.')
    return model

if __name__ == '__main__':
    ft_path = "/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/repository/finetune/epoch_4_loss_2.32170.pth"
    res = Delf_res50_atte(num_classes=10000, ft_state_dict_path=ft_path)
"""
Contains Network architectures.
The network architectures and weights are partly adapted and used from the great repository https://github.com/Cadene/pretrained-models.pytorch.
"""

import torch, os, numpy as np

import torch.nn as nn

import pretrainedmodels as ptm
import pretrainedmodels.utils as utils

import torchvision.models as models



"""============================================================="""
def initialize_weights(model, type='none'):
    if type=='base':
        for idx,module in enumerate(model.modules()):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0,0.01)
                module.bias.data.zero_()
    elif type=='he_n':
        for idx,module in enumerate(model.modules()):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                module.bias.data.zero_()
    else:
        pass



"""=================================================================================================================================="""
### ATTRIBUTE CHANGE HELPER
def rename_attr(model, attr, name):
    setattr(model, name, getattr(model, attr))
    delattr(model, attr)


def multi_getattr(obj, attr):
    if not isinstance(attr,list):
        attributes = attr.split(".")
    else:
        attributes = attr
    for i in attributes:
        obj = getattr(obj, i)
    return obj


def multi_setattr(obj, obj_2_set, attr):
    if not isinstance(attr,list):
        attributes, attr_to_set = attr.split(".")[:-1],attr.split(".")[-1]
    else:
        attributes, attr_to_set = attr[:-1],attr[-1]

    for i in attributes:
        obj = getattr(obj, i)
    setattr(obj,attr_to_set,obj_2_set)



"""=================================================================================================================================="""
### NETWORK SELECTION FUNCTION
def networkselect(opt):
    if opt.arch == 'resnet50':
        network =  NetworkSuperClass_ResNet50(opt)
    elif opt.arch == 'bninception':
        network =  NetworkSuperClass_BNInception(opt)
    else:
        raise Exception('Network {} not available!'.format(opt.arch))
    return network



"""============================================================="""
class NetworkSuperClass_ResNet50(nn.Module):
    def __init__(self, opt):
        super(NetworkSuperClass_ResNet50, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)

        x = self.model.last_linear(x)
        return torch.nn.functional.normalize(x, dim=-1)





"""============================================================="""
class NetworkSuperClass_BNInception(nn.Module):
    def __init__(self, opt):
        super(NetworkSuperClass_BNInception, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['bninception'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['bninception'](num_classes=1000, pretrained=None)

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

    def forward(self, x):
        x = self.model(x)
        return torch.nn.functional.normalize(x, dim=-1)

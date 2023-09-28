# Demo Code for Paper:
# [Title]  - "MSBA-Net: Multi-Scale Behavior Analysis Network for Random Hand Gesture Authentication"
# [Author] - Huilong Xie, Wenwei Song, Wenxiong Kang
# [Github] - https://github.com/SCUT-BIP-Lab/MSBA-Net.git

import torch
import torch.nn as nn
import torchvision
from module.MSBA_module import MSBA_module


class Model_MSBANet(torch.nn.Module):
    """
    # 模型样板
    """
    def __init__(self, frame_length, feature_dim, out_dim):
        super(Model_MSBANet, self).__init__()

        # there are 20 frames in each random hand gesture video
        self.frame_length = frame_length

        self.out_dim = out_dim  # the identity feature dim

        #conf
        self.num_heads = [32, 32, 64, 128, 256]
        self.sr_ratios = [8, 8, 4, 2, 1]
        self.T_sp = [5, 5, 5, 5, 5]
        self.local_time = [True, True, True, False, False]

        self._pstnet_layer()

        # cv模型
        self.model = torchvision.models.resnet18(pretrained=True)
        # change the last fc with the shape of 512×128
        self.model.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def _make_transformer(self, embed_dim, num_heads, T_sp, sr_ratio, local_time=False):
        layer = MSBA_module(embed_dim, num_heads, T=self.frame_length, T_sp=T_sp, sr_ratio=sr_ratio, local_time=local_time)
        return layer
    

    def _pstnet_layer(self):
        self.transformer1 = self._make_transformer(64, self.num_heads[0], self.T_sp[0], self.sr_ratios[0], self.local_time[0])
        self.transformer2 = self._make_transformer(64, self.num_heads[1], self.T_sp[1], self.sr_ratios[1], self.local_time[1])
        self.transformer3 = self._make_transformer(128, self.num_heads[2], self.T_sp[2], self.sr_ratios[2], self.local_time[2])
        self.transformer4 = self._make_transformer(256, self.num_heads[3], self.T_sp[3], self.sr_ratios[3], self.local_time[3])
        self.transformer5 = self._make_transformer(512, self.num_heads[4], self.T_sp[4], self.sr_ratios[4], self.local_time[4])

    def transformer_forward(self, layer, feature, T):
        transformer_func = "transformer"+str(layer)
        f = getattr(self, transformer_func)
        return f(feature, T)

    def forward(self, data, label=None):
        
        fis = {} # 字典存结果

        # 如果是GPU
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            data = data.cuda()
            if label is not None:
                label = label.cuda()
        data = data.view((-1,)+data.shape[-3:])

        feature = self.model.conv1(data)
        feature = self.model.bn1(feature)
        feature = self.model.relu(feature)
        feature = self.model.maxpool(feature)

        feature = self.transformer_forward(1, feature, self.frame_length)

        feature = self.model.layer1(feature)

        feature = self.transformer_forward(2, feature, self.frame_length)

        feature = self.model.layer2(feature)

        feature = self.transformer_forward(3, feature, self.frame_length)

        feature = self.model.layer3(feature)

        feature = self.transformer_forward(4, feature, self.frame_length)

        feature = self.model.layer4(feature)

        feature = self.transformer_forward(5, feature, self.frame_length)

        feature = self.avgpool(feature)
        feature = torch.flatten(feature, 1)
        feature = self.model.fc(feature)
        feature = feature.view(-1, self.frame_length, self.out_dim)

        feature = torch.mean(feature, dim=1, keepdim=False)
        feature = torch.div(feature, torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12))
        return feature

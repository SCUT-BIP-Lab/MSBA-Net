# Demo Code for Paper:
# [Title]  - "MSBA-Net: Multi-Scale Behavior Analysis Network for Random Hand Gesture Authentication"
# [Author] - Huilong Xie, Wenwei Song, Wenxiong Kang
# [Github] - https://github.com/SCUT-BIP-Lab/MSBA-Net.git

import torch
from model.MSBANet import Model_MSBANet
# from loss.loss import AMSoftmax

def feedforward_demo(frame_length, feature_dim, out_dim):
    model = Model_MSBANet(frame_length=frame_length, feature_dim=feature_dim, out_dim=out_dim)
    # AMSoftmax loss function
    # criterian = AMSoftmax(in_feats=out_dim, n_classes=143)
    # there are 143 identities in the training set
    data = torch.randn(2, 20, 3, 224, 224) #batch, frame, channel, h, w
    data = data.view(-1, 3, 224, 224) #regard the frame as batch
    id_feature = model(data) # feedforward
    # Use the id_feature to calculate the EER when testing or to calculate the loss when training
    # when training
    # loss_backbone, _ = self.criterian(id_feature, label)

    return id_feature

if __name__ == '__main__':
    # there are 20 frames in each random hand gesture video
    frame_length = 20
    # the feature dim of last feature map (layer4) from ResNet18 is 512
    feature_dim = 512
    # the identity feature dim
    out_dim = 128

    # feedforward process
    id_feature = feedforward_demo(frame_length, feature_dim, out_dim)
    print("Demo is finished!")


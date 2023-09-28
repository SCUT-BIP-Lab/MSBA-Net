# MSBA-Net: Multi-Scale Behavior Analysis Network
Pytorch Implementation of paper:

> **MSBA-Net: Multi-Scale Behavior Analysis Network for Random Hand Gesture Authentication**
>
> Huilong Xie, Wenwei Song, Wenxiong Kang\*.

## Main Contribution
Random hand gesture authentication allows the probe hand gesture types to be inconsistent with the registered ones. While it is highly user-friendly, it poses a significant challenge that requires the authentication model to distill more abstract and complex identity features. Prior efforts on random hand gesture authentication mainly use convolution operations to obtain short-term behavioral information and cannot distill robust behavioral features well. In this paper, we propose a novel Multi-Scale Behavior Analysis Network (MSBA-Net), with a focus on capturing multi-scale behavioral features for random hand gesture authentication, which can simultaneously distill short-term behavioral information and model long-term behavioral relationships in addition to physiological features of hand gestures. In addition, as hand motion can result in inter-frame semantic misalignment, we propose an efficient semantic alignment strategy to mitigate this issue, which helps extract behavior features accurately and improves model performance. MSBA module is a plug-and-play module and could be integrated into existing 2D CNNs to yield a powerful video understanding model (MSBA-Net). Extensive experiments on the SCUT-DHGA dataset demonstrate that our MSBA-Net has compelling advantages over the other 20 state-of-the-art methods.
<div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/MSBA-Net/master/img/MSBA-Net.png" />
</p>
</div>

MSBA-Net. The MSBA module is inserted after each stage of ResNet to form a spatial-temporal feature extractor with the original components of ResNet. The ST Layer at the last four stages has the same structure, which combines Layer and an MSBA module. The GSAP and GTAP denote global spatial average pooling and global temporal average pooling, respectively. $T$ is the frame number. We omit the downsampling block in the residual connection of the first BasicBlock in the last three stages of MSBA-Net (the last three layers of ResNet18). The details of the MSBA module are depicted in the blue dashed box on the right side. The PM and RS denote permute and reshape operations, respectively. In the purple dashed box, the channels of Q, K, and V are equally divided into N heads, and every head does different scale temporal modeling by setting different t. The weights and biases of the 3D Batch Normalization Layer (BN3D) in MSBA module are initialized to zero.


## Dependencies
Please make sure the following libraries are installed successfully:
- [PyTorch](https://pytorch.org/) >= 1.7.0

## How to use
This repository is a demo of MSBA-Net. Through debugging ([main.py](/main.py)), you can quickly understand the configuration and building method of [MSBA-Net](/model/MSBANet.py), including the MSBA module.

If you want to explore the entire random hand gesture authentication framework, please refer to our pervious work [SCUT-DHGA](https://github.com/SCUT-BIP-Lab/SCUT-DHGA) 
or send an email to Prof. Kang (auwxkang@scut.edu.cn).

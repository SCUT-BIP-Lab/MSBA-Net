# Demo Code for Paper:
# [Title]  - "MSBA-Net: Multi-Scale Behavior Analysis Network for Random Hand Gesture Authentication"
# [Author] - Huilong Xie, Wenwei Song, Wenxiong Kang
# [Github] - https://github.com/SCUT-BIP-Lab/MSBA-Net.git

from torch import nn
import torch
import torch.nn.functional
import numpy as np
from timm.models.layers import trunc_normal_

class MSBA_module(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 T=20,
                 T_sp=10,
                 qkv_bias=True,
                 sr_ratio=1,
                 local_time=False):
        super().__init__()

        self.T_sp = T_sp
        self.num_heads = num_heads
        self.local_time = local_time

        self.q = nn.Linear(dim, num_heads, bias=qkv_bias)
        self.kv = nn.Linear(dim, num_heads * 2, bias=qkv_bias)
        self.proj = nn.Conv2d(num_heads, dim, kernel_size=1, stride=1)
        self.sr_ratio = sr_ratio

        # position embedding
        if local_time or self.T_sp == T:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * T_sp - 1), num_heads))  # 2*Ts, nH
            # get pair-wise relative position index for each token inside the window
            coords_t = torch.arange(T_sp)
            coords = torch.stack(torch.meshgrid([coords_t]))  # 2, T
            coords_flatten = torch.flatten(coords, 1)  # 2, T
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, T, T
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # T, T, 2
            relative_coords[:, :, 0] += T_sp - 1  # shift to start from 0
            relative_position_index = relative_coords.sum(-1)  # T, T
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)
        else:
            self.relative_position_bias_table1 = nn.Parameter(
                torch.zeros((2 * T - 1), num_heads // 2))  # 2*T, nH
            # get pair-wise relative position index for each token inside the window
            coords_t = torch.arange(T)
            coords = torch.stack(torch.meshgrid([coords_t]))  # 2, T
            coords_flatten = torch.flatten(coords, 1)  # 2, T
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, T, T
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # T, T, 2
            relative_coords[:, :, 0] += T - 1  # shift to start from 0
            relative_position_index1 = relative_coords.sum(-1)  # T, T
            self.register_buffer("relative_position_index1", relative_position_index1)
            trunc_normal_(self.relative_position_bias_table1, std=.02)

            self.relative_position_bias_table2 = nn.Parameter(
                torch.zeros((2 * T_sp - 1), num_heads // 2))  # 2*Ts-1, nH
            # get pair-wise relative position index for each token inside the window
            coords_t = torch.arange(T_sp)
            coords = torch.stack(torch.meshgrid([coords_t]))  # 2, T
            coords_flatten = torch.flatten(coords, 1)  # 2, T
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, T, T
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # T, T, 2
            relative_coords[:, :, 0] += T_sp - 1  # shift to start from 0
            relative_position_index2 = relative_coords.sum(-1)  # T, T
            self.register_buffer("relative_position_index2", relative_position_index2)
            trunc_normal_(self.relative_position_bias_table2, std=.02)

        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(num_heads, num_heads, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=num_heads)

        self.norm = nn.BatchNorm2d(dim)
        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0.0)

    def im2mstwin(self, x, T, T_sp):
        B, Head, N, C = x.shape
        H = W = int(np.sqrt(N // T))
        x = x.transpose(-2,-1).contiguous().view(B, Head, C, T, H, W)
        img_reshape = x.view(B, Head, C, T // T_sp, T_sp, H, W)
        img_perm = img_reshape.permute(0, 3, 1, 4, 5, 6, 2).contiguous().reshape(-1, Head, T_sp* H* W, C)

        return img_perm # B'(B*T//T_sp) H N C

    def mstwin_tcat(self, x, T, T_sp):
        B, H, N, C = x.shape
        x = x.reshape(-1, T//T_sp, H, N, C).permute(0, 2, 1, 3, 4).contiguous().view(-1, H, T // T_sp * N, C)
        return x # B H N C

    def forward(self, x, T):
        B, C, H, W = x.shape

        qk_scale = 1
        sr_ratio = self.sr_ratio
        qk_scale = (H * W // sr_ratio // sr_ratio) ** 0.5
        z = x.permute(0, 2, 3, 1).reshape(B, H, W, C) # B H W C
        kv = self.kv(z).reshape(B, H, W, 2, self.num_heads).permute(3, 0, 1, 2, 4)
        k = kv[0]
        v = kv[1]

        # temporal enhancement
        z = z.reshape(B // T, T, H, W, C)
        z1 = z[:, -1, :, :, :].unsqueeze(1)
        z = 2 * z[:, :-1, :, :, :] - z[:, 1:, :, :, :]
        z = torch.cat((z, z1), dim=1)

        q = self.q(z).reshape(B, H, W, self.num_heads)

        if sr_ratio > 1:
            q = q.permute(0, 3, 1, 2).contiguous()
            k = k.permute(0, 3, 1, 2).contiguous()
            q = self.sr(q).reshape(B, self.num_heads, H // sr_ratio, W // sr_ratio).permute(0, 2, 3, 1) # B H W C
            k = self.sr(k).reshape(B, self.num_heads, H // sr_ratio, W // sr_ratio).permute(0, 2, 3, 1) # B H W C
        q = q.reshape(B // T, T, H // sr_ratio, W // sr_ratio, self.num_heads) 
        k = k.reshape(B // T, T, H // sr_ratio, W // sr_ratio, self.num_heads)      
        
        if self.local_time:
            q = q.reshape(B // self.T_sp, self.T_sp, H * W // sr_ratio // sr_ratio, self.num_heads)
            k = k.reshape(B // self.T_sp, self.T_sp, H * W // sr_ratio // sr_ratio, self.num_heads)
            v = v.reshape(B // self.T_sp, self.T_sp, H * W, self.num_heads)
        else:
            q = q.reshape(B // T, T, H // sr_ratio * W // sr_ratio, self.num_heads)
            k = k.reshape(B // T, T, H // sr_ratio * W // sr_ratio, self.num_heads)
            v = v.reshape(B // T, T, H * W, self.num_heads)

        q = q.permute(0, 3, 1, 2).contiguous() # B C T HW
        k = k.permute(0, 3, 1, 2).contiguous()
        v = v.permute(0, 3, 1, 2).contiguous()


        if self.T_sp == T or self.local_time:
            attn = (q @ k.transpose(-2, -1)) * qk_scale # B C T T

            relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:self.T_sp, :self.T_sp].reshape(-1)].reshape(
            self.T_sp, self.T_sp, -1)  # T, T, nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, T, T
            attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, T, T

            attn = attn.softmax(dim=-1)

            z = (attn @ v).transpose(1, 2) # B T C HW

        else:
            q1 = q[:, : self.num_heads // 2, :, :]
            q2 = q[:, self.num_heads // 2 :, :, :]
            k1 = k[:, : self.num_heads // 2, :, :]
            k2 = k[:, self.num_heads // 2 :, :, :]
            v1 = v[:, : self.num_heads // 2, :, :]
            v2 = v[:, self.num_heads // 2 :, :, :]
            attn1 = (q1 @ k1.transpose(-2, -1)) * qk_scale # B H N N

            relative_position_bias = self.relative_position_bias_table1[self.relative_position_index1[:T, :T].reshape(-1)].reshape(
            T, T, -1)  # T, T, nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, T, T
            attn1 = attn1 + relative_position_bias.unsqueeze(0) # B_, nH, T, T
            
            attn1 = attn1.softmax(dim=-1)
            x1 = attn1 @ v1

            q2 = self.im2mstwin(q2, T, self.T_sp)
            k2 = self.im2mstwin(k2, T, self.T_sp)
            v2 = self.im2mstwin(v2, T, self.T_sp)
            attn2 = (q2 @ k2.transpose(-2, -1)) * qk_scale # B'(B*T//T_sp) H N N
            
            relative_position_bias = self.relative_position_bias_table2[self.relative_position_index2[:self.T_sp, :self.T_sp].reshape(-1)].reshape(
            self.T_sp, self.T_sp, -1)  # T, T, nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, T, T
            attn2 = attn2 + relative_position_bias.unsqueeze(0) # B_, nH, T, T
            
            attn2 = attn2.softmax(dim=-1)
            x2 = attn2 @ v2    #B'(B*T//T_sp) H N C
            x2 = self.mstwin_tcat(x2, T, self.T_sp)
            z = torch.cat((x1, x2), dim=1).transpose(1, 2)

        # B T C HW
        z = z.reshape(B, self.num_heads, H, W)

        z = self.proj(z)
        z = self.norm(z)
        return x + z
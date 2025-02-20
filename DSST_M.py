import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import LongTensor, Tensor  #--20231125--BRA用
import warnings  #--20231031--PatA用
import math  #--20231031--PatA用
from torch import einsum  #--20231031--PatA用


def window_partition(x, window_size: int):  #--20231216--(bs,H,W,nC)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)  #--20231216--(bs,ph,h,pw,w,nC)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)  #--20231216--(bs*ph*pw,h,w,nC)
    return windows  #--20231216--(bs*ph*pw,h,w,nC)

def window_partition2(x, window_size):  #--20231216--(bs,nC,H,W)
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)  #--20231216--(bs,nC,ph,w,pw,h)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)  #--20231216--(bs*ph*pw,w*h,nC)
    return windows  #--20231216--(bs*ph*pw,hw,nC)

def window_reverse(windows, window_size: int, H: int, W: int):  #--20231216--windows(bs*ph*pw,h,w,nC)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)  #--20231216--(bs,ph,pw,h,w,nC)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)  #--20231216--(bs,H,W,nC)
    return x  #--20231216--(bs,H,W,nC)

def window_reverse2(windows, window_size, H: int, W: int):  #--20231216--windows(bs*ph*pw,h*w,nC)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)  #--20231216--(bs,pw,ph,w,h,nC)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)  #--20231216--(bs,nC,H,W)
    return x  #--20231216--(bs,nC,H,W)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):  #--20231031--HS_MSA用
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):  #--20231031--HS_MSA用，利用正态分布生成点，直至在(-2,2)区间
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

"""
class MGM(nn.Module):  #--20231024--MaskGuidedMechanism
    def __init__(self, num_channel):  #--20231024--num_channel输入特征通道数量
        super(MGM, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, 1, 1, 0, bias=True)  #--20231024--conv1×1
        self.conv2 = nn.Conv2d(num_channel, num_channel, 1, 1, 0, bias=True)  #--20231024--conv1×1
        self.conv3 = nn.Conv2d(num_channel, num_channel, 5, 1, 2, bias=True, groups=num_channel)  #--20231024--conv5×5

    def forward(self, mask_shift):  #--20231024--mask_shift(bs,nC,H,Wp)

        [bs, nC, H, Wp] = mask_shift.shape
        mask_shift = self.conv1(mask_shift)
        mask_shift_1 = self.conv2(mask_shift)
        mask_shift_1 = self.conv3(mask_shift_1)
        mask_shift_1 = torch.sigmoid(mask_shift_1)
        res = mask_shift * mask_shift_1
        mask_shift = res + mask_shift

        #--20231024--shift back operation
        [bs, nC, H, Wp] = mask_shift.shape  #--20231024--mask_shift(bs,nC,H,W')
        down_sample = 256 // H
        step = float(2) / float(down_sample * down_sample)
        for i in range(nC):
            mask_shift[:, i, :, : H] = mask_shift[:, i, :, int(step * i) : int(step * i) + H]
        # --20231024--shift back operation

        mask_output = mask_shift[:, :, :, :H]
        return mask_output  #--20231024--mask_output(bs,nC,H,W)
"""

class MixAttention(nn.Module):  #--20231218--2022MixFormer  local attn由swin transformer改为dauhst的local attn  #--20240101--空间attn通道数加倍,加线性环节  #--20240102--改进交叉方式，光谱attn中后端linear用DW替换
    def __init__(self, num_channel, num_head, win_size, num_win, resol):  #--20231216--num_channel特征通道数量，num_head注意力头数量，win_size窗口大小，num_win分割块数量(1维)，resol为空间分辨率
        super(MixAttention, self).__init__()
        self.num_channel = num_channel
        self.num_head = num_head
        self.win_size = win_size
        self.num_win = num_win
        self.resol = resol

        self.pos_emb = nn.Parameter(torch.Tensor(1, 1, num_head, resol // num_win * resol // num_win, resol // num_win * resol // num_win))  #--20231031--(1,1,nh,hw,hw)
        trunc_normal_(self.pos_emb)

        # Dwconv Branch
        self.proj_cnn = nn.Linear(num_channel, num_channel)  #--20231216--nC Dwconv Branch
        self.proj_cnn_norm = nn.LayerNorm(num_channel)
        self.to_qkv_cnn = nn.Linear(num_channel, 28 * num_head * 3, bias=False)  #--20231219--(_,_,_,nC)->(_,_,_,28*nh*3)
        # self.softmax_cnn = nn.Softmax(dim=-1)
        self.sigma_cnn = nn.Parameter(torch.ones(num_win * num_win, num_head, 1, 1))  #--20231219--(p2,nh,1,1)
        self.to_out_cnn = nn.Linear(28 * num_head, num_channel)  #--20231219--(_,_,_,28*nh)->(_,_,_,nC)
        self.Pos_Emb_cnn = nn.Sequential(nn.Conv2d(num_channel, num_channel, 3, 1, 1, bias=False, groups=num_channel),  #--20231024--DW conv3×3
                                         nn.GELU(),
                                         nn.Conv2d(num_channel, num_channel, 3, 1, 1, bias=False, groups=num_channel)  #--20231024--DW conv3×3
                                         )
        # self.Mask_Guide = MGM(num_channel)
        self.dw_cnn = nn.Conv2d(num_channel, num_channel, 3, 1, 1, bias=True, groups=num_channel)  #--20240102--新增
        self.projection = nn.Conv2d(num_channel, num_channel // 2, 1, 1, 0, bias=True)
        # self.conv_norm = nn.BatchNorm2d(num_channel // 2)

        # window-attention branch
        self.proj_attn = nn.Linear(num_channel, num_channel)  #--20240101--nC  Attention Branch
        self.proj_attn_norm = nn.LayerNorm(num_channel)  #--20240101--
        self.to_qkv = nn.Linear(num_channel, 28 * num_head * 3, bias=False)  #--20240101--(_,_,_,nC)->(_,_,_,28*nh*3)
        # self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(28 * num_head, num_channel)  #--20240101--(_,_,_,28*nh)->(_,_,_,nC)
        # self.attn_norm = nn.LayerNorm(num_channel // 2)
        self.projection_attn = nn.Conv2d(num_channel, num_channel // 2, 1, 1, 0, bias=True)  # --20240101--与DW通道一致而新增

        # Interaction
        self.channel_interaction = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                 nn.Conv2d(num_channel, num_channel // 4, 1, 1, 0, bias=True),
                                                 # nn.BatchNorm2d(num_channel // 8),
                                                 nn.GELU(),
                                                 nn.Conv2d(num_channel // 4, num_channel, 1, 1, 0, bias=True)
                                                 )  #--20240101--Channel Interation

        self.spatial_interaction = nn.Sequential(nn.Conv2d(num_channel, num_channel // 8, 1, 1, 0, bias=True),
                                                 # nn.BatchNorm2d(num_channel // 16),
                                                 nn.GELU(),
                                                 nn.Conv2d(num_channel // 8, 1, 1, 1, 0, bias=True)  # 最终空间信息输出通道为1
                                                 )  #--20240101--Spatial Interation

        # final projection
        self.proj = nn.Linear(num_channel, num_channel)  #--20240101--


    def forward(self, x, H, W):  #--20231216--x(bs*ph*pw,hw,nC),mask_input(bs,nC,H,Wp)
        # Dwconv Branch
        x_cnn = self.proj_cnn_norm(self.proj_cnn(x))  #--20231216--x(bs*ph*pw,hw,nC)->(bs*ph*pw,hw,nC) Linear+LayerNorm
        # x_cnn = window_reverse2(x_cnn, self.win_size, H, W)  #--20231216--(bs*ph*pw,hw,nC)->(bs,nC,H,W)
        qkv = self.to_qkv_cnn(x_cnn)  #--20231219--(bs*ph*pw,hw,nC)->(bs*ph*pw,hw,28*nh*3)
        q_inp, k_inp, v_inp = qkv.chunk(3, dim=-1)  #--20231219--q(bs*ph*pw,hw,28*nh),k(bs*ph*pw,hw,28*nh),v(bs*ph*pw,hw,28*nh)
        q = rearrange(q_inp, 'bsp2 hw (dh nh) -> bsp2 nh hw dh', nh=self.num_head)  #--20231027--(bs*p2,h*w,dh*nh)->(bs*p2,nh,h*w,dh)，为多头注意力实现
        k = rearrange(k_inp, 'bsp2 hw (dh nh) -> bsp2 nh hw dh', nh=self.num_head)  #--20231027--(bs*p2,h*w,dh*nh)->(bs*p2,nh,h*w,dh)，为多头注意力实现
        v = rearrange(v_inp, 'bsp2 hw (dh nh) -> bsp2 nh hw dh', nh=self.num_head)  #--20231027--(bs*p2,h*w,dh*nh)->(bs*p2,nh,h*w,dh)，为多头注意力实现
        # Mask_Attention = self.Mask_Guide(mask_input)  #--20231024--mask_input(bs,nC,H,Wp)->mask_output(bs,nC,H,W)
        # Mask_Attention = (Mask_Attention.permute(0, 2, 3, 1))  #--20231024--(bs,nC,H,W)->(bs,H,W,nC)
        # Mask_Attention = rearrange(Mask_Attention, 'bs (ph h) (pw w) (dh nh) -> (bs ph pw) nh (h w) dh', ph=self.num_win, pw=self.num_win, nh=self.num_head)  #--20231024--(bs,H,W,dh*nh)->(bsp2,nh,hw,dh)，为多头注意力实现
        # v = v * Mask_Attention  #--20231024--(bs*p2,nh,h*w,dh)
        q = F.normalize(q, dim=-2, p=2)  #--20231027--(bs*p2,nh,h*w,dh)
        k = F.normalize(k, dim=-2, p=2)  #--20231027--(bs*p2,nh,h*w,dh)
        Attention = torch.matmul(k.transpose(-2, -1), q)  #--20231027--(KT*Q) (bs*p2,nh,dh,dh)
        Attention = rearrange(Attention, '(bs p2) nh dh1 dh2 -> bs p2 nh dh1 dh2', p2=self.num_win * self.num_win)  #--20231027--(bs,p2,nh,dh,dh)
        Attention = Attention * self.sigma_cnn  #--20231027--(bs,p2,nh,dh,dh) * (p2,nh,1,1) = (bs,p2,nh,dh,dh)
        Attention = rearrange(Attention, 'bs p2 nh dh1 dh2 -> (bs p2) nh dh1 dh2')  #--20231027--(bs*p2,nh,dh,dh)
        Attention = F.softmax(Attention, dim=-2)  #--20231027--后续矩阵相乘V*Attention，所以softmax操作维度是-2  (bs*p2,nh,dh,dh)
        x_cnn = torch.matmul(v, Attention)  # --20231027--(bs*p2,nh,h*w,dh)
        x_cnn = rearrange(x_cnn, '(bs ph pw) nh (h w) dh -> bs (ph h pw w) (dh nh)', ph=self.num_win, pw=self.num_win, h=self.win_size, w=self.win_size)  #--20231027--(bs*p2,nh,h*w,dh)->(bs,H*W,dh*nh)
        x_cnn = rearrange(x_cnn, 'bs (h w) nc -> bs h w nc', h=H)  #--20231024--(bs,H*W,nC)->(bs,H,W,nC)
        x_cnn = self.to_out_cnn(x_cnn)  #--20231024--(bs,H,W,nC)->(bs,H,W,nC)
        x_cnn = x_cnn.permute(0, 3, 1, 2)  #--20231216--(bs,nC,H,W)

        v_inp = rearrange(v_inp, '(bs ph pw) (h w) dhnh -> bs (ph h) (pw w) dhnh', ph=self.num_win,  pw=self.num_win, h=self.win_size)  #--20231024--(bs*ph*pw,hw,28*nh)->(bs,H,W,28*nh)
        v_inp = v_inp.permute(0, 3, 1, 2)  #--20231024--(bs,dh*nh,H,W)
        x_p = self.Pos_Emb_cnn(v_inp)  #--20231024--(bs,dh*nh,H,W)->(bs,nC,H,W)
        x_cnn = x_cnn + x_p  #--20231216--(bs,nC,H,W)

        channel_interaction = self.channel_interaction(x_cnn)  #--20240101--(bs,nC,H,W)->(bs,nC,1,1)


        # window-attention branch
        x_atten = self.proj_attn_norm(self.proj_attn(x))  #--20240101--x(bs*ph*pw,hw,nC)->(bs*ph*pw,hw,nC) Linear+LayerNorm
        x_atten = window_reverse2(x_atten, self.win_size, H, W)  #--20240101--(bs*ph*pw,hw,nC)->(bs,nC,H,W)
        qkv = self.to_qkv(x_atten.permute(0, 2, 3, 1))  #--20240101--(bs,nC,H,W)->(bs,H,W,28*nh*3)
        q, k, v = qkv.chunk(3, dim=-1)  #--20240101--q(bs,H,W,28*nh),k(bs,H,W,28*nh),v(bs,H,W,28*nh)
        # x_cnn2v = torch.sigmoid(channel_interaction).permute(0, 2, 3, 1)  #--20240101--(bs,nC,1,1)->(bs,1,1,nC)
        # v = v * x_cnn2v  #--20240101--v(bs,H,W,nC)

        q, k, v = map(lambda t: rearrange(t, 'bs (ph h) (pw w) cnh -> bs (ph pw) (h w) cnh', ph=self.num_win, pw=self.num_win), (q, k, v))  #--20240101--(bs,H,W,28*nh)->(bs,ph*pw,h*w,28*nh)
        q, k, v = map(lambda t: rearrange(t, 'bs phpw hw (nh c) -> bs phpw nh hw c', nh=self.num_head), (q, k, v))  #--20240101--(bs,ph*pw,h*w,28*nh)->(bs,ph*pw,nh,h*w,28)
        q = q * (28 ** -0.5)  #--20240101--
        sim = einsum('b n h i d, b n h j d -> b n h i j', q, k)  #--20231031--(bs,ph*pw,nh,h*w,28),(bs,ph*pw,nh,h*w,28)->(bs,ph*pw,nh,h*w,h*w)
        sim = sim + self.pos_emb  # --20231031--(bs,ph*pw,nh,h*w,h*w)+(1,1,nh,hw,hw)
        attn = sim.softmax(dim=-1)
        out = einsum('b n h i j, b n h j d -> b n h i d', attn, v)  #--20231031--(bs,ph*pw,nh,h*w,h*w)->(bs,ph*pw,nh,h*w,28)
        out = rearrange(out, 'bs phpw nh hw c -> bs phpw hw (nh c)')  #--20231031--(bs,ph*pw,nh,h*w,28)->(bs,ph*pw,h*w,28*nh)
        out = self.to_out(out)  #--20240101--(bs,ph*pw,h*w,28*nh)->(bs,ph*pw,h*w,nC)
        out = rearrange(out, 'bs (ph pw) (h w) c -> bs c (ph h) (pw w)', ph=self.num_win, h=self.resol // self.num_win)  #--20240101--(bs,ph*pw,h*w,nC)->(bs,nC,H,W)
        out = out.permute(0, 2, 3, 1)  #--20240101--(bs,H,W,nC)

        spatial_interaction = self.spatial_interaction(out.permute(0, 3, 1, 2))  #--20240101--(bs,nC,H,W)->(bs,1,H,W)



        # cross effect
        x_cnn = torch.sigmoid(spatial_interaction) * x_cnn  #--20231216--(bs,1,H,W)*(bs,nC,H,W)->(bs,nC,H,W)
        x_cnn = self.dw_cnn(x_cnn)  #--20240102--(bs,nC,H,W)
        x_cnn = self.projection(x_cnn)  # --20231216--(bs,nC,H,W)->(bs,nC/2,H,W)

        # x_atten = self.attn_norm(out)  #--20231217--(bs,H,W,nC/2)
        out = (torch.sigmoid(channel_interaction).permute(0, 2, 3, 1)) * out  #--20240102--(bs,nC,1,1)->(bs,1,1,nC)
        out = self.projection_attn(out.permute(0, 3, 1, 2))  #--20240101--(bs,nC,H,W)->(bs,nC/2,H,W)


        x = torch.cat([out.permute(0, 2, 3, 1), x_cnn.permute(0, 2, 3, 1)], dim=3)  #--20240101--(bs,H,W,nC)
        x = self.proj(x)  #--20231217--(bs,H,W,nC)->(bs,H,W,nC)

        x = window_partition2(x.permute(0, 3, 1, 2), self.win_size)  #--20231217--(bs,H,W,nC)->(bs*ph*pw,hw,nC)
        return x  # --20231216--(bs*ph*pw,hw,nC)


class MixBlock(nn.Module):  #--20231216--MixAttention+FFN
    def __init__(self, num_channel, num_head, num_win, win_size, resol):  #--20231216--num_channel特征通道数量，num_head注意力头数量，win_size窗口大小，num_win分割块数量(1维)，resol为空间分辨率
        super(MixBlock, self).__init__()
        self.num_channel = num_channel
        self.num_head = num_head
        self.win_size = win_size

        self.norm1 = nn.LayerNorm(num_channel)
        self.attn = MixAttention(num_channel, num_head, win_size, num_win, resol)
        self.norm2 = nn.LayerNorm(num_channel)

        self.mlp = nn.Sequential(nn.Conv2d(num_channel, num_channel * 4, 1, 1, 0, bias=False),  #--20231217--用patch attn替换了
                                 nn.GELU(),
                                 nn.Conv2d(num_channel * 4, num_channel * 4, 3, 1, 1, bias=False, groups=num_channel * 4),
                                 nn.GELU(),
                                 nn.Conv2d(num_channel * 4, num_channel, 1, 1, 0, bias=False)
                                 )

    def forward(self, x):  #--20231216--(bs,H,W,nC)
        bs, H, W, nC = x.shape  #--20231216--(bs,H,W,nC)
        shortcut = x
        x = self.norm1(x)

        shifted_x = x  #--20231216--(bs,H,W,nC)
        x_windows = window_partition(shifted_x, self.win_size)  #--20231216--(bs,H,W,nC)->(bs*ph*pw,h,w,nC)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, nC)  #--20231216--(bs*ph*pw,h,w,nC)->(bs*ph*pw,h*w,nC)
        attn_windows = self.attn(x_windows, H, W)  #--20231216--(bs*ph*pw,h*w,nC)->(bs*ph*pw,h*w,nC)
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, nC)  #--20231216--(bs*ph*pw,h*w,nC)->(bs*ph*pw,h,w,nC)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  #--20231216--(bs*ph*pw,h,w,nC)->(bs,H,W,nC)
        x = shifted_x  #--20231216--(bs,H,W,nC)

        x = shortcut + x  #--20231216--(bs,H,W,nC)
        x = x + (self.mlp((self.norm2(x)).permute(0, 3, 1, 2))).permute(0, 2, 3, 1)  #--20231216--(bs,H,W,nC)

        return x  #--20231216--(bs,H,W,nC)




class Run(nn.Module):
    def __init__(self):
        super(Run, self).__init__()

        # Fusion physical mask and shifted measurement
        self.fusion = nn.Conv2d(56, 28, 1, 1, 0, bias=False)

        # Sparsity Estimator
        # self.fea_extr = Sparsity_Estimator()
        self.upsamp21_12 = nn.ConvTranspose2d(56, 28, 2, 2, bias=True)  #--20240114--
        self.dnsamp12_22 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)  #--20240114--
        self.upsamp31_13 = nn.ConvTranspose2d(112, 28, 4, 4, bias=True)  #--20240114--
        self.dnsamp11_22 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)  #--20240114--
        self.upsamp21_13 = nn.ConvTranspose2d(56, 28, 2, 2, bias=True)  #--20240114--

        self.linear12 = nn.Conv2d(2 * 28, 28, 1, 1, 0, bias=True)  #--20240114--
        self.linear22 = nn.Conv2d(4 * 56, 56, 1, 1, 0, bias=True)  #--20240114--
        self.linear13 = nn.Conv2d(5 * 28, 28, 1, 1, 0, bias=True)  #--20240114--



        # Encoder
        self.encoder_layers_1 = nn.Sequential(MixBlock(28, 1, 256 // 16, 16, 256),  #--20231126--patch size 8->16
                                              MixBlock(28, 1, 256 // 16, 16, 256)
                                              )
        self.downsamp_layers_1 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)

        self.encoder_layers_2 = nn.Sequential(MixBlock(56, 2, 128 // 8, 8, 128),
                                              MixBlock(56, 2, 128 // 8, 8, 128),
                                              MixBlock(56, 2, 128 // 8, 8, 128),
                                              MixBlock(56, 2, 128 // 8, 8, 128)
                                              )
        self.downsamp_layers_2 = nn.Conv2d(56, 112, 4, 2, 1, bias=False)

        # Bottleneck
        self.bottleneck = nn.Sequential(MixBlock(112, 4, 64 // 8, 8, 64),
                                        MixBlock(112, 4, 64 // 8, 8, 64),
                                        MixBlock(112, 4, 64 // 8, 8, 64),
                                        MixBlock(112, 4, 64 // 8, 8, 64),
                                        MixBlock(112, 4, 64 // 8, 8, 64),
                                        MixBlock(112, 4, 64 // 8, 8, 64)
                                        )

        # Decoder
        self.upsamp_layers_2 = nn.ConvTranspose2d(112, 56, 2, 2, bias=True)
        self.decoder_layers_2 = nn.Sequential(MixBlock(56, 2, 128 // 8, 8, 128),
                                              MixBlock(56, 2, 128 // 8, 8, 128),
                                              MixBlock(56, 2, 128 // 8, 8, 128),
                                              MixBlock(56, 2, 128 // 8, 8, 128)
                                              )

        self.upsamp_layers_1 = nn.ConvTranspose2d(56, 28, 2, 2, bias=True)
        self.decoder_layers_1 = nn.Sequential(MixBlock(28, 1, 256 // 16, 16, 256),  #--20231126--patch size 8->16
                                              MixBlock(28, 1, 256 // 16, 16, 256)
                                              )


        # Intermediate
        self.interme_layers_1 = nn.Sequential(MixBlock(28, 1, 256 // 16, 16, 256),  #--20240114--
                                              MixBlock(28, 1, 256 // 16, 16, 256)
                                              )


        # Output projection
        self.out_proj = nn.Conv2d(28, 28, 3, 1, 1, bias=False)

    def forward(self, xmask):  #--20231123--x(bs,nC,H,W),mask(bs,nC,H,W)


        # Fusion
        x = self.fusion(xmask)  #--20231123--x(bs,nC,H,W)

        # Feature Extraction
        # feature = self.fea_extr(x)  #--20231123--(bs,nC,H,W)->(bs,nC,H,W)
        feature = x  #--20231125--移除Sparsity_Estimater部分

        # Encoder
        fea_encoder = []
        feature = self.encoder_layers_1(feature.permute(0, 2, 3, 1))  #--20231123--(bs,nC,H,W)->(bs,H,W,nC)
        fea_encoder.append(feature.permute(0, 3, 1, 2))  #--20231123--(bs,nC,H,W)
        feature = self.downsamp_layers_1(feature.permute(0, 3, 1, 2))  #--20231123--(bs,nC,H,W)->(bs,nC,H,W)

        feature = self.encoder_layers_2(feature.permute(0, 2, 3, 1))  #--20231123--(bs,nC,H,W)->(bs,H,W,nC)
        fea_encoder.append(feature.permute(0, 3, 1, 2))  #--20231123--(bs,nC,H,W)
        feature = self.downsamp_layers_2(feature.permute(0, 3, 1, 2))  #--20231123--(bs,nC,H,W)->(bs,nC,H,W)

        # Intermediate
        tmp1 = self.upsamp21_12(fea_encoder[1])  #--20240115--(bs,nC,H,W)
        interfea = torch.cat([fea_encoder[0], tmp1], dim=1)  #--20240114--(bs,nC,H,W)
        interfea = self.linear12(interfea)  #--20240114--(bs,nC,H,W)
        interfea = self.interme_layers_1(interfea.permute(0, 2, 3, 1))   #--20240114--(bs,H,W,nC)


        # Bottleneck
        feature = self.bottleneck(feature.permute(0, 2, 3, 1))  #--20231123--(bs,nC,H,W)->(bs,H,W,nC)
        fea_encoder.append(feature.permute(0, 3, 1, 2))  #--20240114--(bs,nC,H,W)


        # Decoder
        feature = self.upsamp_layers_2(feature.permute(0, 3, 1, 2))  #--20231123--(bs,H,W,nC)->(bs,nC,H,W)
        tmp1 = self.dnsamp11_22(fea_encoder[0])  #--20240114--(bs,nC,H,W)
        tmp2 = self.dnsamp12_22(interfea.permute(0, 3, 1, 2))  #--20240114--(bs,nC,H,W)
        feature = torch.cat([feature, fea_encoder[1], tmp1, tmp2], dim=1)  #--20240114--(bs,nC,H,W)
        feature = self.linear22(feature)  #--20240114--(bs,56,128,128)
        feature = self.decoder_layers_2(feature.permute(0, 2, 3, 1))  #--20231123--(bs,nC,H,W)->(bs,H,W,nC)

        feature = self.upsamp_layers_1(feature.permute(0, 3, 1, 2))  #--20231123--(bs,H,W,nC)->(bs,nC,H,W)
        tmp1 = self.upsamp31_13(fea_encoder[2])  #--20240114--(bs,nC,H,W)
        tmp2 = self.upsamp21_13(fea_encoder[1])  #--20240114--(bs,nC,H,W)
        feature = torch.cat([feature, tmp1, tmp2, interfea.permute(0, 3, 1, 2), fea_encoder[0]], dim=1)  #--20240114--(bs,nC,H,W)
        feature = self.linear13(feature)  #--20240114--(bs,28,256,256)
        feature = self.decoder_layers_1(feature.permute(0, 2, 3, 1))  #--20231123--(bs,nC,H,W)->(bs,H,W,nC)

        # Output projection
        feature = self.out_proj(feature.permute(0, 3, 1, 2))  #--20231123--(bs,H,W,nC)->(bs,nC,H,W)
        out = feature + x  #--20231123--(bs,nC,H,W)

        return out  #--20231123--(bs,nC,H,W)





import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import LongTensor, Tensor   
import warnings   
import math   
from torch import einsum   


def window_partition(x, window_size: int):   
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)   
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)   
    return windows   

def window_partition2(x, window_size):   
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)   
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)   
    return windows   

def window_reverse(windows, window_size: int, H: int, W: int):   
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)   
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)   
    return x   

def window_reverse2(windows, window_size, H: int, W: int):   
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)   
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)   
    return x   


def _no_grad_trunc_normal_(tensor, mean, std, a, b):   
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
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):   
     
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

"""
class MGM(nn.Module):   
    def __init__(self, num_channel):   
        super(MGM, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, 1, 1, 0, bias=True)   
        self.conv2 = nn.Conv2d(num_channel, num_channel, 1, 1, 0, bias=True)   
        self.conv3 = nn.Conv2d(num_channel, num_channel, 5, 1, 2, bias=True, groups=num_channel)   

    def forward(self, mask_shift):   

        [bs, nC, H, Wp] = mask_shift.shape
        mask_shift = self.conv1(mask_shift)
        mask_shift_1 = self.conv2(mask_shift)
        mask_shift_1 = self.conv3(mask_shift_1)
        mask_shift_1 = torch.sigmoid(mask_shift_1)
        res = mask_shift * mask_shift_1
        mask_shift = res + mask_shift

         
        [bs, nC, H, Wp] = mask_shift.shape   
        down_sample = 256 // H
        step = float(2) / float(down_sample * down_sample)
        for i in range(nC):
            mask_shift[:, i, :, : H] = mask_shift[:, i, :, int(step * i) : int(step * i) + H]
         

        mask_output = mask_shift[:, :, :, :H]
        return mask_output   
"""

class MixAttention(nn.Module):   
    def __init__(self, num_channel, num_head, win_size, num_win, resol):   
        super(MixAttention, self).__init__()
        self.num_channel = num_channel
        self.num_head = num_head
        self.win_size = win_size
        self.num_win = num_win
        self.resol = resol

        self.pos_emb = nn.Parameter(torch.Tensor(1, 1, num_head, resol // num_win * resol // num_win, resol // num_win * resol // num_win))   
        trunc_normal_(self.pos_emb)

         
        self.proj_cnn = nn.Linear(num_channel, num_channel)   
        self.proj_cnn_norm = nn.LayerNorm(num_channel)
        self.to_qkv_cnn = nn.Linear(num_channel, 28 * num_head * 3, bias=False)   
         
        self.sigma_cnn = nn.Parameter(torch.ones(num_win * num_win, num_head, 1, 1))   
        self.to_out_cnn = nn.Linear(28 * num_head, num_channel)   
        self.Pos_Emb_cnn = nn.Sequential(nn.Conv2d(num_channel, num_channel, 3, 1, 1, bias=False, groups=num_channel),   
                                         nn.GELU(),
                                         nn.Conv2d(num_channel, num_channel, 3, 1, 1, bias=False, groups=num_channel)   
                                         )
         
        self.dw_cnn = nn.Conv2d(num_channel, num_channel, 3, 1, 1, bias=True, groups=num_channel)   
        self.projection = nn.Conv2d(num_channel, num_channel // 2, 1, 1, 0, bias=True)
         

         
        self.proj_attn = nn.Linear(num_channel, num_channel)   
        self.proj_attn_norm = nn.LayerNorm(num_channel)   
        self.to_qkv = nn.Linear(num_channel, 28 * num_head * 3, bias=False)   
         
        self.to_out = nn.Linear(28 * num_head, num_channel)   
         
        self.projection_attn = nn.Conv2d(num_channel, num_channel // 2, 1, 1, 0, bias=True)   

         
        self.channel_interaction = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                 nn.Conv2d(num_channel, num_channel // 4, 1, 1, 0, bias=True),
                                                  
                                                 nn.GELU(),
                                                 nn.Conv2d(num_channel // 4, num_channel, 1, 1, 0, bias=True)
                                                 )   

        self.spatial_interaction = nn.Sequential(nn.Conv2d(num_channel, num_channel // 8, 1, 1, 0, bias=True),
                                                  
                                                 nn.GELU(),
                                                 nn.Conv2d(num_channel // 8, 1, 1, 1, 0, bias=True)   
                                                 )   

         
        self.proj = nn.Linear(num_channel, num_channel)   


    def forward(self, x, H, W):   
         
        x_cnn = self.proj_cnn_norm(self.proj_cnn(x))   
         
        qkv = self.to_qkv_cnn(x_cnn)   
        q_inp, k_inp, v_inp = qkv.chunk(3, dim=-1)   
        q = rearrange(q_inp, 'bsp2 hw (dh nh) -> bsp2 nh hw dh', nh=self.num_head)   
        k = rearrange(k_inp, 'bsp2 hw (dh nh) -> bsp2 nh hw dh', nh=self.num_head)   
        v = rearrange(v_inp, 'bsp2 hw (dh nh) -> bsp2 nh hw dh', nh=self.num_head)   
         
         
         
         
        q = F.normalize(q, dim=-2, p=2)   
        k = F.normalize(k, dim=-2, p=2)   
        Attention = torch.matmul(k.transpose(-2, -1), q)   
        Attention = rearrange(Attention, '(bs p2) nh dh1 dh2 -> bs p2 nh dh1 dh2', p2=self.num_win * self.num_win)   
        Attention = Attention * self.sigma_cnn   
        Attention = rearrange(Attention, 'bs p2 nh dh1 dh2 -> (bs p2) nh dh1 dh2')   
        Attention = F.softmax(Attention, dim=-2)   
        x_cnn = torch.matmul(v, Attention)   
        x_cnn = rearrange(x_cnn, '(bs ph pw) nh (h w) dh -> bs (ph h pw w) (dh nh)', ph=self.num_win, pw=self.num_win, h=self.win_size, w=self.win_size)   
        x_cnn = rearrange(x_cnn, 'bs (h w) nc -> bs h w nc', h=H)   
        x_cnn = self.to_out_cnn(x_cnn)   
        x_cnn = x_cnn.permute(0, 3, 1, 2)   

        v_inp = rearrange(v_inp, '(bs ph pw) (h w) dhnh -> bs (ph h) (pw w) dhnh', ph=self.num_win,  pw=self.num_win, h=self.win_size)   
        v_inp = v_inp.permute(0, 3, 1, 2)   
        x_p = self.Pos_Emb_cnn(v_inp)   
        x_cnn = x_cnn + x_p   

        channel_interaction = self.channel_interaction(x_cnn)   


         
        x_atten = self.proj_attn_norm(self.proj_attn(x))   
        x_atten = window_reverse2(x_atten, self.win_size, H, W)   
        qkv = self.to_qkv(x_atten.permute(0, 2, 3, 1))   
        q, k, v = qkv.chunk(3, dim=-1)   
         
         

        q, k, v = map(lambda t: rearrange(t, 'bs (ph h) (pw w) cnh -> bs (ph pw) (h w) cnh', ph=self.num_win, pw=self.num_win), (q, k, v))   
        q, k, v = map(lambda t: rearrange(t, 'bs phpw hw (nh c) -> bs phpw nh hw c', nh=self.num_head), (q, k, v))   
        q = q * (28 ** -0.5)   
        sim = einsum('b n h i d, b n h j d -> b n h i j', q, k)   
        sim = sim + self.pos_emb   
        attn = sim.softmax(dim=-1)
        out = einsum('b n h i j, b n h j d -> b n h i d', attn, v)   
        out = rearrange(out, 'bs phpw nh hw c -> bs phpw hw (nh c)')   
        out = self.to_out(out)   
        out = rearrange(out, 'bs (ph pw) (h w) c -> bs c (ph h) (pw w)', ph=self.num_win, h=self.resol // self.num_win)   
        out = out.permute(0, 2, 3, 1)   

        spatial_interaction = self.spatial_interaction(out.permute(0, 3, 1, 2))   



         
        x_cnn = torch.sigmoid(spatial_interaction) * x_cnn   
        x_cnn = self.dw_cnn(x_cnn)   
        x_cnn = self.projection(x_cnn)   

         
        out = (torch.sigmoid(channel_interaction).permute(0, 2, 3, 1)) * out   
        out = self.projection_attn(out.permute(0, 3, 1, 2))   


        x = torch.cat([out.permute(0, 2, 3, 1), x_cnn.permute(0, 2, 3, 1)], dim=3)   
        x = self.proj(x)   

        x = window_partition2(x.permute(0, 3, 1, 2), self.win_size)   
        return x   


class MixBlock(nn.Module):   
    def __init__(self, num_channel, num_head, num_win, win_size, resol):   
        super(MixBlock, self).__init__()
        self.num_channel = num_channel
        self.num_head = num_head
        self.win_size = win_size

        self.norm1 = nn.LayerNorm(num_channel)
        self.attn = MixAttention(num_channel, num_head, win_size, num_win, resol)
        self.norm2 = nn.LayerNorm(num_channel)

        self.mlp = nn.Sequential(nn.Conv2d(num_channel, num_channel * 4, 1, 1, 0, bias=False),   
                                 nn.GELU(),
                                 nn.Conv2d(num_channel * 4, num_channel * 4, 3, 1, 1, bias=False, groups=num_channel * 4),
                                 nn.GELU(),
                                 nn.Conv2d(num_channel * 4, num_channel, 1, 1, 0, bias=False)
                                 )

    def forward(self, x):   
        bs, H, W, nC = x.shape   
        shortcut = x
        x = self.norm1(x)

        shifted_x = x   
        x_windows = window_partition(shifted_x, self.win_size)   
        x_windows = x_windows.view(-1, self.win_size * self.win_size, nC)   
        attn_windows = self.attn(x_windows, H, W)   
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, nC)   
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)   
        x = shifted_x   

        x = shortcut + x   
        x = x + (self.mlp((self.norm2(x)).permute(0, 3, 1, 2))).permute(0, 2, 3, 1)   

        return x   




class Run(nn.Module):
    def __init__(self):
        super(Run, self).__init__()

         
        self.fusion = nn.Conv2d(56, 28, 1, 1, 0, bias=False)

         
         
        self.upsamp21_12 = nn.ConvTranspose2d(56, 28, 2, 2, bias=True)   
        self.dnsamp12_22 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)   
        self.upsamp31_13 = nn.ConvTranspose2d(112, 28, 4, 4, bias=True)   
        self.dnsamp11_22 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)   
        self.upsamp21_13 = nn.ConvTranspose2d(56, 28, 2, 2, bias=True)   

        self.linear12 = nn.Conv2d(2 * 28, 28, 1, 1, 0, bias=True)   
        self.linear22 = nn.Conv2d(4 * 56, 56, 1, 1, 0, bias=True)   
        self.linear13 = nn.Conv2d(5 * 28, 28, 1, 1, 0, bias=True)   



         
        self.encoder_layers_1 = nn.Sequential(MixBlock(28, 1, 256 // 16, 16, 256),   
                                              MixBlock(28, 1, 256 // 16, 16, 256)
                                              )
        self.downsamp_layers_1 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)

        self.encoder_layers_2 = nn.Sequential(MixBlock(56, 2, 128 // 8, 8, 128),
                                              MixBlock(56, 2, 128 // 8, 8, 128)
                                              )
        self.downsamp_layers_2 = nn.Conv2d(56, 112, 4, 2, 1, bias=False)

         
        self.bottleneck = nn.Sequential(MixBlock(112, 4, 64 // 8, 8, 64),
                                        MixBlock(112, 4, 64 // 8, 8, 64)
                                        )

         
        self.upsamp_layers_2 = nn.ConvTranspose2d(112, 56, 2, 2, bias=True)
        self.decoder_layers_2 = nn.Sequential(MixBlock(56, 2, 128 // 8, 8, 128),
                                              MixBlock(56, 2, 128 // 8, 8, 128)
                                              )

        self.upsamp_layers_1 = nn.ConvTranspose2d(56, 28, 2, 2, bias=True)
        self.decoder_layers_1 = nn.Sequential(MixBlock(28, 1, 256 // 16, 16, 256),   
                                              MixBlock(28, 1, 256 // 16, 16, 256)
                                              )


         
        self.interme_layers_1 = nn.Sequential(MixBlock(28, 1, 256 // 16, 16, 256),   
                                              MixBlock(28, 1, 256 // 16, 16, 256)
                                              )


         
        self.out_proj = nn.Conv2d(28, 28, 3, 1, 1, bias=False)

    def forward(self, xmask):   


         
        x = self.fusion(xmask)   

         
         
        feature = x   

         
        fea_encoder = []
        feature = self.encoder_layers_1(feature.permute(0, 2, 3, 1))   
        fea_encoder.append(feature.permute(0, 3, 1, 2))   
        feature = self.downsamp_layers_1(feature.permute(0, 3, 1, 2))   

        feature = self.encoder_layers_2(feature.permute(0, 2, 3, 1))   
        fea_encoder.append(feature.permute(0, 3, 1, 2))   
        feature = self.downsamp_layers_2(feature.permute(0, 3, 1, 2))   

         
        tmp1 = self.upsamp21_12(fea_encoder[1])   
        interfea = torch.cat([fea_encoder[0], tmp1], dim=1)   
        interfea = self.linear12(interfea)   
        interfea = self.interme_layers_1(interfea.permute(0, 2, 3, 1))    


         
        feature = self.bottleneck(feature.permute(0, 2, 3, 1))   
        fea_encoder.append(feature.permute(0, 3, 1, 2))   


         
        feature = self.upsamp_layers_2(feature.permute(0, 3, 1, 2))   
        tmp1 = self.dnsamp11_22(fea_encoder[0])   
        tmp2 = self.dnsamp12_22(interfea.permute(0, 3, 1, 2))   
        feature = torch.cat([feature, fea_encoder[1], tmp1, tmp2], dim=1)   
        feature = self.linear22(feature)   
        feature = self.decoder_layers_2(feature.permute(0, 2, 3, 1))   

        feature = self.upsamp_layers_1(feature.permute(0, 3, 1, 2))   
        tmp1 = self.upsamp31_13(fea_encoder[2])   
        tmp2 = self.upsamp21_13(fea_encoder[1])   
        feature = torch.cat([feature, tmp1, tmp2, interfea.permute(0, 3, 1, 2), fea_encoder[0]], dim=1)   
        feature = self.linear13(feature)   
        feature = self.decoder_layers_1(feature.permute(0, 2, 3, 1))   

         
        feature = self.out_proj(feature.permute(0, 3, 1, 2))   
        out = feature + x   

        return out   





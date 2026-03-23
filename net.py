import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from swt import TransformerBlock
import numbers
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1, padding=0)
        self.active0 = nn.LeakyReLU(0.1, inplace=True)
        self.active1 = nn.LeakyReLU(0.1, inplace=True)
        self.active2 = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        x_ = self.active0(self.conv0(x))
        x_ = self.active1(self.conv1(x_))
        x_ = self.active2(self.conv2(x_+x))
        return x_
   
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Spectral_Attention(nn.Module):
    def __init__(self, dim=32, expansion_factor=2):
        super(Spectral_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spectral_atten = nn.Sequential(
            nn.Linear(dim, dim//2, bias=False),
            nn.GELU(),
            nn.Linear(dim//2, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.spectral_atten(y).view(b, c, 1, 1)
        return x*y.expand_as(x)

class Prompt1(nn.Module):
    def __init__(self, dim, num_blocks=3):
        super(Prompt1, self).__init__()
        # num_blocks = 3
        heads = 8
        ffn_expansion_factor = 2
        self.conv0 = nn.Conv2d(dim+1, dim, 1, 1, 0)
        self.conv1 = nn.Conv2d(dim+1, dim, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim+1, dim, 1, 1, 0)
        
        self.process1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU())
        
        self.process2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU())
        
        self.process5 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU(),
            Spectral_Attention(dim),
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU())
        
        self.process3 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU())
        self.process4 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU())
        self.mlp1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.GELU(),
            # nn.Conv2d(dim//2, dim, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU(),
            # nn.Conv2d(dim, dim, 1, 1, 0),
            nn.Sigmoid()
        )
        
        kernel_size = 5
        self.avg_pool0 = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.sigmoid3 = nn.Sigmoid()

        # self.edge_conv = SpatialAttention(3)

        self.mst = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                            bias='False', LayerNorm_type='WithBias') for i in range(num_blocks)])

    def forward(self, x, pan_deg=None, ms_deg=None):
        # pan_deg = pan_deg*self.edge_conv(pan_deg)
        contrast = torch.abs(pan_deg - self.avg_pool0(pan_deg))
        pan_deg = pan_deg*self.sigmoid3(contrast)
        x_res = x.clone()
        b,c,H,W = x.shape
        x_res_fre = torch.fft.fft2(x_res, norm='backward')
        x_res_mag_image = torch.abs(x_res_fre)
        x_res_pha_image = torch.angle(x_res_fre)

        x_ms_fre = torch.fft.fft2(ms_deg, norm='backward')
        x_ms_mag_image = torch.abs(x_ms_fre)
        x_mag_image = self.conv0(torch.concat([x_res_mag_image,x_ms_mag_image], 1))
        x_mag_image = self.process1(x_mag_image)*self.sigmoid1(self.process5(x_mag_image))
        
        x_mag_image = x_mag_image + x_res_mag_image

        x_pan_fre = torch.fft.fft2(pan_deg, norm='backward')
        x_pan_pha_image = torch.angle(x_pan_fre)
        x_pha_image = self.conv1(torch.concat([x_res_pha_image,x_pan_pha_image], 1))
        x_pha_image = self.process2(x_pha_image) + x_res_pha_image


        x_mag_image_ = self.process3(self.mlp1(self.avg_pool1(x_pha_image)) * x_mag_image) + x_mag_image
        x_pha_image_ = self.process4(self.mlp2(self.avg_pool2(x_mag_image)) * x_pha_image) + x_pha_image

        real_image_enhanced = x_mag_image_ * torch.cos(x_pha_image_)
        imag_image_enhanced = x_mag_image_ * torch.sin(x_pha_image_)
        x = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W), norm='backward').real
        x = x+x_res
        x = self.mst(x)
        return x


class CloudPan(nn.Module):
    def __init__(self, in_bands=4, dim=32):
        super(CloudPan, self).__init__()
        self.shallow_conv = nn.Conv2d(in_channels=in_bands+1, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.last_conv = nn.Conv2d(in_channels=dim, out_channels=in_bands, kernel_size=3, stride=1, padding=1)
        self.shallow_res = ResBlock(dim)   # 这个地方参看别的Pan方法是如何提取浅层特征的

        self.prompt0 = Prompt1(dim=dim,num_blocks=1)
        self.prompt1 = Prompt1(dim=dim*2,num_blocks=2)
        self.prompt2 = Prompt1(dim=dim*4,num_blocks=2)
        self.prompt3 = Prompt1(dim=dim*2,num_blocks=2)
        self.prompt4 = Prompt1(dim=dim,num_blocks=1)
        self.down_x0 = nn.Conv2d(dim, dim*2, 4, 2, 1, bias=False)
        self.down_x1 = nn.Conv2d(dim*2, dim*4, 4, 2, 1, bias=False)
        self.up_x2 = nn.ConvTranspose2d(dim*4, dim*2, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.up_x3 = nn.ConvTranspose2d(dim*2, dim  , stride=2, kernel_size=2, padding=0, output_padding=0)

    def forward(self, lms, pan):
        x = torch.concat([pan, lms], 1)
        x = self.shallow_conv(x)
        x = self.shallow_res(x)
        pan_deg, ms_deg = pan, lms[:,-1:,:,:]

        pan_deg_1 = pan_deg[:,:,1::2,1::2]
        pan_deg_2 = pan_deg[:,:,2::4,2::4]
        ms_deg_1  = ms_deg[:,:,1::2,1::2]
        ms_deg_2  = ms_deg[:,:,2::4,2::4]

        x0 = self.prompt0(x, pan_deg, ms_deg)
        x1 = self.down_x0(x0)
        x1 = self.prompt1(x1, pan_deg_1, ms_deg_1)
        x2 = self.down_x1(x1)
        x2 = self.prompt2(x2, pan_deg_2, ms_deg_2)
        x3 = self.up_x2(x2) + x1
        x3 = self.prompt3(x3, pan_deg_1, ms_deg_1) 
        x4 = self.up_x3(x3) + x0
        x4 = self.prompt4(x4, pan_deg, ms_deg) 
        out = self.last_conv(x4)+lms
        return out

if __name__ == '__main__':
    from torchsummary import summary
    from thop import profile
    import time
    model = CloudPan(4, dim=16).cuda()
    summary(model, [(4,64,64), (1,64,64)], device='cuda')
    pan = torch.zeros([1,1,128,128]).cuda()
    lms = torch.zeros([1,4,128,128]).cuda()
    flop, para = profile(model, inputs=(lms, pan))
    print(flop/1000000000.)
    print(para)
    pan = torch.zeros([1,1,128,128]).cuda()
    lms = torch.zeros([1,4,128,128]).cuda()
    for i in range(100):
        start_time = time.time()
        hms = model(lms, pan)
        print(time.time()-start_time)
    print(hms.shape)
    print('Parameters number of modelE_MSI is ', sum(param.numel() for param in model.parameters()))

    # for name, module in model.named_children():
    #     num_params = sum(p.numel() for p in module.parameters())
    #     print(f"Module: {name}, Parameters: {num_params}")

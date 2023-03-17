import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri
from timm.models.layers import trunc_normal_
class SE_Block(nn.Module):
    def __init__(self, channel=64):
        super(SE_Block, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_sig = nn.Sequential(
                # nn.Linear(channel, channel, bias=False),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.Sigmoid(),
                # nn.Linear(channel, channel, bias=False),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        # y = y.permute(0,2,3,1) # b,1,1,c
        y = self.fc_sig(y)
        # y = y.permute(0,3,1,2).contiguous() # b,c,1,1
        return x * y


class Inference_Block(nn.Module):
    def __init__(self, channel,num_concat,num_feat=64):
        super(Inference_Block, self).__init__()
        in_channel = channel * (num_concat+1)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(num_feat, channel, 3, 1, 1)
        )

        self.lambda_ = nn.Parameter(torch.ones(1))
        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def norm(self, x,mean,std):
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        x = (x - mean) / std
        x = x.view(b, c, h, w)
        return x
    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        x = x * std + mean
        x = x.view(b, c, h, w)
        return x

    def soft_dc(self, x, mask,input_kspace,sens_maps):
        x = self.sens_expand(x,sens_maps)
        x = x - torch.where(mask, x - input_kspace, self.zero)* self.lambda_
        x = self.sens_reduce(x, sens_maps)
        return x
    
    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def complex_to_slice_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, x,mask,input_kspace,sens_maps,concat=None,mean=None,std=None):
        """
        x : b,2z,h,w
        mask: b,coil,z,h,w,2
        input_kspace : b,coil,z,h,w,2
        sens_maps : b,coil,z,h,w,2
        """

        if concat is None:
            inp = x.clone()
        elif len(concat)==1:
            inp = torch.cat((x,concat[0]),dim=1)
        else:
            concat = torch.cat(concat,dim=1)
            inp = torch.cat((x,concat),dim=1)
        x = self.conv1(inp) + x
        x = self.unnorm(x,mean,std)
        x = self.chan_complex_to_last_dim(x) ## to b, z, h, w, two
        x = self.soft_dc(x.unsqueeze(1),mask,input_kspace,sens_maps).squeeze(1)
        x = self.complex_to_slice_dim(x) ## to b, 2z, h, w
        x = self.norm(x,mean,std)
        return x
    
    

    
class Inference_Block_single(nn.Module):
    def __init__(self, channel,num_concat,num_feat=64,checkpoint=False):
        super(Inference_Block_single, self).__init__()
        in_channel = channel * (num_concat+1)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(num_feat, channel, 3, 1, 1)
        )

        self.lambda_ = nn.Parameter(torch.ones(1))
        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))
        self.apply(self._init_weights)
        self.checkpoint=checkpoint

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    
    def norm(self, x,mean,std):
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        x = (x - mean) / std
        x = x.view(b, c, h, w)
        return x
    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        x = x * std + mean
        x = x.view(b, c, h, w)
        return x
    
    def soft_dc(self, x, mask,input_kspace):
    
        x = fastmri.fft2c(x)
        x = x - torch.where(mask, x - input_kspace, self.zero)* self.lambda_
        x = fastmri.ifft2c(x)
        
        return x

    def complex_to_slice_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, x,mask,input_kspace,concat=None,mean=None,std=None):
        """
        x : b,2z,h,w
        mask: b,coil,z,h,w,2
        input_kspace : b,coil,z,h,w,2
        sens_maps : b,coil,z,h,w,2
        """

        if concat is None:
            inp = x.clone()
        elif len(concat)==1:
            inp = torch.cat((x,concat[0]),dim=1)
        else:
            concat = torch.cat(concat,dim=1)
            inp = torch.cat((x,concat),dim=1)
            
        if self.checkpoint:
            x = checkpoint.checkpoint(self.conv1,inp) + x
        else:
            x = self.conv1(inp) + x
        x = self.unnorm(x,mean,std)
        x = self.chan_complex_to_last_dim(x) ## to b, z, h, w, two
        x = self.soft_dc(x,mask,input_kspace)
        x = self.complex_to_slice_dim(x) ## to b, 2z, h, w
        x = self.norm(x,mean,std)

        return x

class DFCN(nn.Module):
    def __init__(self, slices):
        super(DFCN, self).__init__()
        self.SEB = SE_Block(channel=2*slices)
        self.infer1 = Inference_Block(channel=2*slices,num_concat=0)
        self.infer2 = Inference_Block(channel=2*slices,num_concat=0)
        self.infer3 = Inference_Block(channel=2*slices,num_concat=1)
        self.infer4 = Inference_Block(channel=2*slices,num_concat=2)
        self.infer5 = Inference_Block(channel=2*slices,num_concat=3)
        self.apply(self._init_weights)
    def complex_to_slice_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass

    def norm(self, x):
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1)
        std = x.std(dim=2).view(b, 2, 1)
        x = (x - mean) / std
        x = x.view(b, c, h, w)
        return x, mean, std
    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        x = x * std + mean
        x = x.view(b, c, h, w)
        return x

    def forward(self, x,mask,input_kspace,sens_maps):
        """
        inp : b,z,h,w,2
        mask: b,coil,z,h,w,2
        input_kspace : b,coil,z,h,w,2
        sens_maps : b,coil,z,h,w,2
        """
        x = self.complex_to_slice_dim(x) # to b,2z,h,w
        x,mean,std = self.norm(x)
        fea0 = self.SEB(x)
        fea1 = self.infer1(fea0,mask,input_kspace,sens_maps,None,mean,std)
        fea2 = self.infer2(fea1,mask,input_kspace,sens_maps,None,mean,std)
        fea3 = self.infer3(fea2,mask,input_kspace,sens_maps,[fea1],mean,std)
        fea4 = self.infer4(fea3,mask,input_kspace,sens_maps,[fea1,fea2],mean,std)
        fea5 = self.infer5(fea4,mask,input_kspace,sens_maps,[fea1,fea2,fea3],mean,std)
    
        x = x + fea5 # b,2z,h,w

        x = self.unnorm(x,mean,std)
        x = self.chan_complex_to_last_dim(x) ## to b, z, h, w, two
        return x
    
    
    
    
class DFCN_single(nn.Module):
    def __init__(self, slices):
        super(DFCN_single, self).__init__()
        self.SEB = SE_Block(channel=2*slices)
        self.infer1 = Inference_Block_single(channel=2*slices,num_concat=0)
        self.infer2 = Inference_Block_single(channel=2*slices,num_concat=0)
        self.infer3 = Inference_Block_single(channel=2*slices,num_concat=1)
        self.infer4 = Inference_Block_single(channel=2*slices,num_concat=2)
        self.infer5 = Inference_Block_single(channel=2*slices,num_concat=3)
        self.apply(self._init_weights)
    def complex_to_slice_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass

    def norm(self, x):
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1)
        std = x.std(dim=2).view(b, 2, 1)
        x = (x - mean) / std
        x = x.view(b, c, h, w)
        return x, mean, std
    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        x = x * std + mean
        x = x.view(b, c, h, w)
        return x

    def forward(self, x,mask):
        """
        input_kspace : b,1,z,h,w,2
        mask: b,1,z,h,w,2
        """
        b,c,z,h,w,two = x.shape
        
        ksp = x.clone().view(b,z,h,w,two)
        img_x = fastmri.ifft2c(x).view(b,z,h,w,two)
        mask = mask.view(b,z,h,w,two)
        
        x = self.complex_to_slice_dim(img_x) # to b,2z,h,w
        x,mean,std = self.norm(x)
        fea0 = self.SEB(x)
        fea1 = self.infer1(fea0,mask,ksp,None,mean,std)
        fea2 = self.infer2(fea1,mask,ksp,None,mean,std)
        fea3 = self.infer3(fea2,mask,ksp,[fea1],mean,std)
        fea4 = self.infer4(fea3,mask,ksp,[fea1,fea2],mean,std)
        fea5 = self.infer5(fea4,mask,ksp,[fea1,fea2,fea3],mean,std)
    
        x = x + fea5 # b,2z,h,w

        x = self.unnorm(x,mean,std)
        x = fastmri.complex_abs(self.chan_complex_to_last_dim(x)) ## to b, z, h, w, two
        
        return x

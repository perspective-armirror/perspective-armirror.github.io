# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from basicsr.archs.local_arch import Local_Base
# from basicsr.utils.registry import ARCH_REGISTRY

class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
        if isinstance(suffix, str):
            name = name + '_' + suffix

        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = obj

    def register(self, obj=None, suffix=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class, suffix)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj, suffix)

    def get(self, name, suffix='basicsr'):
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + '_' + suffix)
            print(f'Name {name} is not found, use name: {name}_{suffix}!')
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()
ARCH_REGISTRY = Registry('arch')

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels,  kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


@ARCH_REGISTRY.register()
class NAFNet_Bing_Bksc_sub(nn.Module):

    def __init__(self, img_channel=3, out_channels=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                 dwex=2, ffex=2):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)
        self.feat_bksc = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                   bias=True)
        self.ending_bksc = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                                     groups=1,
                                     bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, DW_Expand=dwex, FFN_Expand=ffex) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, DW_Expand=dwex, FFN_Expand=ffex) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, DW_Expand=dwex, FFN_Expand=ffex) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x_bksc = self.feat_bksc(x)

        x1 = self.ending(x - x_bksc)
        x1 = F.relu(x1 + inp[:, :3, :, :])

        x2 = self.ending_bksc(x_bksc)
        # x2 = F.relu(x2 + inp[:, :3, :, :])
        x2 = F.relu(x2 + inp[:, 3:, :, :])

        return {'output': x1[:, :, :H, :W], 'bksc': x2[:, :, :H, :W]}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class SFTLayer(nn.Module):
    def __init__(self, inc, ch):
        super(SFTLayer, self).__init__()
        ks = 1
        self.SFT_scale_conv0 = nn.Conv2d(inc, ch, ks, padding=ks//2)
        self.SFT_scale_conv1 = nn.Conv2d(ch, ch, ks, padding=ks//2)
        self.SFT_shift_conv0 = nn.Conv2d(inc, ch, ks, padding=ks//2)
        self.SFT_shift_conv1 = nn.Conv2d(ch, ch, ks, padding=ks//2)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        cond = F.interpolate(x[1], scale_factor=x[0].shape[-1]/x[1].shape[-1], mode='bilinear')
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
        # print(scale.shape, shift.shape, x[0].shape)
        return x[0] * (scale + 1) + shift

@ARCH_REGISTRY.register()
class NAFNet_Bing_Bksc_sub_att2(nn.Module):

    def __init__(self, img_channel=3, out_channels=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dwex=2, ffex=2, sft=False):
        super().__init__()
        self.sft = sft
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        # self.feat_bksc = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
        #                       bias=True)
        self.ending_bksc = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                      bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders1 = nn.ModuleList()

        self.ups1 = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.sftlayers = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, DW_Expand=dwex, FFN_Expand=ffex) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        # self.middle_blks = \
        #     nn.Sequential(
        #         *[NAFBlock(chan, DW_Expand=dwex, FFN_Expand=ffex) for _ in range(middle_blk_num)]
        #     )
        self.middle_blks = nn.Sequential(
                *[AttnBlock(chan) for _ in range(middle_blk_num)]
            )
        temp_chan = chan
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            if self.sft:
                print('use sft', chan)
                self.sftlayers.append(SFTLayer(3, chan))
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, DW_Expand=dwex, FFN_Expand=ffex) for _ in range(num)]
                )
            )

        chan = temp_chan
        for num in dec_blk_nums:
            self.ups1.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders1.append(
                nn.Sequential(
                    *[NAFBlock(chan, DW_Expand=dwex, FFN_Expand=ffex) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward_bksc_decoder(self, x, encs):
        for decoder, up, enc_skip in zip(self.decoders1, self.ups1, encs[::-1]):
            # print(x.shape)
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        return x

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        x_bksc = self.forward_bksc_decoder(x, encs)

        _idx = 0
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            # print(x.shape)
            x = up(x)
            x = x + enc_skip
            if self.sft:
                x = self.sftlayers[_idx]((x, inp[:, 3:, :, :]))
                _idx += 1
            x = decoder(x)

        x1 = self.ending(x - x_bksc)
        x1 = F.relu(x1 + inp[:, :3, :, :])

        x2 = self.ending_bksc(x_bksc)
        # x2 = F.relu(x2 + inp[:, :3, :, :])
        x2 = F.relu(x2 + inp[:, 3:, :, :])

        return {'output': x1[:, :, :H, :W], 'bksc': x2[:, :, :H, :W]}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


@ARCH_REGISTRY.register()
class NAFNet_Bing_Bksc_sub_att2_sft(NAFNet_Bing_Bksc_sub_att2):

    def __init__(self, img_channel=3, out_channels=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dwex=2, ffex=2, sft=True):
        super(NAFNet_Bing_Bksc_sub_att2_sft, self).__init__(img_channel, out_channels, width, middle_blk_num, enc_blk_nums, dec_blk_nums, dwex, ffex, sft)


if __name__ == '__main__':
    img_channel = 6
    width = 32

    enc_blks = [1, 1, 1, 8]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = NAFNet_Bing_Bksc_sub_att2_sft(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)  # 10.576G 8.449M
    net = NAFNet_Bing_Bksc_sub_att2(img_channel=img_channel, img_channelwidth=width, middle_blk_num=middle_blk_num,
                                        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks) # 10.016G 8.270M

    net = NAFNet_Bing_Bksc_sub(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                                         enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)   # 8.534G 7.759M




    inp_shape = (1, 6, 256, 256)

    from thop import profile
    from thop import clever_format

    flops, params = profile(net, inputs=(torch.randn(*inp_shape),))
    flops, params = clever_format([flops, params], '%.3f')

    print(net.__class__.__name__, flops, params)
    exit()


    a = net(torch.randn(*inp_shape))
    for k,v in a.items():
        print(k, v.shape)
    exit()

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)

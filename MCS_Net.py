import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from torch.nn import Softmax
from functools import partial


class LayerNorm(nn.Module):
    """
        From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class Gate_Transport_Block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.w1 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

        self.w2 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size + 2, padding=(kernel_size + 2) // 2),
            # nn.Sigmoid()
            nn.GELU()
        )
        self.wo = nn.Sequential(
            DepthWiseConv2d(in_c, out_c, kernel_size + 2),
            nn.GELU()
        )

        self.cw = nn.Sequential(
            DepthWiseConv2d(in_c, out_c, kernel_size + 2),
            nn.GELU()
        )
    def forward(self, x):
        x1, x2 = self.w1(x), self.w2(x)
        out = self.wo(x1 + x2) + self.cw(x)
        return out

class Split_Combination_Gate_Bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[7, 5, 2, 1]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size + 1)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size + 1)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=group_size + 1)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=group_size + 1)
        )
        self.gtb = Gate_Transport_Block(dim_xl * 2 + 4, dim_xl, 1)

    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.gtb(x)
        return x


class Conv2dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(2, out_channel),
            nn.GELU()
        )
class Conv1dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(4, out_channel),
            nn.GELU()
        )
class InvertedDepthWiseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        layers = []
        # 1x1 pointwise conv
        layers.append(Conv2dGNGELU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            Conv2dGNGELU(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)
class InvertedDepthWiseConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        layers = []
        # 1x1 pointwise conv
        layers.append(Conv1dGNGELU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            Conv1dGNGELU(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class MACrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(MACrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.mwab = MWAB(in_dim)
    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.mwab(self.gamma * (out_H + out_W) + x)

class MWAB(nn.Module):
    '''
        Multi__Aggregation_Block
    '''
    def __init__(self, dim, bias=False, a=16, b=16, c_h=16, c_w=16):
        super().__init__()
        self.register_buffer("dim", torch.as_tensor(dim))
        self.register_buffer("a", torch.as_tensor(a))
        self.register_buffer("b", torch.as_tensor(b))
        self.register_buffer("c_h", torch.as_tensor(c_h))
        self.register_buffer("c_w", torch.as_tensor(c_w))
        self.a_weight = nn.Parameter(torch.Tensor(2, 1, dim // 4, a))
        nn.init.ones_(self.a_weight)
        self.b_weight = nn.Parameter(torch.Tensor(2, 1, dim // 4, b))
        nn.init.ones_(self.b_weight)
        self.c_weight = nn.Parameter(torch.Tensor(2, dim // 4, c_h, c_w))
        nn.init.ones_(self.c_weight)
        self.dw_conv = InvertedDepthWiseConv2d(dim // 4, dim // 4)
        self.wg_a = nn.Sequential(
            InvertedDepthWiseConv1d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, dim // 4),
        )
        self.wg_b = nn.Sequential(
            InvertedDepthWiseConv1d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, dim // 4),
        )
        self.wg_c = nn.Sequential(
            InvertedDepthWiseConv2d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv2d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv2d(2 * dim // 4, dim // 4),
        )
    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, c, a, b = x1.size()
        # ----- a convlution -----#
        x1 = x1.permute(0, 2, 1, 3)  # B, a, c, b
        x1 = torch.fft.rfft2(x1, dim=(2, 3), norm='ortho')
        a_weight = self.a_weight
        a_weight = self.wg_a(F.interpolate(a_weight, size=x1.shape[2:4],
                                           mode='bilinear', align_corners=True
                                           ).squeeze(1)).unsqueeze(1).permute(1, 2, 3, 0)
        a_weight = torch.view_as_complex(a_weight.contiguous())
        x1 = x1 * a_weight
        x1 = torch.fft.irfft2(x1, s=(c, b), dim=(2, 3), norm='ortho').permute(0, 2, 1, 3)

        # ----- b convlution -----#
        x2 = x2.permute(0, 3, 1, 2)  # B, b, c, a
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
        b_weight = self.b_weight
        b_weight = self.wg_b(F.interpolate(b_weight, size=x2.shape[2:4],
                                           mode='bilinear', align_corners=True
                                           ).squeeze(1)).unsqueeze(1).permute(1, 2, 3, 0)
        b_weight = torch.view_as_complex(b_weight.contiguous())
        x2 = x2 * b_weight
        x2 = torch.fft.irfft2(x2, s=(c, a), dim=(2, 3), norm='ortho').permute(0, 2, 3, 1)
        # ----- c convlution -----#
        x3 = torch.fft.rfft2(x3, dim=(2, 3), norm='ortho')
        c_weight = self.c_weight
        c_weight = self.wg_c(F.interpolate(c_weight, size=x3.shape[2:4],
                                           mode='bilinear', align_corners=True)).permute(1, 2, 3, 0)
        c_weight = torch.view_as_complex(c_weight.contiguous())
        x3 = x3 * c_weight
        x3 = torch.fft.irfft2(x3, s=(a, b), dim=(2, 3), norm='ortho')
        # ----- dw convlution -----#
        x4 = self.dw_conv(x4)
        # ----- concat -----#
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.mlp = nn.Sequential(InvertedDepthWiseConv2d(dim, mlp_ratio * dim),
                                 InvertedDepthWiseConv2d(mlp_ratio * dim, dim),
                                 nn.GELU())
    def forward(self, x):
        return self.mlp(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.GroupNorm(4, dim)
        self.fn = fn
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class MACA(nn.Module):

    def __init__(self, dim, depth, mlp_ratio=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MACrossAttention(dim)),
                PreNorm(dim, MLP(dim, mlp_ratio))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


nonlinearity = partial(F.relu, inplace=True)
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CALayer(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CALayer, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv_h(x_h).sigmoid()
        x_w = self.conv_w(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        out = identity * x_w * x_h
        return out


class CAResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(CAResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.ca = CALayer(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        # x2 = self.conv2(x1)
        # x2 = self.bn2(x2)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.ca(x3)

        x4 = x1 + x3
        x4 = self.relu(x4)
        return x4


# 8,16,32,48,64,96
# GFLOPs :0.61, Params : 2.42
# 16,24,32,48,64,128
# GFLOPs :0.76, Params : 3.39
# 8,16,32,64,128,160
# GFLOPs :1.01, Params : 6.71
# 16,32,48,64,128,256
# GFLOPs :1.73, Params : 12.22
# 16,32,64,128,160,256
# GFLOPs :3.05, Params : 15.74
# 16,32,64,128,256,512
# GFLOPs :4.88, Params : 47.64

# depth = 1,1,1,1
# GFLOPs :0.61, Params : 2.42
# depth = 1,1,2,2
# GFLOPs :0.71, Params : 4.00
# depth = 1,2,2,4
# GFLOPs :0.94, Params : 6.47
# depth = 2,2,4,
# GFLOPs :1.29, Params : 7.58
class MCS_Net(nn.Module):
    def __init__(self, n_classes=1, n_channels=3, c_list=[8,16,32,48,64,96], depth=[1,1,1,1], bridge=True,
                 gt_ds=True, mlp_ratio=4):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.bridge = bridge
        self.gt_ds = gt_ds

        self.encoder1 = nn.Sequential(
            nn.Conv2d(n_channels, c_list[0], 3, stride=1, padding=1),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            MACA(c_list[2], depth[0], mlp_ratio)
        )


        self.encoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1),
            MACA(c_list[3], depth[1], mlp_ratio),
        )


        self.encoder5 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1),
            MACA(c_list[4], depth[2], mlp_ratio)
        )


        self.encoder6 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[5], 3, stride=1, padding=1),
            MACA(c_list[5], depth[3], mlp_ratio)
        )

        #Bottleneck layers
        self.ca = CAResBlock(c_list[5],c_list[5])

        if bridge:
            self.SCGB1 = Split_Combination_Gate_Bridge(c_list[1], c_list[0])
            self.SCGB2 = Split_Combination_Gate_Bridge(c_list[2], c_list[1])
            self.SCGB3 = Split_Combination_Gate_Bridge(c_list[3], c_list[2])
            self.SCGB4 = Split_Combination_Gate_Bridge(c_list[4], c_list[3])
            self.SCGB5 = Split_Combination_Gate_Bridge(c_list[5], c_list[4])
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))

        self.decoder1 = nn.Sequential(
            MACA(c_list[5], depth[3], mlp_ratio),
            nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1)
        )

        self.decoder2 = nn.Sequential(
            MACA(c_list[4], depth[2], mlp_ratio),
            nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1)
        )

        self.decoder3 = nn.Sequential(
            MACA(c_list[3], depth[1], mlp_ratio),
            nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1),
        )

        self.decoder4 = nn.Sequential(
            MACA(c_list[2], depth[0], mlp_ratio),
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )

        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], n_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        # out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3_1(self.encoder3(out))),2,2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        # out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4_1(self.encoder4(out))),2,2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        # out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5_1(self.encoder5(out))),2,2))
        t5 = out  # b, c4, H/32, W/32

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        # out = F.gelu(self.encoder6_1(self.encoder6(out))) # b, c5, H/32, W/32
        t6 = out

        out_bottleneck = self.ca(out)

        out5 = F.gelu(self.dbn1(self.decoder1(out_bottleneck)))  # b, c4, H/32, W/32
        # out5 = F.gelu(self.dbn1(self.decoder1_1(self.decoder1(out)))) # b, c4, H/32, W/32
        if self.gt_ds:
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.SCGB5(t6, t5, gt_pre5)
            # gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode ='bilinear', align_corners=True)
        else:
            t5 = self.SCGB5(t6, t5)
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        # out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2_1(self.decoder2(out5))),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.SCGB4(t5, t4, gt_pre4)
            # gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        else:
            t4 = self.SCGB4(t5, t4)
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        # out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3_1(self.decoder3(out4))),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.SCGB3(t4, t3, gt_pre3)
            # gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        else:
            t3 = self.SCGB3(t4, t3)
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        # out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4_1(self.decoder4(out3))),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.SCGB2(t3, t2, gt_pre2)
            # gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        else:
            t2 = self.SCGB2(t3, t2)
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.SCGB1(t2, t1, gt_pre1)
            # gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
        else:
            t1 = self.SCGB1(t2, t1)
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W
        return torch.sigmoid(out0)



# from thop import profile
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model =MCS_Net().to(device)
# input = torch.randn(1, 3, 224, 224).to(device)
# flops, params = profile(model, inputs=(input, ))
# print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops/1e9,params/1e6)) #flops单位G，para单位M
# # GFLOPs :0.61, Params : 2.42

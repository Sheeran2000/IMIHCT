import torch
import torch.nn as nn
import torch.nn.functional as F

# ERB模块
lrelu_value = 0.1
act = nn.LeakyReLU(lrelu_value)


class RRRB(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.

    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|


    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(RRRB, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, ratio * n_feats, 1, 1, 0)
        self.fea_conv = nn.Conv2d(ratio * n_feats, ratio * n_feats, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(ratio * n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out

        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x

        return out


class ERB(nn.Module):
    """ Enhanced residual block for building FEMN.

    Diagram:
        --RRRB--LeakyReLU--RRRB--

    Args:
        n_feats (int): Number of feature maps.
        ratio (int): Expand ratio in RRRB.
    """

    def __init__(self, n_feats, ratio=2):
        super(ERB, self).__init__()
        self.conv1 = RRRB(n_feats, ratio)
        self.conv2 = RRRB(n_feats, ratio)

    def forward(self, x):
        out = self.conv1(x)
        out = act(out)
        out = self.conv2(out)

        return out


def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t


# ERB模块

# RepBlock
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        act_func = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        act_func = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        act_func = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    # TODO: 新增silu和gelu激活函数
    elif act_type == 'silu':
        pass
    elif act_type == 'gelu':
        pass
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return act_func


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        if self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
        # explicitly padding with bias
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        # conv-3x3
        y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None
        tmp = self.scale * self.mask
        k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
        for i in range(self.out_planes):
            k1[i, i, :, :] = tmp[i, 0, :, :]
        b1 = self.bias
        # re-param conv kernel
        RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
        # re-param conv bias
        RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
        RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation('lrelu')

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=1,
                                         padding=1, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            self.rbr_3x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
            self.rbr_3x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                            stride=1, padding=(1, 0), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                            stride=1, padding=(0, 1), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x1_3x3_branch_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels,
                                                    kernel_size=(1, 1),
                                                    stride=1, padding=(0, 0), dilation=1, groups=1,
                                                    padding_mode='zeros', bias=False)
            self.rbr_1x1_3x3_branch_3x3 = nn.Conv2d(in_channels=2 * in_channels, out_channels=out_channels,
                                                    kernel_size=(3, 3),
                                                    stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros',
                                                    bias=False)
            self.rbr_conv1x1_sbx_branch = SeqConv3x3('conv1x1-sobelx', self.in_channels, self.out_channels)
            self.rbr_conv1x1_sby_branch = SeqConv3x3('conv1x1-sobely', self.in_channels, self.out_channels)
            self.rbr_conv1x1_lpl_branch = SeqConv3x3('conv1x1-laplacian', self.in_channels, self.out_channels)

    def forward(self, inputs):
        if (self.deploy):
            return self.activation(self.rbr_reparam(inputs))
        else:
            return self.activation(
                self.rbr_3x3_branch(inputs) + self.rbr_3x1_branch(inputs) + self.rbr_1x3_branch(
                    inputs) + self.rbr_1x1_branch(inputs) + self.rbr_1x1_3x3_branch_3x3(
                    self.rbr_1x1_3x3_branch_1x1(inputs)) + inputs + self.rbr_conv1x1_sbx_branch(
                    inputs) + self.rbr_conv1x1_sby_branch(inputs) + self.rbr_conv1x1_lpl_branch(inputs))

    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                     stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_3x3_branch')
        self.__delattr__('rbr_3x1_branch')
        self.__delattr__('rbr_1x3_branch')
        self.__delattr__('rbr_1x1_branch')
        self.__delattr__('rbr_1x1_3x3_branch_1x1')
        self.__delattr__('rbr_1x1_3x3_branch_3x3')
        self.__delattr__('rbr_conv1x1_sbx_branch')
        self.__delattr__('rbr_conv1x1_sby_branch')
        self.__delattr__('rbr_conv1x1_lpl_branch')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        # 3x3 branch
        kernel_3x3, bias_3x3 = self.rbr_3x3_branch.weight.data, self.rbr_3x3_branch.bias.data

        # 1x1 1x3 3x1 branch
        kernel_1x1_1x3_3x1_fuse, bias_1x1_1x3_3x1_fuse = self._fuse_1x1_1x3_3x1_branch(self.rbr_1x1_branch,
                                                                                       self.rbr_1x3_branch,
                                                                                       self.rbr_3x1_branch)
        # 1x1+3x3 branch
        kernel_1x1_3x3_fuse = self._fuse_1x1_3x3_branch(self.rbr_1x1_3x3_branch_1x1,
                                                        self.rbr_1x1_3x3_branch_3x3)
        # identity branch
        device = kernel_1x1_3x3_fuse.device  # just for getting the device
        kernel_identity = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device)
        for i in range(self.out_channels):
            kernel_identity[i, i, 1, 1] = 1.0

        kernel_1x1_sbx, bias_1x1_sbx = self.rbr_conv1x1_sbx_branch.rep_params()
        kernel_1x1_sby, bias_1x1_sby = self.rbr_conv1x1_sby_branch.rep_params()
        kernel_1x1_lpl, bias_1x1_lpl = self.rbr_conv1x1_lpl_branch.rep_params()

        return kernel_3x3 + kernel_1x1_1x3_3x1_fuse + kernel_1x1_3x3_fuse + kernel_identity + kernel_1x1_sbx + kernel_1x1_sby + kernel_1x1_lpl, bias_3x3 + bias_1x1_1x3_3x1_fuse + bias_1x1_sbx + bias_1x1_sby + bias_1x1_lpl

    def _fuse_1x1_1x3_3x1_branch(self, conv1, conv2, conv3):
        weight = F.pad(conv1.weight.data, (1, 1, 1, 1)) + F.pad(conv2.weight.data, (0, 0, 1, 1)) + F.pad(
            conv3.weight.data, (1, 1, 0, 0))
        bias = conv1.bias.data + conv2.bias.data + conv3.bias.data
        return weight, bias

    def _fuse_1x1_3x3_branch(self, conv1, conv2):
        weight = F.conv2d(conv2.weight.data, conv1.weight.data.permute(1, 0, 2, 3))
        return weight


# RepBlock

# ---------轻量化门控卷积-----------

class depth_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super(depth_separable_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class sc_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super(sc_conv, self).__init__()
        self.single_channel_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1
        )

    def forward(self, input):
        out = self.single_channel_conv(input)
        return out


# -----------------------------------------------
#                Gated ConvBlock
# -----------------------------------------------
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='replicate',
                 activation='elu', norm='none', sc=False, sn=False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sc:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = sc_conv(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            # self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = depth_separable_conv(in_channels, out_channels, kernel_size, stride, padding=0,
                                                    dilation=dilation)

        self.sigmoid = torch.nn.Sigmoid()
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels,
        #                   kernel_size=1, stride=stride, bias=False),
        #         self.norm
        #     )

    def forward(self, x_in):
        x = self.pad(x_in)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        if self.norm:
            conv = self.norm(conv)
        if self.activation:
            conv = self.activation(conv)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask

        # x += self.shortcut(x_in)

        return x



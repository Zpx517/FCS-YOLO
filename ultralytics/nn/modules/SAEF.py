import torch
from torch import nn


__all__=['SAEF']
class SAEF(nn.Module):
    def __init__(self, dim, ratio=16):
        super(SAEF, self).__init__()
        self.dim_1 = dim

        self.branch_conv0 = nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1)
        self.branch_conv1 = nn.Conv2d(in_channels=4*self.dim_1 // 8,
                                      out_channels=4*self.dim_1 // 8,
                                      stride=1, kernel_size=7, groups=4*self.dim_1 // 8,
                                      dilation=3, padding=(1+6*3) // 2)
        self.branch_conv2 = nn.Conv2d(in_channels=3*self.dim_1 // 8,
                                      out_channels=3*self.dim_1 // 8,
                                      stride=1, kernel_size=5, groups=3*self.dim_1 // 8,
                                      dilation=2, padding=(1+4*2) // 2)
        self.branch_conv3 = nn.Conv2d(in_channels=self.dim_1 // 8,
                                      out_channels=self.dim_1 // 8,
                                      stride=1, kernel_size=3, groups=self.dim_1 // 8,
                                      dilation=1, padding=(1+2*1) // 2)
        self.P_conv = nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1)

        self.avg_pool_1 = nn.AvgPool2d(7, 1, 3)

        self.conv1X1 = nn.Sequential(nn.Conv2d(self.dim_1, self.dim_1, 1),
                                     nn.BatchNorm2d(self.dim_1),
                                     nn.ReLU(inplace=True))

        self.depth_convs = nn.Sequential(
            nn.Conv2d(self.dim_1, self.dim_1, kernel_size=(1, 7), padding=(0, 7 // 2), groups=self.dim_1),
            nn.Conv2d(self.dim_1, self.dim_1, kernel_size=(7, 1), padding=(7 // 2, 0), groups=self.dim_1),
            nn.Conv2d(self.dim_1, self.dim_1, kernel_size=(1, 11), padding=(0, 11 // 2), groups=self.dim_1),
            nn.Conv2d(self.dim_1, self.dim_1, kernel_size=(11, 1), padding=(11 // 2, 0), groups=self.dim_1)
        )

        self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(dim, dim // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(dim // ratio, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        shortcut, x1, x2 = x, x, x

        x1 = self.branch_conv0(x1)
        x1_1, x1_2, x1_3 = torch.split(x1, [4*self.dim_1 // 8, 3*self.dim_1 // 8, self.dim_1 // 8], dim=1)

        x1_1 = self.branch_conv1(x1_1)
        x1_2 = self.branch_conv2(x1_2)
        x1_3 = self.branch_conv3(x1_3)

        x1_out = self.act1(self.P_conv(torch.cat([x1_1, x1_2, x1_3], dim=1)))

        x2_out = self.act2(self.conv1X1(self.depth_convs(self.conv1X1(self.avg_pool_1(x2)))))

        out = self.P_conv(x1_out * x2_out) + shortcut

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool_2(out))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(out))))

        ca_out = avg_out + max_out
        output = out * self.sigmoid(ca_out)

        return output

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

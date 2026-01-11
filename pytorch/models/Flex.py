import torch
import torch.nn as nn
import torch.nn.functional as F


# print('here 1')


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kern=3, dropout=0.0):
        super(double_conv, self).__init__()
        padding = (kern - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kern, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_ch, out_ch, kern, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kern=3, dropout=0.0):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, kern=kern, dropout=dropout)

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch, kern=3, dropout=0.0):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, kern=kern, dropout=dropout)
        )

    def forward(self, x):
        return self.mpconv(x)


# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, kern=3, dropout=0.0, bilinear=True):
#         super(up, self).__init__()

#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

#         self.conv = double_conv(in_ch, out_ch, kern=kern, dropout=dropout)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
class up(nn.Module):
    def __init__(self, up_in_ch, skip_ch, out_ch, kern=3, dropout=0.0, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(up_in_ch, up_in_ch, kernel_size=2, stride=2)

        self.conv = double_conv(up_in_ch + skip_ch, out_ch, kern=kern, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concat upsampled and skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
    



class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class U1(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, ups, downs, kerns, dropout=0.0, bilinear=True):
        super(U1, self).__init__()

        self.inc = inconv(n_channels_in, downs[0], kern=kerns[0], dropout=dropout)
        self.down_Layers = nn.ModuleList()
        self.up_Layers = nn.ModuleList()

        # Down path
        for i in range(len(downs) - 1):
            self.down_Layers.append(
                down(downs[i], downs[i + 1], kern=kerns[0], dropout=dropout)
            )

        # Up path (skips = reversed downs)
        skips = downs[::-1]
        up_ins = ups  # channels from up path
        up_outs = ups[1:] + [ups[-1]]  # output channels after convs

        # First up layer (last down output + second-last down as skip)
        self.up_Layers.append(
            up(up_in_ch=downs[-1], skip_ch=skips[1], out_ch=ups[1], kern=kerns[1], dropout=dropout, bilinear=bilinear)
        )

        for i in range(1, len(ups) - 1):
            self.up_Layers.append(
                up(up_in_ch=ups[i], skip_ch=skips[i + 1], out_ch=ups[i + 1], kern=kerns[1], dropout=dropout, bilinear=bilinear)
            )

        self.outc = outconv(ups[-1], n_channels_out)

    def forward(self, x):
        xN = self.inc(x)
        xN_ = [xN]

        for d in self.down_Layers:
            xN = d(xN)
            xN_.append(xN)

        xN = self.up_Layers[0](xN_[-1], xN_[-2])
        for i, u in enumerate(self.up_Layers[1:]):
            xN = u(xN, xN_[-3 - i])

        return self.outc(xN)

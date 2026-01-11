import torch
import torch.nn as nn
import torch.nn.functional as F

class U1_drop(nn.Module):
    def __init__(self, n_channels_in, n_channels_out,
                 dropout_enc=0.0, dropout_bottleneck=0.0, dropout_dec=0.0):
        super(U1_drop, self).__init__()

        # Encoder
        self.inc = inconv(n_channels_in, 64, dropout_enc)
        self.down1 = down(64, 128, dropout_enc)
        self.down2 = down(128, 256, dropout_enc)
        self.down3 = down(256, 512, dropout_enc)
        self.down4 = down(512, 512, dropout_bottleneck)  # bottleneck

        # Decoder
        self.up1 = up(1024, 256, dropout_dec, bilinear=True)
        self.up2 = up(512, 128, dropout_dec, bilinear=True)
        self.up3 = up(256, 64, dropout_dec, bilinear=True)
        self.up4 = up(128, 64, dropout_dec, bilinear=True)

        self.outc = outconv(64, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # bottleneck
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class double_conv(nn.Module):
    '''(conv => BN => ReLU => Dropout) * 2'''
    def __init__(self, in_ch, out_ch, dropout_rate=0.0):
        super(double_conv, self).__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        layers.extend([
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ])
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.0):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dropout_rate)

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.0):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, dropout_rate)
        )

    def forward(self, x):
        return self.mpconv(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.0, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.conv(x)

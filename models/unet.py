#unet.py
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.down1 = DoubleConv(1, 64, dropout=0.10)
        self.down2 = DoubleConv(64, 128, dropout=0.20)
        self.down3 = DoubleConv(128, 256, dropout=0.30)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128, dropout=0.20)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64, dropout=0.10)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool(c1)

        c2 = self.down2(p1)
        p2 = self.pool(c2)

        c3 = self.down3(p2)

        u1 = self.up1(c3)
        u1 = torch.cat([u1, c2], dim=1)
        c4 = self.conv1(u1)

        u2 = self.up2(c4)
        u2 = torch.cat([u2, c1], dim=1)
        c5 = self.conv2(u2)

        return self.out(c5)
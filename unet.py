import imageio
import numpy as np
import torch as th
import torch.nn as nn
import torchvision.transforms
from PIL import Image


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.convs = nn.Sequential(
            # use padding same to avoid later cropping and better embedding
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convs(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.convs = nn.Sequential(
            DoubleConv(in_channels, out_channels)
        )

        self.downnet = nn.Sequential(
            nn.MaxPool2d(2, 2, 0)
        )

    def forward(self, x):
        # first part through the normal convs:
        x = self.convs(x)
        # save the output for later cropping and concat
        x_down = self.downnet(x)
        # return both versions for further processing
        return x_down, x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.upconvs = nn.Sequential(
            # not sure if align_corners is relevant, but probably better
            # TODO: read up on this
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        )

        self.reducefeatures = nn.Sequential(
            # doing a simple 1x1 conv for first demo

        )

        self.convs = nn.Sequential(
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x, x_skip):
        x = self.upconvs(x)
        # x = self.reducefeatures(x)
        target_dims = x.size()[-2:]  # cut off the feature dimension should give a 2d resolution

        # check if this works or blows up because it cant handle the feature dimension
        x_skip = torchvision.transforms.CenterCrop(size=target_dims)(x_skip)
        # dim 1 is the correct one because of
        # data being in (batch, channel, H, W) format. We want to stack channel
        x = th.cat((x_skip, x), dim=1)

        # the usual double encoder (in this case decoder
        x = self.convs(x)
        return x


class UNET(nn.Module):
    def __init__(self, in_channels):
        super(UNET, self).__init__()

        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.mid = DoubleConv(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outconv = nn.Sequential(
            # out_channel=1 since we only want to have a highlight of the target region
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        )

    def forward(self, x):
        x, x1 = self.down1(x)
        x, x2 = self.down2(x)
        x, x3 = self.down3(x)
        x, x4 = self.down4(x)
        x = self.mid(x)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outconv(x)

        return out



if __name__ == '__main__':
    model = UNET(in_channels=1)

    im = imageio.imread("camera.png")

    im = Image.fromarray(im)
    im = im.resize(size=(572, 572), resample=Image.LANCZOS)
    im = np.asarray(im)

    im = im[np.newaxis, np.newaxis, :, :]
    im_tensor = th.Tensor(im)

    model(im_tensor)

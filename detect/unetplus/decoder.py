import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv2dReLU
from ..base.model import Model


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3,
                       padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3,
                       padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        y = x[0]
        skip = x[1:]
        skip_len = len(skip)
        y = F.interpolate(y, scale_factor=2, mode='nearest')
        if skip[0] is not None:
            if skip_len == 1:
                y = torch.cat([y, skip[0]], dim=1)
            elif skip_len == 2:
                y = torch.cat([y, skip[0], skip[1]], dim=1)
            elif skip_len == 3:
                y = torch.cat([y, skip[0], skip[1], skip[2]], dim=1)
            elif skip_len == 4:
                y = torch.cat([y, skip[0], skip[1], skip[2], skip[3]], dim=1)
            elif skip_len == 5:
                y = torch.cat([y, skip[0], skip[1], skip[2],
                               skip[3], skip[4]], dim=1)
        y = self.block(y)
        return y


class CenterBlock(DecoderBlock):

    def forward(self, x):
        return self.block(x)


class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            inference_layer=4,
            use_batchnorm=True,
            center=False,
    ):
        super().__init__()
        self.inference_layer = inference_layer
        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(
                channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(
            in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer21 = DecoderBlock(
            in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderBlock(
            in_channels[2], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer31 = DecoderBlock(
            in_channels[3], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer32 = DecoderBlock(
            in_channels[4], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderBlock(
            in_channels[5], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer41 = DecoderBlock(
            in_channels[6], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer42 = DecoderBlock(
            in_channels[7], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer43 = DecoderBlock(
            in_channels[8], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderBlock(
            in_channels[9], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderBlock(
            in_channels[10], out_channels[4], use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(
            out_channels[4], final_channels, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],  # x
            encoder_channels[1] + encoder_channels[2],
            encoder_channels[2]+decoder_channels[1]+decoder_channels[0],
            encoder_channels[2] + encoder_channels[3],
            encoder_channels[3] + decoder_channels[2] + decoder_channels[1],
            encoder_channels[3]+2*decoder_channels[2]+decoder_channels[1],
            encoder_channels[3]+encoder_channels[4],
            encoder_channels[4]+decoder_channels[3]+decoder_channels[2],
            encoder_channels[4]+2*decoder_channels[3]+decoder_channels[2],
            encoder_channels[4]+3*decoder_channels[3]+decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        if self.training == True or self.inference_layer == 4:
            x1 = self.layer1([encoder_head, skips[0]])
            x21 = self.layer21([skips[0], skips[1]])
            x2 = self.layer2([x1, skips[1], x21])
            x31 = self.layer31([skips[1], skips[2]])
            x32 = self.layer32([x21, skips[2], x31])
            x3 = self.layer3([x2, skips[2], x31, x32])
            x41 = self.layer41([skips[2], skips[3]])
            x42 = self.layer42([x31, skips[3], x41])
            x43 = self.layer43([x32, skips[3], x41, x42])
            y = self.layer4([x3, skips[3], x41, x42, x43])
        else:
            if self. inference_layer == 1:
                y = self.layer41([skips[2], skips[3]])
            elif self.inference_layer == 2:
                x31 = self.layer31([skips[1], skips[2]])
                x41 = self.layer41([skips[2], skips[3]])
                y = self.layer42([x31, skips[3], x41])
            elif self.inference_layer == 3:
                x21 = self.layer21([skips[0], skips[1]])
                x31 = self.layer31([skips[1], skips[2]])
                x32 = self.layer32([x21, skips[2], x31])
                x41 = self.layer41([skips[2], skips[3]])
                x42 = self.layer42([x31, skips[3], x41])
                y = self.layer43([x32, skips[3], x41, x42])
        y = self.layer5([x42, None])
        y = self.final_conv(y)
        return y

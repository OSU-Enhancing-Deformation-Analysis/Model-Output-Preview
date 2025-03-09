import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.k_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.o_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scale parameter

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.q_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # [B, N, C'] N=H*W, C'=C//8
        key = self.k_conv(x).view(batch_size, -1, width * height)  # [B, C', N]
        value = self.v_conv(x).view(batch_size, -1, width * height)  # [B, C, N]

        attention = torch.bmm(query, key)  # [B, N, N]
        attention = torch.softmax(attention, dim=-1)

        attention_value = torch.bmm(value, attention)  # [B, C, N]
        attention_value = attention_value.view(batch_size, channels, height, width)  # [B, C, H, W]

        output = self.o_conv(attention_value)
        return self.gamma * output + x  # Residual connection with learnable scale


class ComboMotionVectorRegressionNetwork(nn.Module):
    def __init__(self, input_images=2, base_channels=64, num_conv_blocks=3):
        super().__init__()
        self.input_images = input_images
        self.vector_channels = 2
        channels = base_channels

        # Downsampling path (No changes needed)
        self.conv1 = ConvolutionBlock(input_images, channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = ConvolutionBlock(channels, channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        conv_blocks = []
        current_channels = channels * 2
        feature_channels = [channels, channels * 2]
        down_feature_channel_sizes = []
        for _ in range(num_conv_blocks):
            channels *= 2
            down_feature_channel_sizes.append(current_channels)
            conv_blocks.extend(
                [
                    ConvolutionBlock(current_channels, channels, kernel_size=3),
                    ConvolutionBlock(channels, channels, kernel_size=3),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            current_channels = channels
            feature_channels.append(channels)
        self.conv_layers_down = nn.Sequential(*conv_blocks[:-1])
        down_feature_channel_sizes.append(current_channels)

        # Bottleneck with Attention (No changes needed)
        self.attention = SelfAttention(channels)

        # Upsampling path (Corrected in Attempt 4)
        up_channels = channels  # up_channels = 512
        self.upconv1 = nn.ConvTranspose2d(up_channels, up_channels // 2, kernel_size=4, stride=2, padding=1)  # 512 -> 256
        self.conv_up1 = ConvolutionBlock(up_channels // 2 + down_feature_channel_sizes[-2], up_channels // 2)  # Input: 512, Output: 256

        up_channels = 256  # Reset up_channels to output of conv_up1 = 256
        self.upconv2 = nn.ConvTranspose2d(up_channels, up_channels // 2, kernel_size=4, stride=2, padding=1)  # Revised: 256 -> 128
        self.conv_up2 = ConvolutionBlock(up_channels // 2 + down_feature_channel_sizes[-3], up_channels // 2)  # Revised Input: 384, Output: 128

        up_channels //= 2  # up_channels = 128
        self.upconv3 = nn.ConvTranspose2d(up_channels, up_channels // 2, kernel_size=4, stride=2, padding=1)  # Revised: 128 -> 64
        self.conv_up3 = ConvolutionBlock(up_channels // 2 + feature_channels[1], up_channels // 2)  # Revised Input: 192, Output: 64

        up_channels //= 2  # up_channels = 64
        self.upconv4 = nn.ConvTranspose2d(up_channels, up_channels // 2, kernel_size=4, stride=2, padding=1)  # Revised: 64 -> 32
        self.conv_up4 = ConvolutionBlock(up_channels // 2 + feature_channels[0], up_channels // 2)  # Revised Input: 96, Output: 32

        up_channels //= 2  # up_channels = 32
        self.output_conv = nn.Conv2d(up_channels, self.vector_channels, kernel_size=3, stride=1, padding=1)  # Revised: Input 32 -> Output 2

    def forward(self, x):
        # Downsampling
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv_down = pool2
        intermediate_features = [conv1, conv2]
        down_features_to_concat = []

        # Deeper Downsampling
        for i, layer in enumerate(self.conv_layers_down):
            conv_down = layer(conv_down)
            if isinstance(layer, ConvolutionBlock) and (i + 1) % 3 == 1:
                down_features_to_concat.append(conv_down)

        conv_bottleNeck = self.attention(conv_down)

        # Upsampling
        upconv1 = self.upconv1(conv_bottleNeck)
        upconv1_concat = torch.cat([upconv1, down_features_to_concat[-2]], dim=1)  # Corrected index
        conv_up1 = self.conv_up1(upconv1_concat)

        upconv2 = self.upconv2(conv_up1)
        upconv2_concat = torch.cat([upconv2, down_features_to_concat[-3]], dim=1)  # Corrected index
        conv_up2 = self.conv_up2(upconv2_concat)

        upconv3 = self.upconv3(conv_up2)
        upconv3_concat = torch.cat([upconv3, intermediate_features[1]], dim=1)
        conv_up3 = self.conv_up3(upconv3_concat)

        upconv4 = self.upconv4(conv_up3)
        upconv4_concat = torch.cat([upconv4, intermediate_features[0]], dim=1)
        conv_up4 = self.conv_up4(upconv4_concat)

        output = self.output_conv(conv_up4)
        return output

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


class DeeperWiderMotionVectorRegressionNetwork(nn.Module):
    def __init__(self, input_images=2, base_channels=64, num_conv_blocks=3):  # Parameters for depth and width
        super().__init__()
        self.input_images = input_images
        self.vector_channels = 2
        channels = base_channels  # Start with wider base channels

        conv_layers = [
            ConvolutionBlock(input_images, channels, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
        ]
        current_channels = channels
        for _ in range(num_conv_blocks):  # Add more convolutional blocks (deeper)
            channels *= 2  # Wider channels as we go deeper
            conv_layers.extend(
                [
                    ConvolutionBlock(current_channels, channels, kernel_size=3),
                    ConvolutionBlock(channels, channels, kernel_size=3),  # Added another block in each stage
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            current_channels = channels

        self.convolution = nn.Sequential(*conv_layers[:-1])  # Remove last pooling

        self.output = nn.Sequential(
            # Upsampling path - adjusted to match deeper convolution and output size
            nn.ConvTranspose2d(channels, channels // 2, kernel_size=4, stride=2, padding=1),  # Upsample to 64x64
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(channels // 2),
            nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size=4, stride=2, padding=1),  # Upsample to 128x128
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(channels // 4),
            nn.ConvTranspose2d(channels // 4, channels // 8, kernel_size=4, stride=2, padding=1),  # Upsample to 256x256 - NEW LAYER
            nn.LeakyReLU(0.1),  # NEW LAYER
            nn.BatchNorm2d(channels // 8),  # NEW LAYER
            nn.Conv2d(channels // 8, self.vector_channels, kernel_size=3, stride=1, padding=1),  # Output layer
        )

    def forward(self, x):
        x = self.convolution(x)
        x = self.output(x)
        return x

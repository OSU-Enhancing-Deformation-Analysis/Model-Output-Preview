import os

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_FILE = "./data/model_snapshot.pt"

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


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


class CorrelationLayer(nn.Module):
    def __init__(self, patch_size=1, kernel_size=1, stride=1, padding=0, max_displacement=20):
        super().__init__()
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_displacement = max_displacement

    def forward(self, feature_map_1, feature_map_2):
        """
        Args:
            feature_map_1: Feature map from image 1 (N, C, H, W)
            feature_map_2: Feature map from image 2 (N, C, H, W)

        Returns:
            correlation_volume: Correlation volume (N, 1, H, W, (2*max_displacement+1)**2)
        """
        batch_size, channels, height, width = feature_map_1.size()

        # Pad feature_map_2 to handle displacements
        padding_size = self.max_displacement
        feature_map_2_padded = F.pad(feature_map_2, (padding_size, padding_size, padding_size, padding_size))

        correlation_volume_list = []

        # loop over all possible displacements within max_displacement
        for displacement_y in range(-self.max_displacement, self.max_displacement + 1):
            for displacement_x in range(-self.max_displacement, self.max_displacement + 1):
                # Shift feature_map_2
                shifted_feature_map_2 = feature_map_2_padded[
                    :,
                    :,
                    padding_size + displacement_y : padding_size + displacement_y + height,
                    padding_size + displacement_x : padding_size + displacement_x + width,
                ]

                # now we compute correlation and reshape for correlation volume
                correlation_map = (feature_map_1 * shifted_feature_map_2).sum(dim=1, keepdim=True)  # Sum over channels
                correlation_volume_list.append(correlation_map)

        # put them all together
        correlation_volume = torch.cat(correlation_volume_list, dim=1)  # N, (2*max_displacement+1)**2, H, W
        correlation_volume = correlation_volume.permute(0, 2, 3, 1).unsqueeze(1)  # N, 1, H, W, (2*max_displacement+1)**2 - reshape to match expected output

        return correlation_volume


class MotionVectorRegressionNetworkWithCorrelation(nn.Module):
    def __init__(self, input_images=2, max_displacement=20):
        super().__init__()
        # Outputs an xy motion vector per pixel
        self.input_images = input_images
        self.vector_channels = 2
        self.max_displacement = max_displacement  # Store max_displacement

        self.feature_convolution = nn.Sequential(
            ConvolutionBlock(1, 32, kernel_size=3),  # input_images (1) -> 32 channels
            nn.MaxPool2d(kernel_size=2),  # scales down by half
            ConvolutionBlock(32, 64, kernel_size=3),  # 32 -> 64 channels
            nn.MaxPool2d(kernel_size=2),  # scales down by half
            ConvolutionBlock(64, 128, kernel_size=3),  # 64 -> 128 channels
            # ConvolutionBlock(64, 128, kernel_size=3),
        )

        self.correlation_layer = CorrelationLayer(max_displacement=self.max_displacement)  # Correlation Layer

        self.convolution_after_correlation = nn.Sequential(  # Convolution layers after correlation
            ConvolutionBlock(128 + (2 * max_displacement + 1) ** 2, 128, kernel_size=3),
            ConvolutionBlock(128, 128, kernel_size=3),  # 128 -> 128 channels
        )

        self.output = nn.Sequential(
            # scale back up
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 64 channels
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 32 channels
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, self.vector_channels, kernel_size=3, stride=1, padding=1),  # 32 -> 2 channels
        )

    def forward(self, x):
        # split input into image 1 and image 2
        image1 = x[:, 0:1, :, :]
        image2 = x[:, 1:2, :, :]

        features1 = self.feature_convolution(image1)
        features2 = self.feature_convolution(image2)

        # Correlation layer
        correlation_volume = self.correlation_layer(features1, features2)  # N, 1, H, W, (2*max_displacement+1)**2

        # concatenate correlation volume with features1 (you can experiment with features2 or concatenation strategy)
        # reshape correlation volume to (N, C, H, W) where C = (2*max_displacement+1)**2
        correlation_volume_reshaped = correlation_volume.squeeze(1).permute(0, 3, 1, 2)  # N, (2*max_displacement+1)**2, H, W
        combined_features = torch.cat((features1, correlation_volume_reshaped), dim=1)  # Concatenate along channel dimension

        x = self.convolution_after_correlation(combined_features)
        x = self.output(x)  # output layers
        return x


class MotionVectorRegressionNetworkWithWarping(nn.Module):  # Model 3: Model 1 + Warping (2-Stage Stacked)
    def __init__(self, input_images=2, max_displacement=20):
        super().__init__()
        self.input_images = input_images
        self.max_displacement = max_displacement

        self.stage1_model = MotionVectorRegressionNetworkWithCorrelation(input_images=input_images, max_displacement=self.max_displacement)
        self.stage2_model = MotionVectorRegressionNetworkWithCorrelation(input_images=input_images, max_displacement=self.max_displacement)  # Stage 2 model - same architecture as stage 1

    def forward(self, x):
        # Split input into image 1 and image 2
        image1 = x[:, 0:1, :, :]  # Assuming grayscale input, adjust if RGB
        image2 = x[:, 1:2, :, :]

        # Stage 1: Predict flow1
        flow1 = self.stage1_model(x)  # Input is the original image pair

        # Warping layer: Warp image2 using flow1
        # Create grid for warping
        batch_size, _, height, width = image1.size()
        grid = self._create_meshgrid(batch_size, height, width, device=x.device)

        # Normalize flow to grid scale (-1 to 1) - Important for F.grid_sample
        flow1_normalized_x = flow1[:, 0, :, :] / (width / 2)
        flow1_normalized_y = flow1[:, 1, :, :] / (height / 2)
        flow1_normalized = torch.stack((flow1_normalized_x, flow1_normalized_y), dim=1)  # N, 2, H, W
        flow1_normalized = flow1_normalized.permute(0, 2, 3, 1)  # N, H, W, 2 - channels last for grid_sample

        warped_image2_1 = F.grid_sample(
            image2,
            grid + flow1_normalized,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # Stage 2: Predict flow2 (residual flow) - Input is image1 and warped_image2_1
        stage2_input = torch.cat((image1, warped_image2_1), dim=1)  # Concatenate image1 and warped_image2_1 for stage 2 input
        flow2 = self.stage2_model(stage2_input)  # Model 2 predicts residual flow

        # Combine flows: Simple additive combination for now
        final_flow = flow1 + flow2  # Add flow2 (residual) to flow1

        return final_flow

    def _create_meshgrid(self, batch_size, height, width, device):  # Helper function for meshgrid
        x_grid = torch.linspace(-1.0, 1.0, width, device=device)
        y_grid = torch.linspace(-1.0, 1.0, height, device=device)

        x_mesh, y_mesh = torch.meshgrid(x_grid, y_grid, indexing="ij")  # Use indexing='ij' for consistent xy ordering

        # Stack and repeat for batch size
        meshgrid = torch.stack((x_mesh, y_mesh), dim=0).float()  # 2, H, W
        meshgrid = meshgrid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # N, 2, H, W
        return meshgrid.permute(0, 2, 3, 1)  # N, H, W, 2 - channels last for grid_sample


model = MotionVectorRegressionNetworkWithWarping(input_images=2, max_displacement=20).to(device)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location=device))
print(model)


def run_single(x):
    model.eval()

    X = torch.from_numpy(x).float()
    X = X.unsqueeze(0)
    X = X.to(device)

    pred = model(X)
    pred = pred.detach().cpu().numpy()

    return pred[0]


def run_batch(x):
    model.eval()

    X = torch.from_numpy(x).float()
    X = X.to(device)

    pred = model(X)
    pred = pred.detach().cpu().numpy()

    return pred

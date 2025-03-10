from typing import Tuple, List
import numpy as np
from skimage.util import img_as_float
import imageio
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.m4_combo import ComboMotionVectorRegressionNetwork
from models.m4_deeper import DeeperWiderMotionVectorRegressionNetwork
from models.m5_warp import MotionVectorRegressionNetworkWithWarping
from models.m5_warptest import MotionVectorRegressionNetworkWithWarping as MotionVectorRegressionNetworkWithWarpingTest

TILE_SIZE = 256
OVERLAP = 32
BATCH_SIZE = 32


@dataclass
class Tile:
    data: np.ndarray
    position: Tuple[int, int]  # (y, x) position in original image
    size: int
    overlap: int


def create_tiles(image: np.ndarray) -> List[Tile]:
    """Split image into overlapping tiles."""
    height, width = image.shape
    tiles = []

    for y in range(0, height, TILE_SIZE - OVERLAP):
        for x in range(0, width, TILE_SIZE - OVERLAP):

            # Calculate tile boundaries
            y_end = min(y + TILE_SIZE, height)
            x_end = min(x + TILE_SIZE, width)

            # Extract tile data
            tile_data = image[y:y_end, x:x_end]

            # Pad if necessary
            if tile_data.shape != (TILE_SIZE, TILE_SIZE):
                padded_data = np.zeros((TILE_SIZE, TILE_SIZE), dtype=tile_data.dtype)
                padded_data[: tile_data.shape[0], : tile_data.shape[1]] = tile_data
                tile_data = padded_data

            tiles.append(Tile(data=tile_data, position=(y, x), size=TILE_SIZE, overlap=OVERLAP))

    return tiles


def stitch_tiles_f(tiles: List[Tile], original_shape: tuple) -> np.ndarray:
    """Stitch tiles back together with average blending in overlap regions for colored images."""
    height, width = original_shape
    shape_with_channels = (height, width, 3)
    result = np.zeros(shape_with_channels, dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)  # Single-channel weight map

    for tile in tiles:
        y, x = tile.position
        y_end = min(y + tile.size, height)
        x_end = min(x + tile.size, width)

        # Create weight mask for smooth blending
        weight_mask = np.ones((y_end - y, x_end - x, 1), dtype=np.float32)

        # Apply feathering at edges
        if tile.overlap > 0:
            # Feather left edge
            if x > 0:
                weight_mask[:, : tile.overlap, :] *= np.linspace(0, 1, tile.overlap).reshape(1, -1, 1)
            # Feather right edge
            if x_end < width:
                weight_mask[:, -tile.overlap :, :] *= np.linspace(1, 0, tile.overlap).reshape(1, -1, 1)
            # Feather top edge
            if y > 0:
                weight_mask[: tile.overlap, :, :] *= np.linspace(0, 1, tile.overlap).reshape(-1, 1, 1)
            # Feather bottom edge
            if y_end < height:
                weight_mask[-tile.overlap :, :, :] *= np.linspace(1, 0, tile.overlap).reshape(-1, 1, 1)

            # Ensure even the first tile (0,0) is blended by using a small weight instead of 1
            if x == 0:
                weight_mask[:, : tile.overlap, :] *= np.linspace(0.5, 1, tile.overlap).reshape(1, -1, 1)
            if y == 0:
                weight_mask[: tile.overlap, :, :] *= np.linspace(0.5, 1, tile.overlap).reshape(-1, 1, 1)

        # Add weighted tile data
        result[y:y_end, x:x_end, :] += tile.data[: y_end - y, : x_end - x, :] * weight_mask
        weights[y:y_end, x:x_end, :] += weight_mask

    weights = np.where(weights == 0, 1, weights)  # Use np.where to avoid modifying the original weight structure
    result /= weights  # Broadcasting automatically applies normalization to all channels

    return result


if hasattr(torch, "accelerator"):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
else:
    device = "cpu"
print(f"Using {device} device")


def run_single(model, x):
    model.eval()

    X = torch.from_numpy(x).float()
    X = X.unsqueeze(0)
    X = X.to(device)

    pred = model(X)
    pred = pred.detach().cpu().numpy()

    return pred[0]


def run_batch(model, x):
    model.eval()

    X = torch.from_numpy(x).float()
    X = X.to(device)

    pred = model(X)
    pred = pred.detach().cpu().numpy()

    return pred


def process_image(image_path: str, next_image_path: str, folder: str, model_name: str) -> Tuple[np.ndarray, str] | None:
    """Processes an image using the trained model."""
    # Load the image by checking cache
    save_image_path = f"{folder}/{model_name}/{image_path.split('/')[-1].split('.')[0]}_processed.png"
    save_motion_path = f"{folder}/{model_name}/{image_path.split('/')[-1].split('.')[0]}_motion.npy"
    if os.path.exists(save_image_path) and os.path.exists(save_motion_path):
        return imageio.imread(save_image_path), save_motion_path

    image = imageio.imread(image_path)
    next_image = imageio.imread(next_image_path)

    # Load model

    model = None
    if model_name == "m4-combo":
        model = ComboMotionVectorRegressionNetwork(input_images=2).to(device)
        model.load_state_dict(torch.load("models/m4-combo.pt", weights_only=True, map_location=device))

    elif model_name == "m4-deeper":
        model = DeeperWiderMotionVectorRegressionNetwork(input_images=2).to(device)
        model.load_state_dict(torch.load("models/m4-deeper.pt", weights_only=True, map_location=device))

    elif model_name == "m5-warp":
        model = MotionVectorRegressionNetworkWithWarping(input_images=2, max_displacement=20).to(device)
        model.load_state_dict(torch.load("models/m5-warp.pt", weights_only=True, map_location=device))

    elif model_name == "m5-warptest":
        model = MotionVectorRegressionNetworkWithWarpingTest(input_images=2, max_displacement=20).to(device)
        model.load_state_dict(torch.load("models/m5-warptest.pt", weights_only=True, map_location=device))

        # Preprocess the images
        image = img_as_float(image).astype(np.float32)
        next_image = img_as_float(next_image).astype(np.float32)

    else:
        print(f"Unknown model: {model_name}")
        return None

    # Split the images into tiles
    tiles = create_tiles(image)
    next_tiles = create_tiles(next_image)

    batches = [[]]
    for tile1, tile2 in zip(tiles, next_tiles):
        if len(batches[-1]) >= BATCH_SIZE:
            batches.append([])

        batches[-1].append((tile1, tile2))

    displacement_tiles = []

    for batch in batches:
        X = np.array([[tile1.data, tile2.data] for (tile1, tile2) in batch], dtype=np.float32)
        batch_pred = run_batch(model, X)  # [2, H, W]

        for i, (tile1, tile2) in enumerate(batch):
            pred = batch_pred[i]
            pred = np.vstack((pred, np.zeros((1, pred.shape[1], pred.shape[2]))))
            pred = np.transpose(pred, (1, 2, 0))

            displacement_tiles.append(Tile(data=pred, position=tile1.position, size=tile1.size, overlap=tile1.overlap))

    # Clean up model

    del model

    # Stitch the tiles back together
    stitched_tiles = stitch_tiles_f(displacement_tiles, (image.shape[0], image.shape[1]))

    # stitched_image = (stitched_tiles - stitched_tiles.min()) / (stitched_tiles.max() - stitched_tiles.min() + 1e-12)
    stitched_image = (stitched_tiles + 2) / (4)
    stitched_image = np.clip(stitched_image * 255, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_image_path), exist_ok=True)

    # Save the stitched image to file for caching
    imageio.imwrite(save_image_path, stitched_image)

    # Remove the third channel
    stitched_tiles = stitched_tiles[:, :, :2]
    # stitched_tiles.shape = (H, W, 2)

    np.save(save_motion_path, stitched_tiles.astype(np.float64))

    return stitched_image, save_motion_path

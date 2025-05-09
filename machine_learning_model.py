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
OVERLAP = 64
BATCH_SIZE = 32

CENTER_SIZE = 64
CENTER_INSET = (TILE_SIZE - CENTER_SIZE) // 2


@dataclass
class Tile:
    data: np.ndarray
    position: Tuple[int, int]  # (y, x) position in original image
    # size: int
    # overlap: int


def create_tiles(image: np.ndarray) -> List[Tile]:
    """Split image into overlapping tiles."""
    height, width = image.shape
    tiles = []

    for y in range(-CENTER_INSET, height - CENTER_INSET, CENTER_SIZE):
        for x in range(-CENTER_INSET, width - CENTER_INSET, CENTER_SIZE):

            # Calculate tile boundaries
            y_start = max(y, 0)
            x_start = max(x, 0)
            y_end = min(y + TILE_SIZE, height)
            x_end = min(x + TILE_SIZE, width)

            shift_x = x_start - x
            shift_y = y_start - y

            # Extract tile data
            tile_data = image[y_start:y_end, x_start:x_end]

            # Pad if necessary
            if tile_data.shape != (TILE_SIZE, TILE_SIZE):
                padded_data = np.zeros((TILE_SIZE, TILE_SIZE), dtype=tile_data.dtype)
                padded_data[shift_y : shift_y + tile_data.shape[0], shift_x : shift_x + tile_data.shape[1]] = tile_data
                tile_data = padded_data

            tiles.append(Tile(data=tile_data.copy(), position=(y, x)))

    return tiles


def stitch_tiles_f(tiles: List[Tile], original_shape: tuple) -> np.ndarray:
    """Stitch tiles back together with average blending in overlap regions for colored images."""
    height, width = original_shape
    result = np.zeros((height, width, 3), dtype=np.float32)

    for tile in tiles:
        y, x = tile.position

        crop_x_start = max(x + CENTER_INSET, 0)
        crop_y_start = max(y + CENTER_INSET, 0)
        crop_x_end = min(x + TILE_SIZE - CENTER_INSET, width)
        crop_y_end = min(y + TILE_SIZE - CENTER_INSET, height)

        tile_x_start = CENTER_INSET
        tile_x_end = min((TILE_SIZE - CENTER_INSET) + x, width) - x
        tile_y_start = CENTER_INSET
        tile_y_end = min((TILE_SIZE - CENTER_INSET) + y, height) - y

        # print("x_end", TILE_SIZE - CENTER_INSET, " x_end_pos", x + TILE_SIZE - CENTER_INSET, " width", width, " back", tile_x_end)
        # print("y_end", TILE_SIZE - CENTER_INSET, " y_end_pos", y + TILE_SIZE - CENTER_INSET, " height", height, " back", tile_y_end)

        result[crop_y_start:crop_y_end, crop_x_start:crop_x_end, :] = tile.data[tile_y_start:tile_y_end, tile_x_start:tile_x_end]

    return result


# def create_tiles(image: np.ndarray) -> List[Tile]:
#     """Split image into overlapping tiles."""
#     height, width = image.shape
#     tiles = []

#     for y in range(0, height, TILE_SIZE - OVERLAP):
#         for x in range(0, width, TILE_SIZE - OVERLAP):

#             # Calculate tile boundaries
#             y_end = min(y + TILE_SIZE, height)
#             x_end = min(x + TILE_SIZE, width)

#             # Extract tile data
#             tile_data = image[y:y_end, x:x_end]

#             # Pad if necessary
#             if tile_data.shape != (TILE_SIZE, TILE_SIZE):
#                 padded_data = np.zeros((TILE_SIZE, TILE_SIZE), dtype=tile_data.dtype)
#                 padded_data[: tile_data.shape[0], : tile_data.shape[1]] = tile_data
#                 tile_data = padded_data

#             tiles.append(Tile(data=tile_data, position=(y, x)))

#     return tiles


# def stitch_tiles_f(tiles: List[Tile], original_shape: tuple) -> np.ndarray:
#     """Stitch tiles back together with average blending in overlap regions for colored images."""
#     height, width = original_shape
#     shape_with_channels = (height, width, 3)
#     result = np.zeros(shape_with_channels, dtype=np.float32)
#     weights = np.zeros((height, width, 1), dtype=np.float32)  # Single-channel weight map

#     for tile in tiles:
#         y, x = tile.position
#         y_end = min(y + TILE_SIZE, height)
#         x_end = min(x + TILE_SIZE, width)

#         # Create weight mask for smooth blending
#         weight_mask = np.ones((y_end - y, x_end - x, 1), dtype=np.float32)

#         # Apply feathering at edges
#         if OVERLAP > 0:
#             # Feather left edge
#             if x > 0:
#                 weight_mask[:, :OVERLAP, :] *= np.linspace(0, 1, OVERLAP).reshape(1, -1, 1)
#             # Feather right edge
#             if x_end < width:
#                 weight_mask[:, -OVERLAP:, :] *= np.linspace(1, 0, OVERLAP).reshape(1, -1, 1)
#             # Feather top edge
#             if y > 0:
#                 weight_mask[:OVERLAP, :, :] *= np.linspace(0, 1, OVERLAP).reshape(-1, 1, 1)
#             # Feather bottom edge
#             if y_end < height:
#                 weight_mask[-OVERLAP:, :, :] *= np.linspace(1, 0, OVERLAP).reshape(-1, 1, 1)

#             # Ensure even the first tile (0,0) is blended by using a small weight instead of 1
#             if x == 0:
#                 weight_mask[:, :OVERLAP, :] *= np.linspace(0.5, 1, OVERLAP).reshape(1, -1, 1)
#             if y == 0:
#                 weight_mask[:OVERLAP, :, :] *= np.linspace(0.5, 1, OVERLAP).reshape(-1, 1, 1)

#         # Add weighted tile data
#         result[y:y_end, x:x_end, :] += tile.data[: y_end - y, : x_end - x, :] * weight_mask
#         weights[y:y_end, x:x_end, :] += weight_mask

#     weights = np.where(weights == 0, 1, weights)  # Use np.where to avoid modifying the original weight structure
#     result /= weights  # Broadcasting automatically applies normalization to all channels

#     return result


# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_uv = flow_uv.copy()
    assert flow_uv.ndim == 3, "input flow must have three dimensions"

    if flow_uv.shape[0] == 2:
        flow_uv = flow_uv.transpose(1, 2, 0)

    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


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
    save_tile_path = f"{folder}/{model_name}/{image_path.split('/')[-1].split('.')[0]}/"
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

    # print(tiles[8].data.shape)
    # print(tiles[8].data.min(), tiles[0].data.max())

    batches = [[]]
    for tile1, tile2 in zip(tiles, next_tiles):
        if len(batches[-1]) >= BATCH_SIZE:
            batches.append([])

        batches[-1].append((tile1, tile2))

    displacement_tiles: list[Tile] = []

    for batch in batches:
        X = np.array([[tile1.data, tile2.data] for (tile1, tile2) in batch], dtype=np.float32)
        batch_pred = run_batch(model, X)  # [2, H, W]

        for i, (tile1, tile2) in enumerate(batch):
            pred = batch_pred[i]
            pred = np.vstack((pred, np.zeros((1, pred.shape[1], pred.shape[2]))))
            pred = np.transpose(pred, (1, 2, 0))

            displacement_tiles.append(Tile(data=pred, position=tile1.position))  # size=tile1.size, overlap=tile1.overlap

    # Clean up model

    del model

    for tile in displacement_tiles:
        tile_tile = tile.data[:, :, :2].astype(np.float64)
        os.makedirs(save_tile_path, exist_ok=True)
        np.save(save_tile_path + str(tile.position[0]) + "_" + str(tile.position[1]) + ".npy", tile_tile)

    # Stitch the tiles back together
    stitched_tiles = stitch_tiles_f(displacement_tiles, (image.shape[0], image.shape[1]))

    # stitched_image = (stitched_tiles - stitched_tiles.min()) / (stitched_tiles.max() - stitched_tiles.min() + 1e-12)
    # stitched_image = (stitched_tiles + 2) / (4)
    # stitched_image = np.clip(stitched_image * 255, 0, 255).astype(np.uint8)
    stitched_image = flow_to_image(stitched_tiles[:, :, :2])

    os.makedirs(os.path.dirname(save_image_path), exist_ok=True)

    # Save the stitched image to file for caching
    imageio.imwrite(save_image_path, stitched_image)

    # Remove the third channel
    stitched_tiles = stitched_tiles[:, :, :2]
    # stitched_tiles.shape = (H, W, 2)

    np.save(save_motion_path, stitched_tiles.astype(np.float64))

    return stitched_image, save_motion_path

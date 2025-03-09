import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_data(ax, file_path, frame_idx, _num_images):
    """Reads CSV, plots Force vs. Point on an existing axis, and displays the current force and average force."""
    global df, num_images, vline, force_annotation, average_annotation, loaded

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        loaded = False
        ax.clear()
        ax.text(0.5, 0.5, "File not found! Please check the path.", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontsize=12, color="red")
        return None

    df = pd.read_csv(file_path)
    num_images = _num_images

    ax.clear()  # Clear previous data if any
    ax.plot(df["Point"], df["Force"], label="Force vs. Point")

    # Initial red vertical line at x = 0.2 (adjustable later)
    vline = ax.axvline(x=0.2, color="red", linestyle="--", label="Threshold")

    ax.set_xlabel("Point")
    ax.set_ylabel("Force")
    ax.legend()

    loaded = True

    return vline  # Return vline for external updates


def update_red_line(ax, frame_idx):
    """Updates the red vertical line position and updates force annotations."""
    global df, num_images

    if not loaded:
        return 0, 0, 0

    # Update the red vertical line position based on frame_idx
    x_pos = (frame_idx / num_images) * max(ax.get_xlim())  # You can tweak this formula
    vline.set_xdata([x_pos])

    # Update the force annotations
    image_range_start = int((frame_idx / num_images) * len(df))
    image_range_end = int(((frame_idx + 1) / num_images) * len(df))

    force_at_frame = df["Force"].iloc[image_range_start]
    average_force = df["Force"].iloc[image_range_start:image_range_end].mean()
    cumulative_force = df["Force"].iloc[0:image_range_end].sum()

    return force_at_frame, average_force, cumulative_force

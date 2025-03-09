import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.widgets as widgets
import numpy as np
import argparse
import os
import glob
import imageio
import cv2
from plot_force import plot_data, update_red_line

from machine_learning_model import process_image
from strain_calculations import calculate_strain

# Define available models
AVAILABLE_MODELS = ["model_A", "model_B", "model_C"]


# Load images from the folder
def load_images(folder):
    """Loads image file paths from the folder, sorted by numerical order."""
    image_files = sorted(glob.glob(os.path.join(folder, "*")), key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
    return image_files


class ImageGrid:
    def __init__(self, force_data_path, image_folder, cache_folder, initial_model="model_A"):
        self.force_data_path = force_data_path
        self.image_folder = image_folder
        self.cache_folder = cache_folder
        self.image_files = load_images(image_folder)
        self.num_images = len(self.image_files)
        self.frame_idx = 0
        self.playing = False
        self.processed_data = None
        self.strain_data = None
        self.selected_model = initial_model

        # Create figure and grid
        self.fig = plt.figure(figsize=(12, 10))
        self.gs = gridspec.GridSpec(4, 4, figure=self.fig, height_ratios=[1, 1, 1, 0.3])

        # Top row: Force Graph
        self.ax_force = self.fig.add_subplot(self.gs[0, :3])
        self.vline = plot_data(self.ax_force, self.force_data_path, self.frame_idx, self.num_images)

        # Top-right: Data Info
        self.ax_info = self.fig.add_subplot(self.gs[0, 3])
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        self.ax_info.set_frame_on(False)

        # Middle row: Image display
        self.ax_img1 = self.fig.add_subplot(self.gs[1, 0])
        self.ax_img2 = self.fig.add_subplot(self.gs[1, 1])
        self.ax_processed = self.fig.add_subplot(self.gs[1, 2])

        # Middle-right: Model selection Radio Buttons
        self.ax_radio = self.fig.add_subplot(self.gs[1, 3])
        self.radio_buttons = widgets.RadioButtons(self.ax_radio, AVAILABLE_MODELS, active=AVAILABLE_MODELS.index(self.selected_model))
        self.radio_buttons.on_clicked(self.set_model)

        # Bottom row: Strain visualizations
        self.ax_strain1 = self.fig.add_subplot(self.gs[2, 0])
        self.ax_strain2 = self.fig.add_subplot(self.gs[2, 1])
        self.ax_strain3 = self.fig.add_subplot(self.gs[2, 2])

        # Export Buttons Grid
        self.ax_export_grid = self.fig.add_subplot(self.gs[2, 3])
        self.ax_export_grid.axis("off")
        export_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=self.gs[2, 3])
        self.ax_export_motion = self.fig.add_subplot(export_gs[0, 0])
        self.ax_export_strain = self.fig.add_subplot(export_gs[1, 0])
        self.ax_export_frames = self.fig.add_subplot(export_gs[2, 0])

        # Bottom controls: Buttons & slider
        self.ax_left = self.fig.add_subplot(self.gs[3, 0])
        self.ax_slider = self.fig.add_subplot(self.gs[3, 1])
        self.ax_right = self.fig.add_subplot(self.gs[3, 2])
        self.ax_process = self.fig.add_subplot(self.gs[3, 3])

        self.btn_left = widgets.Button(self.ax_left, "←")
        self.btn_playpause = widgets.Button(self.ax_slider, "Play/Pause")
        self.btn_right = widgets.Button(self.ax_right, "→")
        self.btn_process = widgets.Button(self.ax_process, "Run ML Model")
        self.btn_export_motion = widgets.Button(self.ax_export_motion, "Motion GIF")
        self.btn_export_strain = widgets.Button(self.ax_export_strain, "Strain GIF")
        self.btn_export_frames = widgets.Button(self.ax_export_frames, "Export Frames")

        # Connect button events
        self.btn_left.on_clicked(self.prev_frame)
        self.btn_playpause.on_clicked(self.toggle_play)
        self.btn_process.on_clicked(self.run_model_on_all_images)
        self.btn_right.on_clicked(self.next_frame)
        self.btn_export_motion.on_clicked(self.export_motion_gif)
        self.btn_export_strain.on_clicked(self.export_strain_gif)
        self.btn_export_frames.on_clicked(self.export_frames_gif)

        # Initial display
        self.update_images()
        self.update_info_text()

    def set_model(self, model):
        """Sets the selected ML model based on radio button input."""
        self.selected_model = model
        print(f"Selected model: {self.selected_model}")

    def update_images(self):
        """Updates the image displays based on the current frame index."""
        self.ax_img1.clear()
        self.ax_img2.clear()
        self.ax_processed.clear()
        self.ax_strain1.clear()
        self.ax_strain2.clear()
        self.ax_strain3.clear()

        if self.num_images > 0:
            # Load and display original images
            img1 = cv2.imread(self.image_files[self.frame_idx], cv2.IMREAD_GRAYSCALE)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            self.ax_img1.imshow(img1, cmap="gray")
            self.ax_img1.set_title(f"Frame {self.frame_idx + 1}")

            next_idx = (self.frame_idx + 1) % self.num_images
            img2 = cv2.imread(self.image_files[next_idx], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            self.ax_img2.imshow(img2, cmap="gray")
            self.ax_img2.set_title(f"Frame {next_idx + 1}")

            # Display processed displacement data
            if self.processed_data and self.frame_idx < len(self.processed_data):  # Check if processed data is available for current frame
                disp_data = self.processed_data[self.frame_idx]
                disp_img = self.create_displacement_image(disp_data)
                self.ax_processed.imshow(disp_img)
                self.ax_processed.set_title("Processed Frame")

            # Display strain components
            if self.strain_data and self.frame_idx < len(self.strain_data):  # Check if strain data is available for current frame
                strain_xx, strain_yy, strain_xy = self.strain_data[self.frame_idx]  # Now using 2D arrays

                # Display strainXX
                self.ax_strain1.imshow(strain_xx, cmap="coolwarm", origin="upper", aspect="equal")
                self.ax_strain1.set_title("strainXX")

                # Display strainYY
                self.ax_strain2.imshow(strain_yy, cmap="coolwarm", origin="upper", aspect="equal")
                self.ax_strain2.set_title("strainYY")

                # Display strainXY
                self.ax_strain3.imshow(strain_xy, cmap="coolwarm", origin="upper", aspect="equal")
                self.ax_strain3.set_title("strainXY")

                # Force a redraw
                self.fig.canvas.draw_idle()

        self.ax_img1.axis("off")
        self.ax_img2.axis("off")
        self.ax_processed.axis("off")
        self.ax_strain1.axis("off")
        self.ax_strain2.axis("off")
        self.ax_strain3.axis("off")

        update_red_line(self.ax_force, self.frame_idx)

        plt.draw()

    def update_info_text(self):
        """Updates the top-right text box with image count and current frame info."""
        self.ax_info.clear()
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        self.ax_info.set_frame_on(False)

        text = f"Loaded Data Info:\nImages: {self.num_images}\nCurrent Frames: {self.frame_idx + 1} & {(self.frame_idx + 2) % self.num_images + 1}\nModel: {self.selected_model}"  # Added model info
        self.ax_info.text(0.1, 0.5, text, fontsize=10, fontweight="bold")

        plt.draw()

    def next_frame(self, event=None):
        """Moves to the next frame and updates display."""
        self.frame_idx = (self.frame_idx + 1) % self.num_images
        self.update_images()
        self.update_info_text()

    def prev_frame(self, event=None):
        """Moves to the previous frame and updates display."""
        self.frame_idx = (self.frame_idx - 1) % self.num_images
        self.update_images()
        self.update_info_text()

    def toggle_play(self, event=None):
        """Starts/stops automatic frame advancing."""
        self.playing = not self.playing
        if self.playing:
            self.auto_advance()

    def auto_advance(self):
        """Automatically advances frames when play is active."""
        if self.playing:
            self.next_frame()
            self.fig.canvas.flush_events()
            self.fig.canvas.start_event_loop(0.01)  # Delay for smooth animation
            self.auto_advance()

    def run_model_on_all_images(self, event=None):
        """Processes all images through the ML model and computes strain using the selected model."""
        # Loading text
        self.btn_process.label.set_text("Loading...")
        self.btn_process.set_active(False)

        self.processed_data = []
        self.strain_data = []
        os.makedirs(self.cache_folder, exist_ok=True)

        for i in range(len(self.image_files) - 1):  # Stop one before the end
            img_path = self.image_files[i]
            next_image_path = self.image_files[i + 1]

            disp_image, disp_data_path = process_image(img_path, next_image_path, self.cache_folder, model_name=self.selected_model)  # Pass selected_model
            self.processed_data.append(disp_image)

            # Compute strain from displacement data
            strain_result = calculate_strain(disp_data_path)
            self.strain_data.append(strain_result)

        print(f"Processed {len(self.processed_data)} images using model: {self.selected_model}")

        # Update the display with the processed and strain images
        self.update_images()
        self.btn_process.label.set_text("Run ML Model")
        self.btn_process.set_active(True)  # Re-enable button

    def create_displacement_image(self, disp_data):
        """Creates a red-green visualization from displacement data."""
        # H W 3 -> 3 H W
        disp_data = np.transpose(disp_data, (2, 0, 1))

        disp_x = disp_data[0]  # Horizontal displacement
        disp_y = disp_data[1]  # Vertical displacement

        disp_x = disp_x / 255
        disp_y = disp_y / 255

        # Create red-green visualization
        disp_img = np.zeros((disp_x.shape[0], disp_x.shape[1], 3)) + 0.5
        disp_img[..., 0] = disp_x  # Red channel
        disp_img[..., 1] = disp_y  # Green channel

        return disp_img

    def export_frames_gif(self, event=None):
        """Exports a GIF of the original frames side-by-side."""
        if not self.image_files:
            print("No images loaded to export.")
            self.btn_export_frames.label.set_text("No Images (Check Image Folder)")
            return

        self.btn_export_frames.label.set_text("Exporting...")
        self.btn_export_frames.set_active(False)

        output_filename = "frames.gif"  # Changed filename to frames.gif
        print(f"Exporting frames GIF to {output_filename}...")
        motion_frames = []
        for i in range(0, len(self.image_files) - 1):
            img1 = cv2.imread(self.image_files[i])
            img2 = cv2.imread(self.image_files[i + 1])
            combined_img = np.concatenate((img1, img2), axis=1)  # Combine side-by-side
            motion_frames.append(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))  # Convert to RGB for imageio

        imageio.mimsave(output_filename, motion_frames, fps=10)  # Adjust fps as needed
        print(f"Frames GIF exported to {output_filename}")

        self.btn_export_frames.label.set_text("Export Frames GIF")
        self.btn_export_frames.set_active(True)

    def export_motion_gif(self, event=None):
        """Exports a GIF of the processed motion (displacement) data."""
        if not self.processed_data:
            print("No processed motion data available to export. Run ML model first.")
            self.btn_export_motion.label.set_text("No Motion Data (Click Run ML Model)")
            return

        self.btn_export_motion.label.set_text("Exporting...")
        self.btn_export_motion.set_active(False)

        output_filename = "motion.gif"
        print(f"Exporting motion GIF to {output_filename}...")
        motion_frames = []
        for disp_data in self.processed_data:
            disp_img_rgb = self.create_displacement_image(disp_data)  # Get RGB displacement image
            motion_frames.append(np.uint8(disp_img_rgb * 255))  # Scale to 0-255 and convert to uint8

        imageio.mimsave(output_filename, motion_frames, fps=10)  # Adjust fps as needed
        print(f"Motion GIF exported to {output_filename}")

        self.btn_export_motion.label.set_text("Export Motion GIF")
        self.btn_export_motion.set_active(True)

    def export_strain_gif(self, event=None):
        """Exports a GIF of combined strain visualizations."""
        if not self.strain_data:
            print("No strain data available to export. Run ML model first.")
            self.btn_export_strain.label.set_text("No Strain Data (Click Run ML Model)")
            return

        self.btn_export_strain.label.set_text("Exporting...")
        self.btn_export_strain.set_active(False)

        output_filename = "strain.gif"
        print(f"Exporting strain GIF to {output_filename}...")

        strain_frames = []
        np_strain_data = np.array(self.strain_data)  # Convert list of tuples to a NumPy array

        # Extract strain components
        strain_xx_all = np_strain_data[:, 0]  # (frames, H, W)
        strain_yy_all = np_strain_data[:, 1]
        strain_xy_all = np_strain_data[:, 2]

        # Compute global min/max across all frames and components
        global_min = np.min([strain_xx_all.min(), strain_yy_all.min(), strain_xy_all.min()])
        global_max = np.max([strain_xx_all.max(), strain_yy_all.max(), strain_xy_all.max()])

        # Loop through frames
        for frame_idx in range(len(self.strain_data)):
            strain_xx = strain_xx_all[frame_idx]
            strain_yy = strain_yy_all[frame_idx]
            strain_xy = strain_xy_all[frame_idx]

            # Normalize strain components to 0-1
            norm_strain_xx = (strain_xx - global_min) / (global_max - global_min + 1e-6)
            norm_strain_yy = (strain_yy - global_min) / (global_max - global_min + 1e-6)
            norm_strain_xy = (strain_xy - global_min) / (global_max - global_min + 1e-6)

            print(f"Normalized strain components (Frame {frame_idx + 1}):")
            print(f"  strain_xx - shape: {norm_strain_xx.shape}, min: {norm_strain_xx.min()}, max: {norm_strain_xx.max()}")
            print(f"  strain_yy - shape: {norm_strain_yy.shape}, min: {norm_strain_yy.min()}, max: {norm_strain_yy.max()}")
            print(f"  strain_xy - shape: {norm_strain_xy.shape}, min: {norm_strain_xy.min()}, max: {norm_strain_xy.max()}")

            # Stack strain components as RGB channels
            combined_strain_gray = np.stack((norm_strain_xx, norm_strain_yy, norm_strain_xy), axis=-1)  # Shape: (H, W, 3)

            # Convert to uint8 (0-255 range)
            strain_frames.append(np.uint8(combined_strain_gray * 255))

        # Save GIF with a reasonable FPS
        imageio.mimsave(output_filename, strain_frames, fps=10)  # Adjust fps as needed
        print(f"Strain GIF exported to {output_filename}")

        self.btn_export_strain.label.set_text("Export Strain GIF")
        self.btn_export_strain.set_active(True)


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_data_path", type=str, default="data/force_data.csv")
    parser.add_argument("--image_folder", type=str, default="data/raw_images")
    parser.add_argument("--model", type=str, default="model_A", choices=AVAILABLE_MODELS, help="Select ML model for processing")  # CLI argument for model selection
    args = parser.parse_args()

    force_data_path = args.force_data_path
    image_folder = args.image_folder
    model_name = args.model  # Get model name from command line

    cache_folder = "cached_processed_images"
    img_grid = ImageGrid(force_data_path, image_folder, cache_folder, initial_model=model_name)  # Pass initial_model to ImageGrid
    plt.show()

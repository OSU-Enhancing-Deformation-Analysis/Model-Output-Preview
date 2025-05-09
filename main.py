import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.widgets as widgets
from matplotlib.patches import Rectangle
import numpy as np
import argparse
import os
import platform
import dload
import time
import glob
import imageio
import cv2
from plot_force import plot_data, update_red_line
from matplotlib.colors import LinearSegmentedColormap

from machine_learning_model import process_image
from strain_calculations import calculate_strain


# Define available models
AVAILABLE_MODELS = ["m4-combo", "m4-deeper", "m5-warptest"]

# Define a more colorful colormap for strain visualization
strain_cmap = LinearSegmentedColormap.from_list("strain_cmap", ["blue", "cyan", "green", "yellow", "red"])


# Load images from the folder
def load_images(folder):
    """Loads image file paths from the folder, sorted by numerical order."""
    image_files = sorted(glob.glob(os.path.join(folder, "*")), key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
    return image_files


class ImageGrid:
    def __init__(self, force_data_path, image_folder, cache_folder, strain_calc_path):
        self.force_data_path = force_data_path
        self.image_folder = image_folder
        self.cache_folder = cache_folder
        self.strain_calc_path = strain_calc_path
        self.image_files = load_images(image_folder)
        self.num_images = len(self.image_files)
        self.frame_idx = 0
        self.playing = False
        self.processed_motion_images = None
        self.processed_motion_data_paths = None
        self.strain_data = None
        self.selected_model = AVAILABLE_MODELS[0]
        self.motion_sum_data = None  # Store sum of motion data
        self.cumulative_strain_data = None  # Store cumulative strain data
        self.subset_size = 5  # Store subset size
        # Create figure and grid
        self.fig = plt.figure(figsize=(14, 8))  # Increased figure width for new column
        self.gs = gridspec.GridSpec(4, 6, figure=self.fig, height_ratios=[1, 1, 1, 0.3])  # 4x6 Grid

        # Top row: Force Graph
        self.ax_force = self.fig.add_subplot(self.gs[0, :3])  # Force graph takes first two columns
        self.vline = plot_data(self.ax_force, self.force_data_path, self.frame_idx, self.num_images)

        # Top-middle: Force Info
        self.ax_force_info = self.fig.add_subplot(self.gs[0, 3])  # New column, top row
        self.ax_force_info.set_xticks([])
        self.ax_force_info.set_yticks([])
        self.ax_force_info.set_frame_on(False)

        # Top-right: Data Info
        self.ax_info = self.fig.add_subplot(self.gs[0, 4:])  # Data info takes last two columns
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        self.ax_info.set_frame_on(False)

        # Middle row: Image display
        self.ax_img1 = self.fig.add_subplot(self.gs[1, 0])
        self.ax_img2 = self.fig.add_subplot(self.gs[1, 1])
        self.ax_processed = self.fig.add_subplot(self.gs[1, 2])
        self.ax_motion_sum = self.fig.add_subplot(self.gs[1, 3])  # New column, middle row
        self.ax_motion_sum.set_xticks([])
        self.ax_motion_sum.set_yticks([])
        self.ax_motion_sum.set_frame_on(True)  # Keep frame for visual separation
        self.ax_motion_sum_last5 = self.fig.add_subplot(self.gs[1, 4])  # New column, middle row
        self.ax_motion_sum_last5.set_xticks([])
        self.ax_motion_sum_last5.set_yticks([])
        self.ax_motion_sum_last5.set_frame_on(True)  # Keep frame for visual separation

        # Middle-right: Model selection Radio Buttons
        self.ax_radio = self.fig.add_subplot(self.gs[1, 5])  # Radio buttons in the last column
        self.radio_buttons = widgets.RadioButtons(self.ax_radio, AVAILABLE_MODELS, active=AVAILABLE_MODELS.index(self.selected_model))
        self.radio_buttons.on_clicked(self.set_model)

        # Bottom row: Strain visualizations
        self.ax_strain1 = self.fig.add_subplot(self.gs[2, 0])
        self.ax_strain2 = self.fig.add_subplot(self.gs[2, 1])
        self.ax_strain3 = self.fig.add_subplot(self.gs[2, 2])
        self.ax_strain_combined = self.fig.add_subplot(self.gs[2, 3])  # Combined strain
        self.ax_strain_cumulative = self.fig.add_subplot(self.gs[2, 4])  # Cumulative Strain

        # Export Buttons Grid - moved to the last column
        self.ax_export_grid = self.fig.add_subplot(self.gs[2, 5])
        self.ax_export_grid.axis("off")
        export_gs = gridspec.GridSpecFromSubplotSpec(7, 1, subplot_spec=self.gs[2, 5])  # Last column
        self.ax_export_frames = self.fig.add_subplot(export_gs[0, 0])
        self.ax_export_motion = self.fig.add_subplot(export_gs[1, 0])
        self.ax_export_cumulative_motion = self.fig.add_subplot(export_gs[2, 0])
        self.ax_export_last5_motion = self.fig.add_subplot(export_gs[3, 0])
        self.ax_export_strain = self.fig.add_subplot(export_gs[4, 0])
        self.ax_export_cumulative_strain = self.fig.add_subplot(export_gs[5, 0])
        self.ax_export_last5_strain = self.fig.add_subplot(export_gs[6, 0])

        # Bottom controls: Buttons & slider
        self.ax_slider_subset_size = self.fig.add_subplot(self.gs[3, 0:2])  # Slider for subs
        self.ax_left = self.fig.add_subplot(self.gs[3, 2])
        self.ax_playpause = self.fig.add_subplot(self.gs[3, 3])
        self.ax_right = self.fig.add_subplot(self.gs[3, 4])
        self.ax_process = self.fig.add_subplot(self.gs[3, 5])
        # self.ax_empty_control = self.fig.add_subplot(self.gs[3, 4:])  # Empty for alignment

        control_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=self.gs[3, 0:2], height_ratios=[1, 1], hspace=0.35)
        self.ax_slider_subset_size = self.fig.add_subplot(control_gs[0, 0])
        self.slider_subset_size = widgets.Slider(
            ax=self.ax_slider_subset_size,
            label="Subset Size",
            valmin=2,
            valmax=15,
            valinit=self.subset_size,
            valstep=1,
        )
        self.slider_subset_size.on_changed(self.update_subset_size)

        # frame gap
        self.frame_gap = 1  # <-- default
        self.ax_slider_frame_gap = self.fig.add_subplot(control_gs[1, 0])
        self.slider_frame_gap = widgets.Slider(
            ax=self.ax_slider_frame_gap,
            label="Frame Gap",
            valmin=1,
            valmax=max(1, self.num_images - 1),  # safeguard
            valinit=self.frame_gap,
            valstep=1,
        )
        self.slider_frame_gap.on_changed(self.update_frame_gap)

        self.btn_left = widgets.Button(self.ax_left, "←")
        self.btn_playpause = widgets.Button(self.ax_playpause, "Play/Pause")
        self.btn_right = widgets.Button(self.ax_right, "→")
        self.btn_process = widgets.Button(self.ax_process, "Run ML Model")
        self.btn_export_frames = widgets.Button(self.ax_export_frames, "Export Frames GIF")
        self.btn_export_motion = widgets.Button(self.ax_export_motion, "Motion GIF")
        self.btn_export_cumulative_motion = widgets.Button(self.ax_export_cumulative_motion, "Cumulative Motion GIF")
        self.btn_export_last5_motion = widgets.Button(self.ax_export_last5_motion, "Last 5 Motion GIF")
        self.btn_export_strain = widgets.Button(self.ax_export_strain, "Strain GIF")
        self.btn_export_cumulative_strain = widgets.Button(self.ax_export_cumulative_strain, "Cumulative Strain GIF")
        self.btn_export_last5_strain = widgets.Button(self.ax_export_last5_strain, "Last 5 Strain GIF")
        self.slider_subset_size = widgets.Slider(
            ax=self.ax_slider_subset_size, label="Subset Size", valmin=2, valmax=15, valinit=self.subset_size, valstep=1  # Minimum subset size  # Maximum subset size - adjust as needed
        )
        self.slider_subset_size.on_changed(self.update_subset_size)  # C

        # Connect button events
        self.btn_left.on_clicked(self.prev_frame)
        self.btn_playpause.on_clicked(self.toggle_play)
        self.btn_process.on_clicked(self.run_model_on_all_images)
        self.btn_right.on_clicked(self.next_frame)
        self.btn_export_frames.on_clicked(self.export_frames_gif)
        self.btn_export_motion.on_clicked(self.export_motion_gif)
        self.btn_export_cumulative_motion.on_clicked(self.export_cumulative_motion_gif)
        self.btn_export_last5_motion.on_clicked(self.export_last5_motion_gif)
        self.btn_export_strain.on_clicked(self.export_strain_gif)
        self.btn_export_cumulative_strain.on_clicked(self.export_cumulative_strain_gif)
        self.btn_export_last5_strain.on_clicked(self.export_last5_strain_gif)

        # Hide empty axes
        # self.ax_empty_control.axis("off")

        # Initial display
        self.update_images()
        self.update_info_text()
        self.update_force_info(0, 0, 0)  # Initialize force info
        self.update_motion_sum()  # Initialize motion sum
        self.update_motion_sum_last_5()  # Initialize last 5 motion sum
        self.update_combined_strain()  # Initialize combined strain
        self.update_cumulative_strain()  # Initialize cumulative strain

    def update_frame_gap(self, val):
        """Callback when the Frame-Gap slider is moved."""
        self.frame_gap = int(self.slider_frame_gap.val)
        print(f"Frame gap changed to: {self.frame_gap}")
        # Invalidate any previously-cached motion / strain results
        self.processed_motion_images = None
        self.processed_motion_data_paths = None
        self.motion_sum_data = None
        self.strain_data = None
        self.cumulative_strain_data = None
        self.update_info_text()
        self.update_images()

    def set_model(self, model):
        """Sets the selected ML model based on radio button input."""
        self.selected_model = model
        print(f"Selected model changed to: {self.selected_model}. Running ML model...")
        self.run_model_on_all_images()  # Call run_model when model is changed

    def update_subset_size(self, val):
        """Updates the subset size and re-runs strain analysis."""
        self.subset_size = int(self.slider_subset_size.val)  # Get integer value from slider
        plt.pause(0.01)
        print(f"Subset size changed to: {self.subset_size}")
        if self.processed_motion_images is not None:  # Only re-run if processed data exists
            self.recalculate_strain_with_new_subset()

    def recalculate_strain_with_new_subset(self):
        """Recalculates strain data with the new subset size and updates display."""
        if self.processed_motion_images is None or self.processed_motion_data_paths is None:
            print("No processed data available to recalculate strain. Run ML model first.")
            return

        progress_ax = self.fig.add_axes((0.3, 0.48, 0.4, 0.03))
        progress_bar = Rectangle((0, 0), 0, 1, color="green")
        progress_ax.add_patch(progress_bar)
        progress_ax.set_axis_off()
        progress_ax.set_xlim(0, 1)
        progress_ax.set_ylim(0, 1)
        progress_text = self.fig.text(0.5, 0.5, "0%", ha="center", va="center", fontsize=12)
        progress_ax.set_title("Recalculating Strain...")

        self.strain_data = []  # Clear old strain data
        self.cumulative_strain_data = []  # Clear cumulative strain data
        cumulative_strain = None  # Reset cumulative strain

        print("Recalculating strain with new subset size...")
        # First, submit all tasks to the thread pool
        futures = []
        for disp_data_path in self.processed_motion_data_paths:
            future = calculate_strain(self.strain_calc_path, disp_data_path, self.selected_model, self.subset_size, force_rerun=True)
            futures.append(future)

        # Process results as they complete
        cumulative_strain = None
        self.strain_data = []
        self.cumulative_strain_data = []

        last_update_time = time.time()
        update_interval = 1.0  # Update UI at most every 0.5 seconds

        for i, future in enumerate(futures):
            # This will wait for each future to complete
            strain_result = future.result()
            self.strain_data.append(strain_result)

            # Accumulate strain
            if strain_result is not None:
                strain_xx, strain_yy, strain_xy = strain_result
                current_strain = np.stack([strain_xx, strain_yy, strain_xy], axis=-1).astype(np.float32)

                if cumulative_strain is None:
                    cumulative_strain = current_strain.copy()
                else:
                    cumulative_strain += current_strain
                self.cumulative_strain_data.append(cumulative_strain.copy())

            current_time = time.time()
            if current_time - last_update_time > update_interval:
                progress = (i + 1) / len(futures)
                progress_bar.set_width(progress)
                progress_text.set_text(f"{int(progress*100)}%")
                self.fig.canvas.draw_idle()
                plt.pause(0.01)
                last_update_time = current_time

        progress_bar.set_width(1)
        progress_text.set_text("~100%")

        progress_ax.set_title("Collecting Futures...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        print("Strain recalculation complete.")
        self.update_images()  #

        progress_ax.remove()
        progress_text.remove()
        progress_bar.remove()

    def update_images(self):
        """Updates the image displays based on the current frame index."""
        self.ax_img1.clear()
        self.ax_img2.clear()
        self.ax_processed.clear()
        self.ax_strain1.clear()
        self.ax_strain2.clear()
        self.ax_strain3.clear()
        self.ax_strain_combined.clear()
        self.ax_strain_cumulative.clear()
        self.ax_motion_sum.clear()  # Clear motion sum axis on image update
        self.ax_motion_sum_last5.clear()  # Clear last 5 motion sum axis on image update

        if self.num_images > 0:
            # Load and display original images
            img1 = cv2.imread(self.image_files[self.frame_idx], cv2.IMREAD_GRAYSCALE)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            self.ax_img1.imshow(img1, cmap="gray")
            self.ax_img1.set_title(f"Frame {self.frame_idx + 1}")

            next_idx = (self.frame_idx + self.frame_gap) % self.num_images
            img2 = cv2.imread(self.image_files[next_idx], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            self.ax_img2.imshow(img2, cmap="gray")
            self.ax_img2.set_title(f"Frame {next_idx + 1}")

            # Display processed displacement data
            if self.processed_motion_images and self.frame_idx < len(self.processed_motion_images):  # Check if processed data is available for current frame
                disp_data = self.processed_motion_images[self.frame_idx]
                disp_img = self.create_displacement_image(disp_data)
                self.ax_processed.imshow(disp_img)
                self.ax_processed.set_title("Processed Frame")

            # Display strain components - individual
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

        self.ax_img1.axis("off")
        self.ax_img2.axis("off")
        self.ax_processed.axis("off")
        self.ax_strain1.axis("off")
        self.ax_strain2.axis("off")
        self.ax_strain3.axis("off")
        self.ax_strain_combined.axis("off")  # Hide axis for combined strain - will use imshow
        self.ax_strain_cumulative.axis("off")  # Hide axis for cumulative strain
        self.ax_motion_sum.axis("off")  # Hide axis for motion sum plot
        self.ax_motion_sum_last5.axis("off")  # Hide axis for last 5 frame motion sum

        f_frame, avg_force, cumulative_force = update_red_line(self.ax_force, self.frame_idx)
        self.update_force_info(f_frame, avg_force, cumulative_force)  # Update force info on image update
        self.update_motion_sum()  # Update motion sum on image update
        self.update_motion_sum_last_5()  # Update last 5 motion sum
        self.update_combined_strain()  # Update combined strain map on image update
        self.update_cumulative_strain()  # Update cumulative strain map on image update

        plt.draw()

    def update_info_text(self):
        """Updates the top-right text box with image count and current frame info."""
        self.ax_info.clear()
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        self.ax_info.set_frame_on(False)

        # text = f"Loaded Data Info:\nImages: {self.num_images}\nCurrent Frames: {self.frame_idx + 1} & {(self.frame_idx + 2) % self.num_images}\nModel: {self.selected_model}"  # Added model info
        text = (
            f"Loaded Data Info:\n"
            f"Images: {self.num_images}\n"
            f"Current Frames: {self.frame_idx + 1} "
            f"& {(self.frame_idx + self.frame_gap) % self.num_images + 1}\n"
            f"Model: {self.selected_model}"
        )
        self.ax_info.text(0.1, 0.5, text, fontsize=10, fontweight="bold", va="center")

        plt.draw()

    def update_force_info(self, f_frame, avg_force, cumulative_force):
        """Updates the force information text box."""
        self.ax_force_info.clear()
        self.ax_force_info.set_xticks([])
        self.ax_force_info.set_yticks([])
        self.ax_force_info.set_frame_on(False)

        # Placeholder text - replace with actual force data
        avg_force_text = f"Avg Force: {avg_force:.2f}"
        current_force_text = f"Current Force: {f_frame:.2f}"
        total_force_text = "Total Force: {:.2f}".format(cumulative_force)

        text = f"Force Data:\n{avg_force_text}\n{current_force_text}\n{total_force_text}"
        self.ax_force_info.text(0.1, 0.5, text, fontsize=10, fontweight="bold", va="center")  # Vertically centered text
        plt.draw()

    def update_motion_sum(self):
        """Updates the motion sum display (cumulative)."""
        self.ax_motion_sum.clear()
        self.ax_motion_sum.set_xticks([])
        self.ax_motion_sum.set_yticks([])
        self.ax_motion_sum.set_frame_on(False)  # Hide axis

        if self.motion_sum_data and self.frame_idx < len(self.motion_sum_data):
            motion_sum_frame = self.motion_sum_data[self.frame_idx]
            motion_sum_img = self.create_displacement_image(motion_sum_frame)
            motion_sum_img = (motion_sum_img - motion_sum_img.min()) / (motion_sum_img.max() - motion_sum_img.min() + 1e-6)  # Normalize for display

            self.ax_motion_sum.imshow(motion_sum_img)
            self.ax_motion_sum.set_title("Motion Sum (Cumulative)")
        else:
            self.ax_motion_sum.text(0.5, 0.5, "No Motion Sum Data", ha="center", va="center", fontsize=10)  # Display text if no data

        plt.draw()

    def update_motion_sum_last_5(self):
        """Updates the motion sum display (last 5 frames)."""
        self.ax_motion_sum_last5.clear()
        self.ax_motion_sum_last5.set_xticks([])
        self.ax_motion_sum_last5.set_yticks([])
        self.ax_motion_sum_last5.set_frame_on(False)  # Hide axis

        if self.processed_motion_images and self.frame_idx < len(self.processed_motion_images):
            start_frame = max(0, self.frame_idx - 4)  # Start from 0 if frame_idx < 4
            last_5_motion = np.sum(self.processed_motion_images[start_frame : self.frame_idx + 1], axis=0)  # Sum last 5 or fewer frames
            motion_sum_last_5_img = self.create_displacement_image(last_5_motion)
            motion_sum_last_5_img = (motion_sum_last_5_img - motion_sum_last_5_img.min()) / (motion_sum_last_5_img.max() - motion_sum_last_5_img.min() + 1e-6)  # Normalize

            self.ax_motion_sum_last5.imshow(motion_sum_last_5_img)
            self.ax_motion_sum_last5.set_title("Motion Sum (Last 5 Frames)")
        else:
            self.ax_motion_sum_last5.text(0.5, 0.5, "No Motion Data", ha="center", va="center", fontsize=10)  # Display text if no data

        plt.draw()

    def update_combined_strain(self):
        """Updates the combined strain visualization."""
        self.ax_strain_combined.clear()
        self.ax_strain_combined.set_xticks([])
        self.ax_strain_combined.set_yticks([])
        self.ax_strain_combined.set_frame_on(False)  # Hide axis

        if self.strain_data and self.frame_idx < len(self.strain_data):
            strain_xx, strain_yy, strain_xy = self.strain_data[self.frame_idx]

            # Normalize each strain component individually
            norm_strain_xx = (strain_xx - strain_xx.min()) / (strain_xx.max() - strain_xx.min() + 1e-6)
            norm_strain_yy = (strain_yy - strain_yy.min()) / (strain_yy.max() - strain_yy.min() + 1e-6)
            norm_strain_xy = (strain_xy - strain_xy.min()) / (strain_xy.max() - strain_xy.min() + 1e-6)

            # Stack normalized strain components into RGB channels
            combined_strain_rgb = np.stack([norm_strain_xx, norm_strain_yy, norm_strain_xy], axis=-1)

            self.ax_strain_combined.imshow(combined_strain_rgb, origin="upper", cmap=strain_cmap)  # Use more colorful cmap
            self.ax_strain_combined.set_title("Combined Strain")
        else:
            self.ax_strain_combined.text(0.5, 0.5, "No Strain Data", ha="center", va="center", fontsize=10)  # Display text if no data

        plt.draw()

    def update_cumulative_strain(self):
        """Updates the cumulative strain visualization."""
        self.ax_strain_cumulative.clear()
        self.ax_strain_cumulative.set_xticks([])
        self.ax_strain_cumulative.set_yticks([])
        self.ax_strain_cumulative.set_frame_on(False)  # Hide axis

        if self.cumulative_strain_data and self.frame_idx < len(self.cumulative_strain_data):
            cumulative_strain_frame = self.cumulative_strain_data[self.frame_idx]

            # Normalize the cumulative strain for display (normalize each component separately for RGB)
            norm_cumulative_strain = np.zeros_like(cumulative_strain_frame)
            for i in range(cumulative_strain_frame.shape[-1]):  # Iterate over channels (strain components)
                min_val = cumulative_strain_frame[:, :, i].min()
                max_val = cumulative_strain_frame[:, :, i].max()
                norm_cumulative_strain[:, :, i] = (cumulative_strain_frame[:, :, i] - min_val) / (max_val - min_val + 1e-6)

            self.ax_strain_cumulative.imshow(norm_cumulative_strain, origin="upper", cmap=strain_cmap)
            self.ax_strain_cumulative.set_title("Cumulative Strain")
        else:
            self.ax_strain_cumulative.text(0.5, 0.5, "No Cumulative Strain Data", ha="center", va="center", fontsize=10)

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

        progress_ax = self.fig.add_axes((0.3, 0.48, 0.4, 0.03))
        progress_bar = Rectangle((0, 0), 0, 1, color="green")
        progress_ax.add_patch(progress_bar)
        progress_ax.set_axis_off()
        progress_ax.set_xlim(0, 1)
        progress_ax.set_ylim(0, 1)
        progress_text = self.fig.text(0.5, 0.5, "0%", ha="center", va="center", fontsize=12)
        progress_ax.set_title("Processing Images...")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        self.processed_motion_images = []
        self.processed_motion_data_paths = []
        self.strain_data = []
        self.motion_sum_data = []  # Initialize motion sum data
        self.cumulative_strain_data = []  # Initialize cumulative strain data
        os.makedirs(self.cache_folder, exist_ok=True)

        cumulative_motion = None  # Initialize cumulative motion to None, will be created on first frame
        cumulative_strain = None  # Initialize cumulative strain to None

        strain_futures = []

        last_update_time = time.time()
        update_interval = 1.0  # Update UI at most every 0.5 seconds
        total_images = len(self.image_files) - 1

        total_pairs = self.num_images - self.frame_gap

        for i in range(total_pairs):
            img_path = self.image_files[i]
            next_image_path = self.image_files[i + self.frame_gap]

            model_results = process_image(img_path, next_image_path, self.cache_folder, model_name=self.selected_model)  # Pass selected_model

            if model_results is None:
                self.btn_process.label.set_text("Model not found")
                self.btn_process.set_active(False)
                break

            disp_image, disp_data_path = model_results
            self.processed_motion_images.append(disp_image)
            self.processed_motion_data_paths.append(disp_data_path)
            disp_image = disp_image.astype(np.float32)  # Ensure float32 for accumulation

            # Accumulate motion for motion sum
            if cumulative_motion is None:  # Initialize on first frame if size is not predetermined
                cumulative_motion = disp_image.copy()
            else:
                cumulative_motion += disp_image
            self.motion_sum_data.append(cumulative_motion.copy())  # Store cumulative motion for each frame

            # Compute strain from displacement data
            strain_future = calculate_strain(self.strain_calc_path, disp_data_path, self.selected_model, self.subset_size)
            strain_futures.append(strain_future)

            current_time = time.time()
            if current_time - last_update_time > update_interval or i == total_images - 1:
                progress = (i + 1) / total_images
                progress_bar.set_width(progress)
                progress_text.set_text(f"{int(progress*100)}%")
                self.fig.canvas.draw_idle()
                plt.pause(0.01)
                last_update_time = current_time

        progress_bar.set_width(1)
        progress_text.set_text("~100%")

        progress_ax.set_title("Collecting Futures...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        # Process results as they complete
        for future in strain_futures:
            strain_result = future.result()
            self.strain_data.append(strain_result)

            # Accumulate strain
            if strain_result is not None:  # Check if strain_result is valid
                strain_xx, strain_yy, strain_xy = strain_result
                current_strain = np.stack([strain_xx, strain_yy, strain_xy], axis=-1).astype(np.float32)  # Stack and ensure float32

                if cumulative_strain is None:
                    cumulative_strain = current_strain.copy()
                else:
                    cumulative_strain += current_strain
                self.cumulative_strain_data.append(cumulative_strain.copy())  # Store cumulative strain

        print(f"Processed {len(self.processed_motion_images)} images using model: {self.selected_model}")

        progress_bar.remove()
        progress_text.remove()
        progress_ax.remove()
        self.btn_process.label.set_text("Run ML Model")
        self.btn_process.set_active(True)  # Re-enable button

        self.update_images()

    def create_displacement_image(self, disp_data):
        return disp_data

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

        progress_ax = self.fig.add_axes((0.3, 0.48, 0.4, 0.03))
        progress_bar = Rectangle((0, 0), 0, 1, color="green")
        progress_ax.add_patch(progress_bar)
        progress_ax.set_axis_off()
        progress_ax.set_xlim(0, 1)
        progress_ax.set_ylim(0, 1)
        progress_text = self.fig.text(0.5, 0.5, "0%", ha="center", va="center", fontsize=12)
        progress_ax.set_title("Exporting Frames GIF...")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        last_update_time = time.time()
        update_interval = 1.0  # Update UI at most every 0.5 seconds

        output_filename = "frames.gif"  # Changed filename to frames.gif
        print(f"Exporting frames GIF to {output_filename}...")
        motion_frames = []
        for i in range(0, len(self.image_files) - 1):
            img1 = cv2.imread(self.image_files[i])
            img2 = cv2.imread(self.image_files[i + 1])
            combined_img = np.concatenate((img1, img2), axis=1)  # Combine side-by-side
            motion_frames.append(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))  # Convert to RGB for imageio

            current_time = time.time()
            if current_time - last_update_time > update_interval:
                progress = (i + 1) / len(self.image_files)
                progress_bar.set_width(progress)
                progress_text.set_text(f"{int(progress*100)}%")
                self.fig.canvas.draw_idle()
                plt.pause(0.01)
                last_update_time = current_time

        progress_bar.set_width(1)
        progress_text.set_text("~100%")
        progress_ax.set_title("Saving File...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        imageio.mimsave(output_filename, motion_frames, fps=10)  # Adjust fps as needed
        print(f"Frames GIF exported to {output_filename}")

        self.btn_export_frames.label.set_text("Frames GIF")
        self.btn_export_frames.set_active(True)

        progress_bar.remove()
        progress_text.remove()
        progress_ax.remove()
        plt.pause(0.01)

    def export_motion_gif(self, event=None):
        """Exports a GIF of the processed motion (displacement) data."""
        if not self.processed_motion_images:
            print("No processed motion data available to export. Run ML model first.")
            self.btn_export_motion.label.set_text("No Motion Data (Click Run ML Model)")
            return

        self.btn_export_motion.label.set_text("Exporting...")
        self.btn_export_motion.set_active(False)

        progress_ax = self.fig.add_axes((0.3, 0.48, 0.4, 0.03))
        progress_bar = Rectangle((0, 0), 0, 1, color="green")
        progress_ax.add_patch(progress_bar)
        progress_ax.set_axis_off()
        progress_ax.set_xlim(0, 1)
        progress_ax.set_ylim(0, 1)
        progress_text = self.fig.text(0.5, 0.5, "0%", ha="center", va="center", fontsize=12)
        progress_ax.set_title("Exporting Motion GIF...")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        last_update_time = time.time()
        update_interval = 1.0  # Update UI at most every 0.5 seconds

        output_filename = "motion.gif"
        print(f"Exporting motion GIF to {output_filename}...")
        motion_frames = []
        for i, disp_data in enumerate(self.processed_motion_images):
            disp_img_rgb = self.create_displacement_image(disp_data)  # Get RGB displacement image
            motion_frames.append(np.uint8(disp_img_rgb * 255))  # Scale to 0-255 and convert to uint8

            current_time = time.time()
            if current_time - last_update_time > update_interval:
                progress = (i + 1) / len(self.processed_motion_images)
                progress_bar.set_width(progress)
                progress_text.set_text(f"{int(progress*100)}%")
                self.fig.canvas.draw_idle()
                plt.pause(0.01)
                last_update_time = current_time

        progress_bar.set_width(1)
        progress_text.set_text("~100%")
        progress_ax.set_title("Saving File...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        imageio.mimsave(output_filename, motion_frames, fps=10)  # Adjust fps as needed
        print(f"Motion GIF exported to {output_filename}")

        self.btn_export_motion.label.set_text("Motion GIF")
        self.btn_export_motion.set_active(True)

        progress_bar.remove()
        progress_text.remove()
        progress_ax.remove()
        plt.pause(0.01)

    def export_cumulative_motion_gif(self, event=None):
        """Exports a GIF of the cumulative motion data."""
        if not self.motion_sum_data:
            print("No cumulative motion data available to export. Run ML model first.")
            self.btn_export_cumulative_motion.label.set_text("No Cumulative Motion Data (Click Run ML Model)")
            return

        self.btn_export_cumulative_motion.label.set_text("Exporting...")
        self.btn_export_cumulative_motion.set_active(False)

        progress_ax = self.fig.add_axes((0.3, 0.48, 0.4, 0.03))
        progress_bar = Rectangle((0, 0), 0, 1, color="green")
        progress_ax.add_patch(progress_bar)
        progress_ax.set_axis_off()
        progress_ax.set_xlim(0, 1)
        progress_ax.set_ylim(0, 1)
        progress_text = self.fig.text(0.5, 0.5, "0%", ha="center", va="center", fontsize=12)
        progress_ax.set_title("Exporting Cumulative Motion GIF...")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        last_update_time = time.time()
        update_interval = 1.0  # Update UI at most every 0.5 seconds

        output_filename = "cumulative_motion.gif"
        print(f"Exporting cumulative motion GIF to {output_filename}...")
        motion_frames = []
        for i, disp_data in enumerate(self.motion_sum_data):
            disp_img_rgb = self.create_displacement_image(disp_data)  # Get RGB displacement image
            disp_img_rgb = (disp_img_rgb - disp_img_rgb.min()) / (disp_img_rgb.max() - disp_img_rgb.min() + 1e-6)  # Normalize for display
            motion_frames.append(np.uint8(disp_img_rgb * 255))  # Scale to 0-255 and convert to uint8

            current_time = time.time()
            if current_time - last_update_time > update_interval:
                progress = (i + 1) / len(self.motion_sum_data)
                progress_bar.set_width(progress)
                progress_text.set_text(f"{int(progress*100)}%")
                self.fig.canvas.draw_idle()
                plt.pause(0.01)
                last_update_time = current_time

        progress_bar.set_width(1)
        progress_text.set_text("~100%")
        progress_ax.set_title("Saving File...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        imageio.mimsave(output_filename, motion_frames, fps=10)  # Adjust fps as needed
        print(f"Cumulative motion GIF exported to {output_filename}")

        self.btn_export_cumulative_motion.label.set_text("Cumulative Motion GIF")
        self.btn_export_cumulative_motion.set_active(True)

        progress_bar.remove()
        progress_text.remove()
        progress_ax.remove()
        plt.pause(0.01)

    def export_last5_motion_gif(self, event=None):
        """Exports a GIF of the last 5 frames of motion data."""
        if not self.processed_motion_images or len(self.processed_motion_images) < 5:
            print("No motion data available to export. Run ML model first.")
            self.btn_export_last5_motion.label.set_text("No Motion Data (Click Run ML Model)")
            return

        self.btn_export_last5_motion.label.set_text("Exporting...")
        self.btn_export_last5_motion.set_active(False)

        progress_ax = self.fig.add_axes((0.3, 0.48, 0.4, 0.03))
        progress_bar = Rectangle((0, 0), 0, 1, color="green")
        progress_ax.add_patch(progress_bar)
        progress_ax.set_axis_off()
        progress_ax.set_xlim(0, 1)
        progress_ax.set_ylim(0, 1)
        progress_text = self.fig.text(0.5, 0.5, "0%", ha="center", va="center", fontsize=12)
        progress_ax.set_title("Exporting Last 5 Motion GIF...")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        last_update_time = time.time()
        update_interval = 1.0  # Update UI at most every 0.5 seconds

        output_filename = "last5_motion.gif"
        print(f"Exporting last 5 motion GIF to {output_filename}...")
        motion_frames = []
        for i in range(0, len(self.processed_motion_images) - 4):
            last_5_motion = np.sum(self.processed_motion_images[i : i + 5], axis=0)  # Sum last 5 or fewer frames
            motion_sum_last_5_img = self.create_displacement_image(last_5_motion)
            motion_sum_last_5_img = (motion_sum_last_5_img - motion_sum_last_5_img.min()) / (motion_sum_last_5_img.max() - motion_sum_last_5_img.min() + 1e-6)  # Normalize
            motion_frames.append(np.uint8(motion_sum_last_5_img * 255))  # Scale to 0-255 and convert to uint8

            current_time = time.time()
            if current_time - last_update_time > update_interval:
                progress = (i + 1) / len(self.processed_motion_images)
                progress_bar.set_width(progress)
                progress_text.set_text(f"{int(progress*100)}%")
                self.fig.canvas.draw_idle()
                plt.pause(0.01)
                last_update_time = current_time

        progress_bar.set_width(1)
        progress_text.set_text("~100%")
        progress_ax.set_title("Saving File...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        imageio.mimsave(output_filename, motion_frames, fps=10)  # Adjust fps as needed
        print(f"Last 5 motion GIF exported to {output_filename}")

        self.btn_export_last5_motion.label.set_text("Last 5 Motion GIF")
        self.btn_export_last5_motion.set_active(True)

        progress_bar.remove()
        progress_text.remove()
        progress_ax.remove()
        plt.pause(0.01)

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

        np_strain_data = np.array(self.strain_data)  # Convert list of tuples to a NumPy array
        norm_strain_data = (np_strain_data - np_strain_data.min()) / (np_strain_data.max() - np_strain_data.min() + 1e-6)
        image_data = norm_strain_data * 255
        image_data = image_data.astype(np.uint8)
        image_data = image_data.transpose(0, 2, 3, 1)  # Transpose to (frames, height, width, channels)

        # Save GIF with a reasonable FPS
        imageio.mimsave(output_filename, image_data, fps=10)  # Adjust fps as needed
        print(f"Strain GIF exported to {output_filename}")

        self.btn_export_strain.label.set_text("Strain GIF")
        self.btn_export_strain.set_active(True)

    def export_cumulative_strain_gif(self, event=None):
        """Exports a GIF of the cumulative strain data."""
        if not self.strain_data:
            print("No cumulative strain data available to export. Run ML model first.")
            self.btn_export_cumulative_strain.label.set_text("No Cumulative Strain Data (Click Run ML Model)")
            return

        self.btn_export_cumulative_strain.label.set_text("Exporting...")
        self.btn_export_cumulative_strain.set_active(False)

        progress_ax = self.fig.add_axes((0.3, 0.48, 0.4, 0.03))
        progress_bar = Rectangle((0, 0), 0, 1, color="green")
        progress_ax.add_patch(progress_bar)
        progress_ax.set_axis_off()
        progress_ax.set_xlim(0, 1)
        progress_ax.set_ylim(0, 1)
        progress_text = self.fig.text(0.5, 0.5, "0%", ha="center", va="center", fontsize=12)
        progress_ax.set_title("Exporting Cumulative Strain GIF...")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        last_update_time = time.time()
        update_interval = 1.0  # Update UI at most every 0.5 seconds

        output_filename = "cumulative_strain.gif"
        print(f"Exporting cumulative strain GIF to {output_filename}...")

        strain_frames = []

        np_strain_data = np.array(self.strain_data)  # Convert list of tuples to a NumPy array

        # Loop through frames
        for frame_idx in range(len(self.strain_data)):
            sum_strain_data = np.sum(np_strain_data[0 : frame_idx + 1], axis=0)
            sum_strain_data = (sum_strain_data - sum_strain_data.min()) / (sum_strain_data.max() - sum_strain_data.min() + 1e-6)
            sum_strain_data = sum_strain_data * 255
            sum_strain_data = sum_strain_data.astype(np.uint8)
            sum_strain_data = sum_strain_data.transpose(1, 2, 0)  # Transpose to (height, width, channels)
            strain_frames.append(sum_strain_data)

            current_time = time.time()
            if current_time - last_update_time > update_interval:
                progress = (frame_idx + 1) / len(self.strain_data)
                progress_bar.set_width(progress)
                progress_text.set_text(f"{int(progress*100)}%")
                self.fig.canvas.draw_idle()
                plt.pause(0.01)
                last_update_time = current_time

        progress_bar.set_width(1)
        progress_text.set_text("~100%")
        progress_ax.set_title("Saving File...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        # Save GIF with a reasonable FPS
        imageio.mimsave(output_filename, strain_frames, fps=10)  # Adjust fps as needed
        print(f"Cumulative strain GIF exported to {output_filename}")

        self.btn_export_cumulative_strain.label.set_text("Cumulative Strain GIF")
        self.btn_export_cumulative_strain.set_active(True)

        progress_bar.remove()
        progress_text.remove()
        progress_ax.remove()
        plt.pause(0.01)

    def export_last5_strain_gif(self, event=None):
        """Exports a GIF of the last 5 frames of strain data."""
        if not self.strain_data or len(self.strain_data) < 5:
            print("No strain data available to export. Run ML model first.")
            self.btn_export_last5_strain.label.set_text("No Strain Data (Click Run ML Model)")
            return

        self.btn_export_last5_strain.label.set_text("Exporting...")
        self.btn_export_last5_strain.set_active(False)

        progress_ax = self.fig.add_axes((0.3, 0.48, 0.4, 0.03))
        progress_bar = Rectangle((0, 0), 0, 1, color="green")
        progress_ax.add_patch(progress_bar)
        progress_ax.set_axis_off()
        progress_ax.set_xlim(0, 1)
        progress_ax.set_ylim(0, 1)
        progress_text = self.fig.text(0.5, 0.5, "0%", ha="center", va="center", fontsize=12)
        progress_ax.set_title("Exporting Last 5 Strain GIF...")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        last_update_time = time.time()
        update_interval = 1.0  # Update UI at most every 0.5 seconds

        output_filename = "last5_strain.gif"
        print(f"Exporting last 5 strain GIF to {output_filename}...")

        strain_frames = []

        np_strain_data = np.array(self.strain_data)  # Convert list of tuples to a NumPy array

        # Loop through frames
        for frame_idx in range(len(self.strain_data) - 4):
            sum_strain_data = np.sum(np_strain_data[frame_idx : frame_idx + 5], axis=0)
            sum_strain_data = (sum_strain_data - sum_strain_data.min()) / (sum_strain_data.max() - sum_strain_data.min() + 1e-6)
            sum_strain_data = sum_strain_data * 255
            sum_strain_data = sum_strain_data.astype(np.uint8)
            sum_strain_data = sum_strain_data.transpose(1, 2, 0)  # Transpose to (height, width, channels)
            strain_frames.append(sum_strain_data)

            current_time = time.time()
            if current_time - last_update_time > update_interval:
                progress = (frame_idx + 1) / len(self.strain_data)
                progress_bar.set_width(progress)
                progress_text.set_text(f"{int(progress*100)}%")
                self.fig.canvas.draw_idle()
                plt.pause(0.01)
                last_update_time = current_time

        progress_bar.set_width(1)
        progress_text.set_text("~100%")
        progress_ax.set_title("Saving File...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        # Save GIF with a reasonable FPS
        imageio.mimsave(output_filename, strain_frames, fps=10)  # Adjust fps as needed
        print(f"Last 5 strain GIF exported to {output_filename}")

        self.btn_export_last5_strain.label.set_text("Last 5 Strain GIF")
        self.btn_export_last5_strain.set_active(True)

        progress_bar.remove()
        progress_text.remove()
        progress_ax.remove()
        plt.pause(0.01)


# Example usage:
if __name__ == "__main__":
    strain_calc_path = "./strain_calc.exe" if platform.system() == "Windows" else "./strain_calc"

    parser = argparse.ArgumentParser()
    parser.add_argument("--force_data_path", type=str, default="./force_data.csv")
    parser.add_argument("--image_folder", type=str, default="./images/")
    parser.add_argument("--cache_folder", type=str, default="./cached_processed_images/")
    parser.add_argument("--strain_calc_path", type=str, default=strain_calc_path)
    args = parser.parse_args()

    force_data_path = args.force_data_path
    image_folder = args.image_folder
    cache_folder = args.cache_folder
    strain_calc_path = args.strain_calc_path

    # Check for strain calculation executable
    if strain_calc_path != strain_calc_path and not os.path.exists(strain_calc_path):
        print("Error: strain_calc.runme executable not found. Please specify a correct path to the executable or leave it as the default.")
        exit(1)

    if not os.path.exists(strain_calc_path):
        print(f"Error: {strain_calc_path} executable not found. Downloading from GitHub...")
        if platform.system() == "Windows":
            donwload_url = "https://github.com/AidanSchmitigal/displaceToStrain/releases/download/v0.4-beta/build-artifacts-windows-latest-Release-cl.zip"
        elif platform.system() == "Linux":
            donwload_url = "https://github.com/AidanSchmitigal/displaceToStrain/releases/download/v0.4-beta/build-artifacts-ubuntu-latest-Release-clang.zip"
        elif platform.system() == "Darwin":
            donwload_url = "https://github.com/AidanSchmitigal/displaceToStrain/releases/download/v0.4-beta/build-artifacts-macos-latest-Release-clang.zip"
        else:
            print("Error: unsupported platform. Please manually download the executable from https://github.com/AidanSchmitigal/displaceToStrain/releases")
            exit(1)

        dload.save_unzip(donwload_url, os.path.dirname("./"), True)

        if platform.system() == "Linux" or platform.system() == "Darwin":
            os.chmod(strain_calc_path, 0o755)

        print("Download complete.")

    img_grid = ImageGrid(force_data_path, image_folder, cache_folder, strain_calc_path)  # Pass initial_model to ImageGrid
    plt.show()

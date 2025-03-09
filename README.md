# Image Processing and Strain Analysis Project

## Overview

This project provides a graphical interface for visualizing force data, image sequences, machine learning-processed images, and strain analysis. The system loads a sequence of images, processes them through a machine learning model, and visualizes the resulting displacement and strain data in a structured grid layout.

## Features

- Displays force vs. point data with a read out of the current force for the current frame.
- Loads and displays sequential images from a specified folder.
- Processes images through a machine learning model and caches results.
- Computes and visualizes strain data.
- Includes navigation controls (previous, next, play/pause) to move through image frames.
- Displays metadata about the loaded dataset, including the number of images and current frame indices.

## Folder Structure

To set up the project properly, organize your files and folders as follows:

```
project_directory/
│── raw_images/               # Folder containing raw input images
│   ├── image_001.png
│   ├── image_002.png
│   ├── ...
│
│── cached_processed_images/         # Cached processed images (generated automatically)
│                                    # It also contains the model outputs as images and npy files
│                                    # and holds the strain data for each image
│
│── main.py                   # Main script for running the visualization
```

## Setup Instructions

### 1. Install Dependencies

Install PyTorch from [PyTorch.org](https://pytorch.org/).
Ensure you have Python installed along with the necessary libraries:

```sh
pip install matplotlib numpy pandas opencv-python imageio
```

### 2. Organize Images

Place your sequentially numbered images inside the `raw_images/` folder. Its best to have the images named consistently (e.g., `image_001.png`, `image_002.png`, etc.).

### 3. Set File Paths

The default file paths for the image and the force data are these:

```python
file_path = "data/force_data.csv"
image_folder = "data/raw_images"
```

If you want to use other ones, you can pass them as arguments to the main script:

```python
python main.py --file_path "data/force_data.csv" --image_folder "data/raw_images"
```

### 4. Run the Program

Execute the main script to launch the visualization interface:

```sh
python main.py
```

## Usage

- **Left/Right Arrows**: Navigate through frames.
- **Play/Pause Button**: Automatically advance frames.
- **Run Processing**: Ensure cached results are available before analysis.

## Notes

- The `cached_processed_images/` folder will store machine learning-processed images to speed up subsequent runs.
- If new images are added, ensure they are named correctly and clear old cached results if necessary.

## Troubleshooting

- If images do not load, check that the `raw_images/` folder exists and contains valid images.
- If the force plot does not display correctly, verify the CSV file format for force vs. point data. CSV files should have the following format:

```
"Point","Elongation","Force","Position","Code","Samplerate","Motorspeed"
.1,0,3.7384,-1.359,2,100,"0.1 mm/min"
.2,0,3.7228,-1.359,0,100,"0.1 mm/min"
.3,0,3.7197,-1.359,0,100,"0.1 mm/min"
.4,0,3.7852,-1.359,0,100,"0.1 mm/min"
...
```

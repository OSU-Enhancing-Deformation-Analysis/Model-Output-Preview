import os
import subprocess
import numpy as np
import pandas as pd

SUBSET_SIZE = 10

# Run the C++ executable
executable_path = "./strain_calc.runme"


import matplotlib.pyplot as plt
import numpy as np


# Function to read the txt file and extract strain data (strainXX, strainYY, strainXY)
def read_strain(file_path):
    strains = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.split()
            x = int(parts[0])
            y = int(parts[1])
            strain_xx = float(parts[2])
            strain_yy = float(parts[3])
            strain_xy = float(parts[4])
            strains.append((x, y, strain_xx, strain_yy, strain_xy))
    return strains


def calculate_strain(disp_data_path):
    strain_result_file = "cached_processed_images/" + os.path.basename(disp_data_path).split(".")[0] + "_strain_result.txt"

    if not os.path.exists(strain_result_file):
        args = [executable_path, disp_data_path, str(SUBSET_SIZE)]
        subprocess.run(args, check=True)

    # Read the strain data from file
    strains = read_strain(strain_result_file)  # Assuming read_strain() returns a list of lists

    # Extract x, y, and strain components
    x_coords = np.array([strain[0] for strain in strains])
    y_coords = np.array([strain[1] for strain in strains])
    strain_xx = np.array([strain[2] for strain in strains])
    strain_yy = np.array([strain[3] for strain in strains])
    strain_xy = np.array([strain[4] for strain in strains])

    # Determine grid size (assuming structured grid)
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    grid_width = len(x_unique)
    grid_height = len(y_unique)

    # Reshape strain data into 2D arrays
    strain_xx_2d = strain_xx.reshape((grid_height, grid_width)).T
    strain_yy_2d = strain_yy.reshape((grid_height, grid_width)).T
    strain_xy_2d = strain_xy.reshape((grid_height, grid_width)).T

    # strain_xx_2d = np.flipud(strain_xx_2d)  # Flip the strain data vertically
    # strain_yy_2d = np.flipud(strain_yy_2d)
    # strain_xy_2d = np.flipud(strain_xy_2d)

    return strain_xx_2d, strain_yy_2d, strain_xy_2d

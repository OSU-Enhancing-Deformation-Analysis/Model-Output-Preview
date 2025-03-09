import os
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Global thread pool that can be reused
executor = ThreadPoolExecutor(max_workers=os.cpu_count())


def calculate_strain_worker(args):
    executable_path, disp_data_path, model_name, subset_size, force_rerun = args
    strain_result_file = "cached_processed_images/" + model_name + "/" + os.path.basename(disp_data_path).split(".")[0] + "_strain_result.npy"

    executable_path = os.path.abspath(executable_path)

    if not os.path.exists(strain_result_file) or force_rerun:
        cmd_args = [executable_path, disp_data_path, str(subset_size), "--numpy", "--output_dir", os.path.dirname(strain_result_file)]
        subprocess.run(cmd_args, check=True)

    # Read the strain data from file
    strains = np.load(strain_result_file)

    strain_xx = strains[:, :, 0]
    strain_yy = strains[:, :, 1]
    strain_xy = strains[:, :, 2]

    return strain_xx, strain_yy, strain_xy


def calculate_strain(executable_path, disp_data_path, model_name, subset_size, force_rerun=False):
    # Submit the task to the thread pool
    future = executor.submit(calculate_strain_worker, (executable_path, disp_data_path, model_name, subset_size, force_rerun))
    # Return the future directly - this allows immediate submission without waiting
    return future

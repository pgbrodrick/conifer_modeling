

import argparse
import pandas as pd
import numpy as np
import subprocess
import os


def main():
    parser = argparse.ArgumentParser(description='Run conifer predictions on a set of flightlines.')
    parser.add_argument('radiance_file_list', type=str)
    parser.add_argument('reflectance_file_list', type=str)
    args = parser.parse_args()

    OUTPUT_DIR = 'predicted_conifer'
    if (os.path.isdir(OUTPUT_DIR) == False):
        os.mkdir(OUTPUT_DIR)

    rad_files = np.squeeze(np.array(pd.read_csv(args.radiance_file_list, header=None))).tolist()
    refl_files = [os.path.join(args.reflectance_file_list, os.path.basename(
        x).replace('rdn', 'acorn_autovis_refl_ciacorn')) for x in rad_files]
    con_files = [os.path.join(OUTPUT_DIR, os.path.basename(x).replace(
        'rdn', 'conifer_prediction.tif')) for x in rad_files]

    for _i in range(0, len(rad_files)):
        cmd_str = 'sbatch -n 1 --mem=30000 -p DGE -o logs/o -e logs/e --wrap="export MKL_NUM_THREADS=1; python apply_conifer_nn.py {} {}"'.format(
            refl_files[_i], con_files[_i])
        subprocess.call(cmd_str, shell=True)
        print(cmd_str)


if __name__ == "__main__":
    main()

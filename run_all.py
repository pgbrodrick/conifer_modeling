





import pandas as pd
import numpy as np
import subprocess
import os,sys




OUTPUT_DIR = 'predicted_conifer'
if (os.path.isdir(OUTPUT_DIR) == False):
  os.mkdir(OUTPUT_DIR)

rad_files = np.squeeze(np.array(pd.read_csv('../acorn_atmospheric_correction_2019/all_radiance_files.txt',header=None))).tolist()
refl_files = [os.path.join('../acorn_atmospheric_correction_2019/acorn_mid_variable_vis',os.path.basename(x).replace('rdn','acorn_autovis_refl_ciacorn')) for x in rad_files]
con_files = [os.path.join(OUTPUT_DIR,os.path.basename(x).replace('rdn','conifer_prediction.tif')) for x in rad_files]

for _i in range(0,len(rad_files)):
  cmd_str = 'sbatch -n 1 --mem=30000 -p DGE -o logs/o -e logs/e --wrap="export MKL_NUM_THREADS=1; python apply_conifer_nn.py {} {}"'.format(refl_files[_i],con_files[_i])
  subprocess.call(cmd_str,shell=True)
  print(cmd_str)



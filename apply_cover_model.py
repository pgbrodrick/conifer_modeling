import numpy as np
import gdal
import os
import argparse
import keras
from tqdm import tqdm
#from sklearn.externals import joblib
import joblib
#check order of bn & standarization

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Apply chem equation to BIL')
parser.add_argument('refl_dat_f')
parser.add_argument('output_name')
parser.add_argument('-model_f', default='output/trained_models/cover_modeldata_splits_12_nl_2_dr_0.4_nn_250_it_0.h5')
parser.add_argument('-scaler', default=None)
parser.add_argument('-bn', default=1, type=int, choices=[0, 1])
parser.add_argument('-argmax', default=1, type=int, choices=[0, 1])
args = parser.parse_args()

args.bn = args.bn == 1
args.argmax = args.argmax == 1

if os.path.isfile(args.output_name):
    print('output file: {} found.  terminating'.format(args.output_name))
    quit()

# make sure these match the settings file corresponding to the coefficient file

# open up chemical equation data
model = keras.models.load_model(args.model_f)

# open up raster sets
dataset = gdal.Open(args.refl_dat_f, gdal.GA_ReadOnly)
data_trans = dataset.GetGeoTransform()

max_y = dataset.RasterYSize
max_x = dataset.RasterXSize


# create blank output file
driver = gdal.GetDriverByName('GTiff')
driver.Register()


full_bad_bands = np.zeros(426).astype(bool)
full_bad_bands[:8] = True
full_bad_bands[192:205] = True
full_bad_bands[284:327] = True
full_bad_bands[417:] = True

bad_bands = np.zeros(142).astype(bool)
bad_bands[:2] = True
bad_bands[64:68] = True
bad_bands[95:109] = True
bad_bands[139:] = True


if args.scaler is not None:
    scaler = joblib.load(args.scaler)

n_bands = 1
if args.argmax is False:
    dat_bands = np.squeeze(dataset.ReadAsArray(0, 0, 5, 1)).astype(np.float32)
    dat_bands = dat_bands[np.logical_not(full_bad_bands), ...]
    dat_bands = np.transpose(dat_bands)
    pred_shape = model.predict(dat_bands)
    n_bands = len(pred_shape[-1])

output_predictions = np.zeros((max_y, max_x, n_bands))
outDataset = driver.Create(args.output_name,
                           max_x,
                           max_y,
                           n_bands,
                           gdal.GDT_Float32,
                           options=['COMPRESS=DEFLATE', 'TILED=YES'])

outDataset.SetProjection(dataset.GetProjection())
outDataset.SetGeoTransform(dataset.GetGeoTransform())

# loop through lines [y]
for l in tqdm(range(0, max_y), ncols=80):
    dat = np.squeeze(dataset.ReadAsArray(0, l, max_x, 1)).astype(np.float32)
    dat = dat[np.logical_not(full_bad_bands), ...]
    dat = np.transpose(dat)
    if np.nansum(dat) > 0:
        if args.bn:
            dat = dat / np.sqrt(np.nanmean(np.power(dat, 2), axis=1))[:, np.newaxis]

        if args.scaler is not None:
            dat = scaler.transform(dat)

        if args.argmax is True:
            output_predictions[l, :, 0] = np.argmax(model.predict(dat), axis=-1)
        else:
            output_predictions[l, ...] = model.predict(dat)

for _band in range(n_bands):
    outDataset.GetRasterBand(_band + 1).WriteArray(output_predictions[...,_band], 0, 0)
del outDataset

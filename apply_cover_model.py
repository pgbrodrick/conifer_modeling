import numpy as np
import gdal
import os
import argparse
import keras
from tqdm import tqdm
from sklearn.externals import joblib


import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Apply chem equation to BIL')
parser.add_argument('refl_dat_f')
parser.add_argument('output_name')
parser.add_argument('-model_f', default='trained_models/conifer_nn_set_29.h5')
parser.add_argument('-scaler', default='trained_models/nn_conifer_scaler')
parser.add_argument('-bn', default=True, type=bool)
args = parser.parse_args()

if (os.path.isfile(args.output_name)):
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


outDataset = driver.Create(args.output_name,
                           max_x,
                           max_y,
                           1,
                           gdal.GDT_Float32,
                           options=['COMPRESS=DEFLATE', 'TILED=YES'])

outDataset.SetProjection(dataset.GetProjection())
outDataset.SetGeoTransform(dataset.GetGeoTransform())

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
output_predictions = np.zeros((max_y, max_x))


if args.scaler is not None:
    scaler = joblib.load(args.scaler)

# loop through lines [y]
for l in tqdm(range(0, max_y), ncols=80):
    dat = np.squeeze(dataset.ReadAsArray(0, l, max_x, 1)).astype(np.float32)
    dat = dat[np.logical_not(full_bad_bands), ...]
    dat = np.transpose(dat)
    if (np.nansum(dat) > 0):
        if (args.bn):
            dat = dat / np.sqrt(np.nanmean(np.power(dat, 2), axis=1))[:, np.newaxis]

        dat = scaler.transform(dat)
        output_predictions[l, :] = model.predict(dat)[:, 0]

outDataset.GetRasterBand(1).WriteArray(output_predictions, 0, 0)
del outDataset

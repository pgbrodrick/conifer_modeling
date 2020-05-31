

import gdal
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import os
import subprocess
import glob


def main():
    parser = argparse.ArgumentParser(
        description='efficiently extract data from a vector file and multiple accompanying rasters')

    parser.add_argument('out_base', type=str)
    parser.add_argument('-all_shape_dir',type=str)
    parser.add_argument('-shape_dirs', nargs='+', type=str)
    parser.add_argument('-shp_attribute', type=str, default='id')
    parser.add_argument('-max_samples_per_class', type=int, default=20000)
    parser.add_argument('-source_files', nargs='+', type=str)
    args = parser.parse_args()

    if args.all_shape_dir is not None:
        args.shape_dirs = glob.glob(os.path.join(args.all_shape_dir,'*'))

    if args.max_samples_per_class == -1:
        args.max_samples_per_class=1e15

    # Open / check all raster files.  Check is very cursory.
    file_sets = [gdal.Open(fi, gdal.GA_ReadOnly) for fi in args.source_files]
    n_features = 0
    for _f in range(len(file_sets)):
        assert file_sets[_f] is not None, 'Invalid input file'
        if (file_sets[_f].RasterXSize != file_sets[0].RasterXSize):
            print('Raster X Size does not match, terminiating')
            quit()
        if (file_sets[_f].RasterYSize != file_sets[0].RasterYSize):
            print('Raster Y Size does not match, terminiating')
            quit()
        n_features += file_sets[_f].RasterCount

    trans = file_sets[0].GetGeoTransform()
    namelist = []
    init=' -te {} {} {} {} -tr {} {} -init -1 '.format(
                          trans[0],
                          trans[3]+trans[5]*file_sets[0].RasterYSize,
                          trans[0]+trans[1]*file_sets[0].RasterXSize,
                          trans[3],
                          trans[1],
                          trans[5])

    cover_raster_file = os.path.join(args.out_base,'cover_raster.tif')
    if (os.path.isfile(cover_raster_file)):
        print('cover raster file already exists at {}, using'.format(cover_raster_file))
        for index, shape_dir in enumerate(args.shape_dirs):
            shape_files = glob.glob(shape_dir + '/*.geojson')
            dirnames = [x for x in shape_dir.split('/') if x != '']
            namelist.append(dirnames[-1])
    else:
        for index, shape_dir in enumerate(args.shape_dirs):

            shape_files = glob.glob(shape_dir + '/*.geojson')
            dirnames = [x for x in shape_dir.split('/') if x != '']
            namelist.append(dirnames[-1])
            
            for shpfile in shape_files:
                cmd_str = 'gdal_rasterize {} {} -burn {} {}'.format(
                          shpfile,
                          cover_raster_file,
                          index,
                          init,
                          )
                print(cmd_str)
                subprocess.call(cmd_str, shell=True)
                init=''

    # Open binary cover file
    cover_set = gdal.Open(cover_raster_file, gdal.GA_ReadOnly)
    cover_trans = cover_set.GetGeoTransform()
    assert cover_set is not None, 'Invalid input file'

    # Get cover coordinates
    covers = cover_set.ReadAsArray()
    un_covers = np.unique(covers[covers != -1]).astype(int)

    coord_lists = []
    num_outputs = 0
    np.random.seed(13)
    for cover in un_covers:

        cover_coords = list(np.where(covers == cover))
        if len(cover_coords[0]) > args.max_samples_per_class:
            perm = np.random.permutation(len(cover_coords[0]))[:args.max_samples_per_class]
            cover_coords[0] = cover_coords[0][perm]
            cover_coords[1] = cover_coords[1][perm]

        coord_lists.append(cover_coords)
        num_outputs += len(cover_coords[0])


    # Read through files and grab relevant data
    output_array = np.zeros((num_outputs, n_features + 3))
    output_names = []

    start_index = 0
    for cover in un_covers:

        cover_coords = coord_lists[cover]
        for _line in tqdm(range(len(cover_coords[0])), ncols=80):

            output_array[start_index + _line, 0] = covers[cover_coords[0][_line], cover_coords[1][_line]]
            output_array[start_index + _line, 1] = cover_coords[1][_line]*cover_trans[1]+cover_trans[0]
            output_array[start_index + _line, 2] = cover_coords[0][_line]*cover_trans[5]+cover_trans[3]

            output_names.append(namelist[cover])

            feat_ind = 3
            for _f in range(len(file_sets)):
                line = file_sets[_f].ReadAsArray(
                    0, int(cover_coords[0][_line]), file_sets[_f].RasterXSize, 1)
                if (len(line.shape) == 2):
                    line = np.reshape(line, (1, line.shape[0], line.shape[1]))

                line = np.squeeze(line[..., cover_coords[1][_line]])

                output_array[start_index + _line, feat_ind:feat_ind+file_sets[_f].RasterCount] = line.copy()
                feat_ind += file_sets[_f].RasterCount

        start_index += len(cover_coords[0])

    output_names = np.array(output_names)

    # Export
    header = ['ID', 'X_UTM', 'Y_UTM',]
    for _f in range(len(file_sets)):
        header.extend([os.path.splitext(os.path.basename(args.source_files[_f]))[0]
                       [-4:] + '_B_' + str(n+1) for n in range(file_sets[_f].RasterCount)])
    out_df = pd.DataFrame(data=output_array, columns=header)
    out_df['covertype'] = output_names
    out_df.to_csv(os.path.join(args.out_base,'cover_extraction.csv'),sep=',', index=False)




if __name__ == "__main__":
    main()

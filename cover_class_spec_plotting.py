
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "Times New Roman"
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.labelpad'] = 6


def main():
    file_path = '~/Google Drive File Stream/My Drive/CB_share/NEON/cover_classification/extraction_output/cover_extraction_20201021.csv'
    wavelengths_file = 'raster_data/neon_wavelengths.txt'
    spectra, cover_types, wv = spec_cleaning(file_path, wavelengths_file, 'refl_B_')
    plot_avg_spectra(spectra, cover_types, wv)


def find_nearest(array_like, v):
    index = np.argmin(np.abs(np.array(array_like) - v))
    return index

def spec_cleaning(file_path, wavelengths_file, band_preface):
    extract = pd.read_csv(file_path)
    headerSpec = list(extract)  # Reflectance bands start at 18 (17 in zero base)

    # defining wavelengths
    wv = np.genfromtxt(wavelengths_file)
    bad_bands = []
    good_band_ranges = []

    bad_band_ranges = [[0, 425], [1345, 1410], [1805, 2020], [2470, 2700]]
    for _bbr in range(len(bad_band_ranges)):
        bad_band_ranges[_bbr] = [find_nearest(wv, x) for x in bad_band_ranges[_bbr]]
        if (_bbr > 0):
            good_band_ranges.append([bad_band_ranges[_bbr-1][1], bad_band_ranges[_bbr][0]])

        for n in range(bad_band_ranges[_bbr][0], bad_band_ranges[_bbr][1]):
            bad_bands.append(n)
    bad_bands.append(len(wv)-1)

    good_bands = np.array([x for x in range(0, 426) if x not in bad_bands])

    # first column of reflectance data
    rfdat = list(extract).index(band_preface + '1')

    all_band_indices = (np.array(good_bands)+rfdat).tolist()
    all_band_indices.extend((np.array(bad_bands)+rfdat).tolist())
    all_band_indices = np.sort(all_band_indices)

    spectra = np.array(extract[np.array(headerSpec)[all_band_indices]])
    spectra[:, bad_bands] = np.nan

    cover_types = extract['covertype']

    return spectra, cover_types, wv


def plot_avg_spectra(spectra, cover_types, wv, brightness_normalize=False):

    c_types = cover_types.unique()

    color_sets = ['navy', 'tan', 'forestgreen', 'royalblue', 'gray', 'darkorange', 'black', 'brown', 'purple']

    # Plot the difference between needles and noneedles in reflectance data
    figure_export_settings = {'dpi': 200, 'bbox_inches': 'tight'}

    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    for _c, cover in enumerate(c_types):
        c_spectra = spectra[cover_types == cover]
        print(color_sets[_c])
        if brightness_normalize:
            scale_factor = 1.
        else:
            scale_factor = 100.
        plt.plot(wv, np.nanmean(c_spectra, axis=0) / scale_factor, c=color_sets[_c], linewidth=2)
        plt.fill_between(wv, np.nanmean(c_spectra, axis=0) / scale_factor - np.nanstd(c_spectra, axis=0) / scale_factor,
                         np.nanmean(
                             c_spectra, axis=0) / scale_factor + np.nanstd(c_spectra, axis=0) / scale_factor, alpha=.15,
                         facecolor=color_sets[_c])

    plt.legend(c_types, prop={'size':10})
    plt.ylabel('Reflectance (%)')
    if brightness_normalize:
        plt.ylabel('Brightness Norm. Reflectance')
    else:
        plt.ylabel('Reflectance (%)')
    plt.xlabel('Wavelength (nm)')

    plt.savefig(os.path.join('figs', 'class_spectra.png'), **figure_export_settings)
    del fig


if __name__ == "__main__":
    main()

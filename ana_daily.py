import os
import pickle
import glob

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

import solar


def read_daily(filename):
    data = solar.read_csv(filename)
    times, power = [np.asarray([x[i] for x in data]) for i in [0, 2]]
    if times[0] == times[12]:
        times, power = times[12:], power[12:]
    if times[0] == times[12]:
        times, power = times[12:], power[12:]
    opt_power, stray_power = solar.compute_powers(times, stray=True)
    return times, power, opt_power, stray_power


if __name__ == "__main__":
    fig = plt.figure()
    grid = AxesGrid(fig, 111,  # similar to subplot(142)
                    nrows_ncols=(4, 3),
                    axes_pad=0.1,
                    aspect=False,
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="2%",
                    cbar_pad="2%")
    cache_file = "daily.pickle"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as pifi:
            hours, powers, maxs, mins, straymean, max_time = pickle.load(pifi)
    else:
        hours = {}
        powers = {}
        maxs = {}
        mins = {}
        straymean = {}
        straymean_cnt = {}
        max_time = {}
        for i in range(12):
            hours[i] = []
            powers[i] = []
            maxs[i] = None
            max_time[i] = np.arange(0, 24, 5 / 60)
        for fn in sorted(glob.glob("*/MyPlant-20??????.csv")):
            if fn in ["2019/MyPlant-20190331.csv"]:
                continue
            filetimes, filepowers, fileoptpowers, filestraypowers = read_daily(fn)
            assert fileoptpowers.shape == filestraypowers.shape
            print(f"{fn} {filepowers.sum():4.0f} {fileoptpowers.sum():4.0f}")
            month = filetimes[0].month - 1
            hours[month].extend([solar.second_of_day(_x) for _x in filetimes])
            powers[month].extend(filepowers)
            if maxs[month] is None:
                maxs[month] = fileoptpowers
                mins[month] = fileoptpowers
                straymean[month] = filestraypowers
                straymean_cnt[month] = 1
                continue

            assert fileoptpowers.shape == maxs[month].shape, \
                   (fileoptpowers.shape, maxs[month].shape)
            assert (np.asarray(max_time[month][:len(fileoptpowers)]) -
                    np.asarray([solar.second_of_day(x) for x in filetimes])).sum() < 1e-4, \
                   (np.asarray(max_time[month][:len(fileoptpowers)]) - np.asarray([solar.second_of_day(x) for x in filetimes]))
            maxs[month] = np.where(
                maxs[month] > fileoptpowers, maxs[month], fileoptpowers)
            mins[month] = np.where(
                mins[month] < fileoptpowers, mins[month], fileoptpowers)
            straymean[month] += filestraypowers
            straymean_cnt[month] += 1

        for month in straymean:
            straymean[month] /= straymean_cnt[month]

        with open(cache_file, "wb") as pifi:
            pickle.dump((hours, powers, maxs, mins, straymean, max_time), pifi)

    for i, month in enumerate(solar.months):
        if len(powers[i]) > 0:
            im = grid[i].hexbin(
                hours[i], powers[i], cmap=plt.cm.gray_r,
                vmax=8, extent=(0, 24, solar.MAX_POWER, 0), rasterized=True)
            grid[i].plot(max_time[i][:len(maxs[i])], maxs[i], color="C1", zorder=100)
            grid[i].plot(max_time[i][:len(mins[i])], mins[i], color="C1", zorder=100)
            grid[i].plot(max_time[i][:len(straymean[i])], straymean[i], color="C0", zorder=100)
        grid[i].set_xticks(np.arange(0, 25, 3))
        grid[i].set_xlim(3, 21)
        grid[i].set_ylim(0, 4)
        grid[i].text(0.05, 0.85, month, ha="left", transform=grid[i].transAxes)
    plt.colorbar(im, grid.cbar_axes[0], extend="max")
    plt.savefig("daily.png")
    plt.savefig("daily.pdf")

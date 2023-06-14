import datetime
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
    data = {}
    for fn in sorted(glob.glob("*/MyPlant-202[23]????.csv")):
        if fn in ["2019/MyPlant-20190331.csv"]:
            continue
        filetimes, filepowers, fileoptpowers, filestraypowers = read_daily(fn)
        month = filetimes[0].month
        total = sum(filepowers)
        if month not in data or data[month][0] < total:
            data[month] = (total, filetimes, filepowers, fileoptpowers)

    # _, times, powers, opts = data[5]
    # print(times[0])
    # times = [(_t.replace(tzinfo=None) - datetime.datetime.combine(_t.replace(tzinfo=None), datetime.datetime.min.time())).total_seconds() for _t in times]
    # plt.plot(times, powers, label=month, color="C0")
    # plt.plot(times, opts, label=month, color="C1")
    # plt.show()
    # exit()
    fig = plt.figure()
    grid = AxesGrid(fig, 111,  # similar to subplot(142)
                    nrows_ncols=(4, 3),
                    axes_pad=0.1,
                    aspect=False,
                    share_all=True,
                    label_mode="L")
    for month, (_, times, powers, opts) in data.items():
        print(times[0])
        times = [(_t.replace(tzinfo=None) - datetime.datetime.combine(_t.replace(tzinfo=None), datetime.datetime.min.time())).total_seconds() for _t in times]
        grid[month - 1].plot(times, powers, label=month, color=f"C0")
        grid[month - 1].plot(times, opts, label=month, color=f"C1")
    plt.legend()
    plt.show()

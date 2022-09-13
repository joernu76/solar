import os
import pickle
import glob
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

from solar import *


if __name__ == "__main__":
    cache_file = "monthly.pickle"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as pifi:
            (times, powers, optpowers, times_mean, powers_mean, times_yearly,
             times_yearly_mean, powers_yearly, powers_yearly_mean
             ) = pickle.load(pifi)
    else:
        times = []
        powers = []
        optpowers = []
        times_mean = []
        powers_mean = []
        times_yearly = {}
        times_yearly_mean = {}
        powers_yearly = {}
        powers_yearly_mean = {}
        for fn in sorted(glob.glob("*/MyPlant-20????.csv")):
            time, power = read_monthly(fn)
            times.extend(time)
            powers.extend(power)
            optpowers.extend(compute_days(time))
            times_mean.append(time[len(time) // 2])
            powers_mean.append(np.mean(power))

            if time[0].year not in times_yearly:
                times_yearly[time[0].year] = []
                times_yearly_mean[time[0].year] = []
                powers_yearly[time[0].year] = []
                powers_yearly_mean[time[0].year] = []
            times_yearly[time[0].year].extend(day_of_year(x) for x in time)
            powers_yearly[time[0].year].extend(power)
            times_yearly_mean[time[0].year].append(day_of_year(time[len(time) // 2]))
            powers_yearly_mean[time[0].year].append(np.mean(power))
        with open(cache_file, "wb") as pifi:
            pickle.dump(
                (times, powers, optpowers, times_mean, powers_mean,
                 times_yearly, times_yearly_mean, powers_yearly, powers_yearly_mean), pifi)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(times, powers, "x", label="daily")
    ax1.plot(times, optpowers, "k-", label="optimum")
    ax1.plot(times_mean, powers_mean, lw=4, label="monthly mean")
    avg_power = {}
    for x, y in zip(times_mean, powers_mean):
        avg_power.setdefault(x.month, []).append(y)
    for i in range(12):
        avg_power[i + 1] = np.mean(avg_power[i + 1])
    powers_avg = [avg_power[x.month] for x in times_mean]
    ax1.plot(times_mean, powers_avg, lw=4, label="multi-year mean")
    ax1.set_ylabel("kWh")
    ax1.set_ylim(0, 31)
    ax1.set_xlim(times[0] - timedelta(days=15), times[-1] + timedelta(days=15))
    ax1.legend()
    for idx, year in enumerate(times_yearly):
        ax2.plot(times_yearly[year], powers_yearly[year], "x",
                 color="C{}".format(idx), label='_nolegend_')

        ax2.plot(times_yearly_mean[year], powers_yearly_mean[year],
                 color="C{}".format(idx), label=year, lw=4)
        continue
    ax2.set_xlim(0, 365)
    ax2.set_xticks(np.arange(15, 365, 365 // 12))
    ax2.set_xticklabels("JFMAMJJASOND")
    ax2.set_ylabel("kWh")
    # ax2.legend()
    fig.savefig("monthly.pdf")
    fig.savefig("monthly.png")

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import tqdm

import solar


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2)
    axs = axs.reshape(-1)
    axs[0].set_ylabel("kW")
    axs[0].set_xlabel("time (hours of day)")
    # axs[1].set_ylabel("kW")
    axs[1].set_xlabel("azimuth")
    # axs[2].set_ylabel("kW")
    axs[2].set_xlabel("elevation")
    axs[3].set_ylabel("elevation")
    axs[3].set_xlabel("azimuth")
    axs[1].axvline(solar.AZIMUTH, color="k")
    axs[2].axvline(solar.ELEVATION, color="k")
    axs[3].axvline(solar.AZIMUTH, color="C2")
    axs[3].axhline(solar.ELEVATION, color="C2")
    az_bak = solar.AZIMUTH
    el_bak = solar.ELEVATION

    # comparison
    minutes = np.arange(0, 60 * 24, 5.)
    azs = np.arange(90, 271, 5.)
    eles = np.arange(0, 90, 5.)
    for month in range(2, 12, 3):
        now = datetime(2022, month + 1, 15)
        now = solar.UTC.localize(now)
        dts = [now + timedelta(minutes=x) for x in minutes]
        power = solar.compute_powers(dts, stray=True)
        axs[0].plot(minutes / 60, power[0], label=solar.months[month])

        powers = []
        for az in azs:
            dts = [now + timedelta(minutes=x) for x in minutes]
            solar.AZIMUTH = az
            power = solar.compute_powers(dts, stray=False)
            powers.append(power.sum() / 12)
            # axs[1].plot(minutes, power, label=str(az))
            solar.AZIMUTH = az_bak
        axs[1].plot(azs, powers / max(powers))
        powers = []
        for ele in eles:
            dts = [now + timedelta(minutes=x) for x in minutes]
            solar.ELEVATION = ele
            power = solar.compute_powers(dts, stray=False)
            powers.append(power.sum() / 12)
            # axs[1].plot(minutes, power, label=str(az))
            solar.ELEVATION = el_bak
        axs[2].plot(eles, powers / max(powers))
    axs[0].legend()

    # overview
    delta_min = 30.
    day_skip = 5
    days = np.arange(0, 365, day_skip)
    minutes = np.concatenate(
        [x * 60 * 24 + np.arange(0, 60 * 24, delta_min) for x in days])
    azs = np.arange(90, 270.5, 15.)
    eles = np.arange(0, 90.5, 5.)

    now = datetime(2022, 1, 1)
    now = solar.UTC.localize(now)
    dts = [now + timedelta(minutes=x) for x in minutes]
    powers = np.zeros((len(eles), len(azs)))
    for ie, ele in enumerate(tqdm.tqdm(eles)):
        for ia, az in enumerate(azs):
            solar.AZIMUTH = az
            solar.ELEVATION = ele
            # print(AZIMUTH, ELEVATION)
            power = solar.compute_powers(dts, stray=True)
            powers[ie, ia] = day_skip * (power.sum() / (60 / delta_min)) / 1000
    solar.AZIMUTH = az_bak
    solar.ELEVATION = el_bak
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((eles, azs), powers)

    azs_p = np.arange(90, 270.5, 1.)
    eles_p = np.arange(0, 90.5, 1.)
    x, y = np.meshgrid(eles_p, azs_p)
    powers_p = interp((x, y)).T

    pm = axs[3].pcolormesh(azs_p, eles_p, powers_p, vmin=4.5, vmax=8.5, cmap="magma")
    co = axs[3].contour(azs_p, eles_p, powers_p, np.arange(5, 9, 0.5), colors="w")
    plt.clabel(co)
    axs[3].set_xticks(azs)
    axs[3].set_yticks(eles)
    cb = plt.colorbar(pm, ax=axs[3], label="MWh")
    cb.add_lines(co)
    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.savefig("comparison.pdf")
    plt.show()

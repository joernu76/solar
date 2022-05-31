import os
import pickle
import glob
import pytz
import codecs
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np


# configuration of solar array
MAX_POWER = 3.92  # ~= 16 * 1.65 * 0.992 * 0.1497
AZIMUTH = 130  # 0 north, 90 east, ...
ELEVATION = 90 - 35  # 90 is a panel lying flat on the ground
LONGITUDE = 6.154
LATITUDE = 50.747
ALTITUDE = 0.26

# loss of efficiency due to temperature
BETA = -0.41  # %/C
NOCT = 45  # +-2 C

UTC = pytz.utc
CET = pytz.timezone("Europe/Berlin")
JSEC_START = UTC.localize(datetime(2000, 1, 1))


months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def get_average_temp(doy, hour):
    """
    very rough approximation of Aachen conditions.
    A bit hand tuned :-)
    """
    # -5 - 25
    # [1, 2, 6, 8, 12, 15, 17, 17, 14, 11, 6, 3]
    # yearly variation
    t_mean = 10 - 10 * np.cos((doy - 30) / 365 * 2 * np.pi)
    # daily variation
    t_mean += 10 * np.cos(2 * np.pi * (hour - 16) / 24)
    return t_mean


def datetime_to_jsec(dt):
    """
    Calculate seconds since Jan 01 2000.
    """
    delta = dt - JSEC_START
    total = delta.days * 3600 * 24
    total += delta.seconds
    total += delta.microseconds * 1e-6
    return total


def compute_hour_of_day(jsecs):
    date = JSEC_START + timedelta(seconds=jsecs)
    return date.hour + date.minute / 60 + date.second / 3600


def compute_solar_angle(jsec, lon, lat):
    # The input to the Astronomer's almanach is the difference between
    # the Julian date and JD 2451545.0 (noon, 1 January 2000)
    time = jsec / (60 * 60 * 24) - 0.5

    # Mean longitude
    mnlong = 280.460 + 0.9856474 * time
    mnlong = mnlong % 360.
    if (mnlong < 0):
        mnlong += 360
        assert(mnlong >= 0)

    # Mean anomaly
    mnanom = 357.528 + .9856003 * time
    mnanom = np.deg2rad(mnanom % 360.)
    if (mnanom < 0):
        mnanom += 2 * np.pi
        assert (mnanom >= 0)

    # Ecliptic longitude and obliquity of ecliptic
    eclong = mnlong + 1.915 * np.sin(mnanom) + 0.020 * np.sin(2 * mnanom)
    eclong = np.deg2rad(eclong % 360.)
    if (eclong < 0):
        eclong += 2 * np.pi
        assert(eclong >= 0)

    oblqec = np.deg2rad(23.439 - 0.0000004 * time)

    # Celestial coordinates
    # Right ascension and declination
    num = np.cos(oblqec) * np.sin(eclong)
    den = np.cos(eclong)
    ra = np.arctan(num / den)
    if (den < 0):
        ra += np.pi
    elif (den >= 0 and num < 0):
        ra += 2 * np.pi

    dec = np.arcsin(np.sin(oblqec) * np.sin(eclong))
    # Local coordinates
    # Greenwich mean sidereal time
    gmst = 6.697375 + 0.0657098242 * time + compute_hour_of_day(jsec)

    gmst = gmst % 24.
    if (gmst < 0):
        gmst += 24
        assert gmst >= 0

    # Local mean sidereal time
    if (lon < 0):
        lon += 360
        assert 0 <= lon <= 360

    lmst = gmst + lon / 15.
    lmst = np.deg2rad(15. * (lmst % 24.))

    # Hour angle
    ha = lmst - ra
    if (ha < -np.pi):
        ha += 2 * np.pi

    if (ha > np.pi):
        ha -= 2 * np.pi

    assert(-np.pi < ha < 2 * np.pi)

    # Latitude to radians
    lat = np.deg2rad(lat)

    # Azimuth and elevation
    zenithAngle = np.arccos(
        np.sin(lat) * np.sin(dec) +
        np.cos(lat) * np.cos(dec) * np.cos(ha))
    azimuthAngle = np.arccos(
        (np.sin(lat) * np.cos(zenithAngle) - np.sin(dec)) /
        (np.cos(lat) * np.sin(zenithAngle)))

    if (ha > 0):
        azimuthAngle += np.pi
    else:
        azimuthAngle = 3 * np.pi - azimuthAngle % (2 * np.pi)

    if (azimuthAngle > np.pi):
        azimuthAngle -= 2 * np.pi

    return np.rad2deg(azimuthAngle), 90 - np.rad2deg(zenithAngle)


def convert_date(string):
    sep = "."
    if "/" in string:
        sep = "/"
    if len(string.split()) == 1:
        day, month, year = [int(x) for x in string.split(sep)]

        dt = datetime(year=year, month=month, day=day)
        dt = UTC.localize(dt)
    else:
        date, time = string.split()
        day, month, year = [int(x) for x in date.split(sep)]
        hour, minute, second = [int(x) for x in time.split(":")]
        # fixes for wrong clocks on SMA
        if (
            (year == 2018 and day > 25 and month == 3) or
            (year == 2018 and month == 4)
        ):
            hour += 1
        elif 2013 < year < 2018:
            hour -= 1
        if not (2 < hour < 23):
            return None
        dt = datetime(year=year, month=month, day=day,
                      hour=hour, minute=minute, second=second)
        if year >= 2018:
            dt = CET.localize(dt).astimezone(UTC)
        else:
            dt = UTC.localize(dt)
    return dt


def read_csv(filename):
    try:
        with codecs.open(filename, "r", "utf-16") as fh:
            lines = fh.readlines()
    except BaseException:
        with open(filename, "r") as fh:
            lines = fh.readlines()
    data = [x.strip().replace("---", "0").split(";")
            for x in lines[3:] if not x.startswith(";")][1:]
    data = [(convert_date(x), float(y.replace(",", ".")), float(z.replace(",", "."))) for x, y, z in data]
    data = [(x, y, z) for x, y, z in data if x is not None]
    start_dt = data[0][0].replace(hour=0, minute=0, second=0, microsecond=0)
    while data[0][0] > start_dt:
        data.insert(0, (data[0][0] - timedelta(minutes=5), data[0][1], 0))
    end_dt = start_dt + timedelta(hours=24, minutes=-5)
    while data[-1][0] < end_dt:
        data.append((data[-1][0] + timedelta(minutes=5), data[-1][1], 0))
    return data


def cosd(x):
    return np.cos(np.deg2rad(x))


def sind(x):
    return np.sin(np.deg2rad(x))


def day_of_year(dt):
    return dt.timetuple().tm_yday


def second_of_day(x):
    return x.hour + x.minute / 60 + x.second / (60 * 60)


def compute_power(dts, stray=False):
    """
    Computes maximal power output under optimal conditions for given
    array of datetimes
    """
    power = []
    straypower = []
    for dt in dts:
        jsec = datetime_to_jsec(dt)
        azi, ele = compute_solar_angle(jsec, LONGITUDE, LATITUDE)

        if ele <= 0:
            power.append(0)
            straypower.append(0)
            continue

        if azi < 0:
            azi += 360

        # https://en.wikipedia.org/wiki/Great-circle_distance
        # i.e. this is the cosine of angle between normal vector and sun
        fac = (cosd(ele) * cosd(ELEVATION) * cosd(AZIMUTH - azi) +
               sind(ele) * sind(ELEVATION))
        # reflection correction, causes loss of a couple of percent at dusk
        # https://www.osti.gov/servlets/purl/1350025
        AOI = np.arccos(fac)
        n_glass = 1.526
        r_0 = 0.0434  # air/glass
        AOI_r = np.arcsin(np.sin(AOI) / n_glass)
        r_AOI = 0.5 * ((np.sin(AOI_r - AOI) ** 2 / np.sin(AOI_r + AOI) ** 2)
                       + (np.tan(AOI_r - AOI) ** 2 / np.tan(AOI_r + AOI) ** 2))
        corr_reflection = max(0, (1 - r_AOI) / (1 - r_0))

        # solar radiation at top of atmosphere
        # https://en.wikipedia.org/wiki/Solar_irradiance
        I = 1.360  # kW/m^2
        # correction for distance of sun from earth
        I *= 1 + 0.033 * np.cos(2 * np.pi * (day_of_year(dt) / 365))
        # attenuation -> https://en.wikipedia.org/wiki/Air_mass_(solar_energy)
        AM = 1. / (sind(ele) + 0.50572 * (6.07995 + ele) ** -1.6364)
        I *= (1 - ALTITUDE / 7.1) * (0.7 ** (AM ** 0.678)) + (ALTITUDE / 7.1)
        # + 15% for diffusion (10% according to wikipedia..?)
        fac = max(0.15, fac * corr_reflection + 0.15)
        # Correction for cell efficiency due to temperature
        # http://crossmark.crossref.org/dialog/?doi=10.1016/j.egypro.2014.10.282&domain=pdf
        t_a = get_average_temp(day_of_year(dt), dt.hour + dt.minute / 60)
        t_c = t_a + (NOCT - 20) * (1000 * I) * fac / 800
        # adhoc (1.3) change of efficiency reduction due
        # to problems in summer (dust??)
        fac2 = 1 + (1.3 * BETA) * (t_c - 25) / 100
        power.append(MAX_POWER * fac * fac2 * I)
        straypower.append(MAX_POWER * 0.15 * fac2 * I)

    if stray:
        return np.asarray(power), np.asarray(straypower)
    return np.asarray(power)


def read_daily(filename):
    data = read_csv(filename)
    times, power = [np.asarray([x[i] for x in data]) for i in [0, 2]]
    if times[0] == times[12]:
        times, power = times[12:], power[12:]
    if times[0] == times[12]:
        times, power = times[12:], power[12:]
    opt_power, stray_power = compute_power(times, stray=True)
    return times, power, opt_power, stray_power


def read_monthly(filename):
    data = read_csv(filename)
    times, power = [[x[i] for x in data] for i in [0, 2]]
    return times, power


def ana_daily():
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
            hours[month].extend([second_of_day(_x) for _x in filetimes])
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
                    np.asarray([second_of_day(x) for x in filetimes])).sum() < 1e-4, \
                   (np.asarray(max_time[month][:len(fileoptpowers)]) - np.asarray([second_of_day(x) for x in filetimes]))
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

    for i, month in enumerate(months):
        if len(powers[i]) > 0:
            im = grid[i].hexbin(
                hours[i], powers[i], cmap=plt.cm.gray_r,
                vmax=5, extent=(0, 24, MAX_POWER, 0), rasterized=True)
            grid[i].plot(max_time[i][:len(maxs[i])], maxs[i], color="C1", zorder=100)
            grid[i].plot(max_time[i][:len(mins[i])], mins[i], color="C1", zorder=100)
            grid[i].plot(max_time[i][:len(straymean[i])], straymean[i], color="C0", zorder=100)
        grid[i].set_xticks(np.arange(0, 25, 3))
        grid[i].set_xlim(3, 21)
        grid[i].set_ylim(0, 4)
        grid[i].text(0.05, 0.85, month, ha="left", transform=grid[i].transAxes)
    plt.colorbar(im, grid.cbar_axes[0], extend="max")
    plt.savefig("daily.pdf")


def compute_day(time):
    minutes = 20
    opt_power = compute_power(
        [time + timedelta(minutes=x) for x in range(0, 24 * 60, minutes)])
    return np.sum(opt_power) / (60 / minutes)


def compute_days(times):
    return [compute_day(time) for time in times]


def ana_monthly():
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
    ax1.legend()
    ax1.set_ylabel("kWh")
    ax1.set_ylim(0, 31)
    ax1.set_xlim(times[0] - timedelta(days=15), times[-1] + timedelta(days=15))
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
    ax2.legend()
    fig.savefig("monthly.pdf")


if __name__ == "__main__":
    ana_daily()
    #ana_monthly()
    plt.show()

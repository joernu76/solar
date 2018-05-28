import codecs
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import os
import pickle
import numpy as np
import glob


JSEC_START = datetime.datetime(2000, 1, 1)

MAX_POWER = 3.92  # ~= 16 * 1.65 * 0.992 * 0.1497
RED_POWER = 0.7 * MAX_POWER

BETA = -0.41  # %/C
NOCT = 45  # +-2 C
TR = 25


def get_average_temp(doy):
    # -5 - 25
    return 10 - 15 * np.cos((doy - 30) / 365 * 2 * np.pi)


def datetime_to_jsec(dt):
    """
    Calculate seconds since Jan 01 2000.
    """
    delta = dt - JSEC_START
    total = delta.days * 3600 * 24
    total += delta.seconds
    total += delta.microseconds * 1e-6
    return total


def computeHourOfDay(jsecs):
    date = JSEC_START + datetime.timedelta(seconds=jsecs)
    return date.hour + date.minute / 60. + date.second / 3600.


def compute_solar_angle(jsec, lon, lat):
    # The input to the Astronomer's almanach is the difference between
    # the Julian date and JD 2451545.0 (noon, 1 January 2000)
    time = jsec / (60. * 60. * 24.) - 0.5

    # Mean longitude
    mnlong = 280.460 + .9856474 * time
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
    gmst = 6.697375 + .0657098242 * time + computeHourOfDay(jsec)

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
    zenithAngle = np.arccos(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha))
    azimuthAngle = np.arccos((np.sin(lat) * np.cos(zenithAngle) - np.sin(dec)) /
                             (np.cos(lat) * np.sin(zenithAngle)))

    if (ha > 0):
        azimuthAngle += np.pi
    else:
        azimuthAngle = 3 * np.pi - azimuthAngle % (2 * np.pi)

    if (azimuthAngle > np.pi):
        azimuthAngle -= 2 * np.pi

    return np.rad2deg(azimuthAngle), 90 - np.rad2deg(zenithAngle)


def convert_date(string):
    if len(string.split()) == 1:
        day, month, year = [int(x) for x in string.split(".")]
        return datetime.datetime(year=year, month=month, day=day)
    else:
        date, time = string.split()
        day, month, year = [int(x) for x in date.split(".")]
        hour, minute, second = [int(x) for x in time.split(":")]
        return datetime.datetime(year=year, month=month, day=day,
                                 hour=hour, minute=minute, second=second)


def read_csv(filename):
    try:
        with codecs.open(filename, "r", "utf-16") as fh:
            lines = fh.readlines()
    except:
        with open(filename, "r") as fh:
            lines = fh.readlines()
    data = [x.strip().replace("---", "0").split(";") for x in lines[3:] if not x.startswith(";")][1:]
    data = [(convert_date(x), float(y.replace(",", ".")), float(z.replace(",", "."))) for x, y, z in data]
    return data


def cosd(x):
    return np.cos(np.deg2rad(x))


def sind(x):
    return np.sin(np.deg2rad(x))


def day_of_year(dt):
    return dt.timetuple().tm_yday


def compute_power(dts):
    power2 = []
    for dt in dts:
        jsec = datetime_to_jsec(dt)
        # Fix for apparently wrong clock/timezone? in data
        # Correction by 8 minutes on 2018-05-14
        if dt.year >= 2014:
            jsec -= 3600
        azi, ele = compute_solar_angle(jsec, 6.154, 50.747)
        if azi < 0:
            azi += 360

        # refraction
        if ele > 0:
            azi_mod = 130
            ele_mod = 90 - 35
            doy = day_of_year(dt)
            t_a = get_average_temp(doy)
            # https://en.wikipedia.org/wiki/Great-circle_distance
            fac = cosd(ele) * cosd(ele_mod) * cosd(azi_mod - azi) + sind(ele) * sind(ele_mod)
            # solar radiation at top of atmosphere
            I = 1.353
            # correction for distance of sun from earth
            I *= 1 + 0.033 * cosd(360 * day_of_year(dt) / 365)
            # attenuation -> https://en.wikipedia.org/wiki/Air_mass_(solar_energy)
            AM = 1. / (sind(ele) + 0.50572 * (6.07995 + ele) ** -1.6364)
            I *= 0.7 ** AM ** 0.678
            # + 15% for diffusion
            fac = max(0.15, fac + 0.15)
            # Correction for cell efficiency due to temperature
            # http://crossmark.crossref.org/dialog/?doi=10.1016/j.egypro.2014.10.282&domain=pdf
            t_c = t_a + (NOCT - 20) * 1000 * I * fac / 800
            fac2 = 1 + BETA * (t_c - 25) / 100
            power2.append(MAX_POWER * fac * fac2 * I)
        else:
            power2.append(0)

    return power2


def read_daily(filename):
    data = read_csv(filename)
    times, power = [[x[i] for x in data] for i in [0, 2]]
    power2 = compute_power(times)
    print(sum(power) / 12, sum(power2) / 12)
    return times, power, power2


def read_monthly(filename):
    data = read_csv(filename)
    times, power = [[x[i] for x in data] for i in [0, 2]]
    return times, power


def ana_daily():
    cache_file = "daily.pickle"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as pifi:
            hours, powers, maxs, max_time = pickle.load(pifi)
    else:
        hours = {}
        powers = {}
        maxs = {}
        for i in range(12):
            hours[i] = []
            powers[i] = []
            maxs[i] = None
        for fn in sorted(glob.glob("My PV plant 1-201[34]????.csv")):
            print(fn)
            a, b, c = read_daily(fn)

            month = a[0].month - 1
            hours[month].extend(
                [x.hour + x.minute / 60. + x.second / 60. / 60. for x in a])
            powers[month].extend(b)
            c = np.asarray(c)
            if maxs[month] is None:
                max_time = [
                    x.hour + x.minute / 60. + x.second / 60. / 60. for x in a]
                maxs[month] = c
            elif c.shape == maxs[month].shape:
                maxs[month] = np.where(maxs[month] > c, maxs[month], c)
        with open(cache_file, "wb") as pifi:
            pickle.dump((hours, powers, maxs, max_time), pifi)

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
    for i in range(12):
        if len(powers[i]) > 0:
            im = grid[i].hexbin(
                hours[i], powers[i], cmap=plt.cm.gray_r, vmax=5, extent=(0, 24, MAX_POWER, 0))
            grid[i].plot(max_time, maxs[i], color="C1")
            grid[i].plot(max_time, np.where(RED_POWER < maxs[i], RED_POWER, maxs[i]), color="C1")
    grid.cbar_axes[0].colorbar(im, extend="max")


def ana_daily2():
    hours = {}
    powers = {}
    maxs = {}
    for i in range(12):
        hours[i] = []
        powers[i] = []
        maxs[i] = None
    plt.figure()
    files = sorted(glob.glob("My PV plant 1-201312[23]?.csv") + glob.glob("My PV plant 1-2014010?.csv"))
    n = round(np.sqrt(len(files)) / np.sqrt(1.61))
    m = len(files) // n + 1
    for idx, fn in enumerate(files):
        print(fn)
        a, b, c = read_daily(fn)

        hour = [x.hour + x.minute / 60. + x.second / 60. / 60. for x in a]
        plt.subplot(n, m, idx + 1)
        plt.title(fn)
        plt.plot(hour, b, "x")
        plt.plot(hour, c)


def compute_day(time):
    power = compute_power(
        [time + datetime.timedelta(minutes=x) for x in range(0, 24 * 60, 20)])
    redux_power = [min(3.0, x) for x in power]
    return np.sum(power) / 3., np.sum(redux_power) / 3.


def compute_days(times):
    return [compute_day(time) for time in times]


def ana_monthly():
    cache_file = "monthly.pickle"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as pifi:
            time1, power1, power2, power3, time1m, power1m, timey, timeym, powery, powerym, powery2, powery3 = pickle.load(pifi)
    else:
        time1 = []
        power1 = []
        power2 = []
        power3 = []
        time1m = []
        power1m = []
        timey = {}
        timeym = {}
        powery = {}
        powerym = {}
        powery2 = {}
        powery3 = {}
        for fn in sorted(glob.glob("My PV plant 1-201???.csv")):
            print(fn)
            time, power = read_monthly(fn)
            time1.extend(time)
            power1.extend(power)
            time1m.append(time[len(time) // 2])
            power1m.append(np.mean(power))
            cds = compute_days(time)
            power2.extend([cd[0] for cd in cds])
            power3.extend([cd[1] for cd in cds])

            if time[0].year not in timey:
                timey[time[0].year] = []
                timeym[time[0].year] = []
                powery[time[0].year] = []
                powerym[time[0].year] = []
                powery2[time[0].year] = []
                powery3[time[0].year] = []
            timey[time[0].year].extend(day_of_year(x) for x in time)
            powery[time[0].year].extend(power)
            powery2[time[0].year].extend([cd[0] for cd in cds])
            powery3[time[0].year].extend([cd[1] for cd in cds])
            timeym[time[0].year].append(day_of_year(time[len(time) // 2]))
            powerym[time[0].year].append(np.mean(power))
        with open(cache_file, "wb") as pifi:
            pickle.dump(
                (time1, power1, power2, power3, time1m, power1m, 
                 timey, timeym, powery, powerym, powery2, powery3), pifi)

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(time1, power1, "x")
    plt.plot(time1, power2, "k-")
    plt.plot(time1, power3, "k-")
    plt.plot(time1m, power1m, lw=4)

    plt.subplot(2, 1, 2)
    plt.plot(timey[2014], powery2[2014], "k-")
    plt.plot(timey[2014], powery3[2014], "k-")
    for idx, year in enumerate(timey):
        plt.plot(timey[year], powery[year], "x",
                 color="C{}".format(idx), label='_nolegend_')
        plt.plot(timeym[year], powerym[year],
                 color="C{}".format(idx), label=year, lw=4)
    plt.xlim(0, 365)
    plt.legend()


if __name__ == "__main__":
    ana_daily()
    # ana_daily2()
    ana_monthly()
    plt.show()


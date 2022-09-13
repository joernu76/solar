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
import tqdm


# configuration of solar array
MAX_POWER = 3.92  # ~= 16 * 1.65 * 0.992 * 0.1497
AZIMUTH = 135  # 0 north, 90 east, ...
ELEVATION = 90 - 35  # 35  # 90 is a panel lying flat on the ground
LONGITUDE = 6.154
LATITUDE = 50.747
ALTITUDE = 0.26

# loss of efficiency due to temperature
BETA = -0.41  # %/C
NOCT = 45  # +-2 C

UTC = pytz.utc
CET = pytz.timezone("Europe/Berlin")
JSEC_START = UTC.localize(datetime(2000, 1, 1))

YEARLY_TEMP_MEAN = 10.
YEARLY_TEMP_VAR = 10.
DAILY_TEMP_VAR = 5.

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def get_average_temp(doy, hour):
    """
    very rough approximation of Aachen conditions.
    Very hand tuned :-)
    """
    # -5 - 25
    # [1, 2, 6, 8, 12, 15, 17, 17, 14, 11, 6, 3]
    # yearly variation
    t_mean = YEARLY_TEMP_MEAN - \
        YEARLY_TEMP_VAR * np.cos(2 * np.pi * (doy - 15) / 365)
    # daily variation
    t_mean += DAILY_TEMP_VAR * np.cos(2 * np.pi * (hour - 16) / 24)
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
    data = [
        (convert_date(x), float(y.replace(",", ".")), float(z.replace(",", ".")))
        for x, y, z in data]
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


def compute_power(dt, stray=False):
    """
    Computes maximal power output under optimal conditions for given
    array of datetimes
    """
    jsec = datetime_to_jsec(dt)
    azi, ele = compute_solar_angle(jsec, LONGITUDE, LATITUDE)

    # This does not take into account hills mountains,
    # partially visible disk, refraction, etc.
    # I.e. bad close to horizon, but power is small anyway
    if ele <= 0:
        if stray:
            return 0, 0
        return 0

    if azi < 0:
        azi += 360

    # https://en.wikipedia.org/wiki/Great-circle_distance
    # i.e. this is the cosine of angle between normal vector and sun
    # print(dt, azi, ele, AZIMUTH, ELEVATION)
    incident_angle_fac = (
        cosd(ele) * cosd(ELEVATION) * cosd(AZIMUTH - azi) +
        sind(ele) * sind(ELEVATION))

    # reflection correction, causes loss of a couple of percent at dusk
    # https://www.osti.gov/servlets/purl/1350025
    # I.e. actually fully negligible
    AOI = np.arccos(incident_angle_fac)
    n_glass = 1.526
    r_0 = 0.0434  # air/glass
    AOI_r = np.arcsin(np.sin(AOI) / n_glass)
    r_AOI = 0.5 * ((np.sin(AOI_r - AOI) ** 2 / np.sin(AOI_r + AOI) ** 2)
                   + (np.tan(AOI_r - AOI) ** 2 / np.tan(AOI_r + AOI) ** 2))
    corr_reflection = max(0, (1 - r_AOI) / (1 - r_0))

    # solar radiation at top of atmosphere
    # https://en.wikipedia.org/wiki/Solar_irradiance
    intensity = 1.360  # kW/m^2

    # correction for distance of sun from earth
    intensity *= 1 + 0.033 * np.cos(2 * np.pi * (day_of_year(dt) / 365))

    # attenuation -> https://en.wikipedia.org/wiki/Air_mass_(solar_energy)
    airmass = 1. / (sind(ele) + 0.50572 * (6.07995 + ele) ** -1.6364)
    intensity *= (
        (1 - ALTITUDE / 7.1) * (0.7 ** (airmass ** 0.678)) +
        (ALTITUDE / 7.1))

    incident_angle_fac = incident_angle_fac * corr_reflection

    # Correction for cell efficiency due to temperature
    # http://crossmark.crossref.org/dialog/?doi=10.1016/j.egypro.2014.10.282&domain=pdf
    t_a = get_average_temp(day_of_year(dt), dt.hour + dt.minute / 60)
    t_c = t_a + (NOCT - 20) * (1000 * intensity) * incident_angle_fac / 800
    temperature_fac = 1 + (1.0 * BETA) * (t_c - 25) / 100

    # derating due to overheating of inverter is missing!
    # affects noon in July/August, mostly
    # https://www.photovoltaik4all.de/media/pdf/34/12/c6/SMA-Wirkungungrade-Derat-TI-de-44.pdf
    fullpower = MAX_POWER * temperature_fac * intensity

    # 10% for diffusion (according to wikipedia..? do not find the link
    # anymore)
    diffusion = 0.1  # 10 %
    power = fullpower * (diffusion + (1 - diffusion) * incident_angle_fac)

    if stray:
        straypower = fullpower * diffusion
        return power, straypower
    return power


def compute_powers(dts, stray=False):
    return np.asarray([compute_power(dt, stray) for dt in dts]).T


def compute_day(time):
    minutes = 20
    opt_power = compute_powers(
        [time + timedelta(minutes=x) for x in range(0, 24 * 60, minutes)])
    return np.sum(opt_power) / (60 / minutes)


def compute_days(times):
    return [compute_day(time) for time in times]

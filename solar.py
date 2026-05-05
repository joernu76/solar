import pytz
import codecs
from datetime import datetime, timedelta
import numpy as np


# configuration of solar array
MAX_POWER = 3.92  # ~= 16 * 1.65 * 0.992 * 0.1497
AZIMUTH = 135  # 0 north, 90 east, ...
ELEVATION = 90 - 35  # 35  # 90 is a panel lying flat on the ground
LONGITUDE = 6.154
LATITUDE = 50.747
ALTITUDE = 0.26

# inverter: SMA Sunny Boy 4000TL-21, nameplate AC hard cap
INVERTER_AC_MAX = 4.0  # kW

# combined system efficiency derate applied before the AC cap:
# inverter DC→AC (~97 %), DC wiring/mismatch (~2 %); set to 1.0 to disable
SYSTEM_EFFICIENCY = 0.97 * 0.98

# scale applied to the climatological monthly-average Linke turbidity in
# compute_power_pvlib.  Monthly means overstate haze on the clearest days:
# best January days have TL ≈ 2.0-2.2 while the climatological Jan mean is
# 3.35, so the scale factor should be around 0.6-0.75 for winter.  Increase
# toward 1.0 if summer comparisons show systematic overestimation.
LINKE_TURBIDITY_SCALE = 0.75

# loss of efficiency due to temperature
BETA = -0.41  # %/C
# NOCT assumes 800 W/m^2, 20 C, 1 m/s wind; roof-mounted modules with limited
# back ventilation run hotter than the datasheet value of ~45 C
# upped to 60!C to fit data
NOCT = 45  # C

UTC = pytz.utc
CET = pytz.timezone("Europe/Berlin")
JSEC_START = UTC.localize(datetime(2000, 1, 1))

YEARLY_TEMP_MEAN = 10.
YEARLY_TEMP_VAR = 20.
DAILY_TEMP_VAR = 5

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
        assert mnlong >= 0

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
        assert eclong >= 0

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

    assert -np.pi < ha < 2 * np.pi

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

    # cosine of angle of incidence between sun and panel normal
    # https://en.wikipedia.org/wiki/Great-circle_distance
    cos_aoi = (
        cosd(ele) * cosd(ELEVATION) * cosd(AZIMUTH - azi) +
        sind(ele) * sind(ELEVATION))
    cos_aoi_pos = max(0.0, cos_aoi)  # sun behind panel contributes no direct beam

    # beam reflection correction (Fresnel, only meaningful for cos_aoi > 0)
    # https://www.osti.gov/servlets/purl/1350025
    n_glass = 1.526
    r_0 = 0.0434  # air/glass at normal incidence
    if cos_aoi_pos > 0:
        AOI = np.arccos(cos_aoi_pos)
        AOI_r = np.arcsin(np.sin(AOI) / n_glass)
        r_AOI = 0.5 * ((np.sin(AOI_r - AOI) ** 2 / np.sin(AOI_r + AOI) ** 2)
                       + (np.tan(AOI_r - AOI) ** 2 / np.tan(AOI_r + AOI) ** 2))
        corr_reflection = max(0.0, (1 - r_AOI) / (1 - r_0))
    else:
        corr_reflection = 0.0

    # atmospheric transmittance (Meinel beam + Liu-Jordan diffuse)
    # https://en.wikipedia.org/wiki/Air_mass_(solar_energy)
    airmass = 1. / (sind(ele) + 0.50572 * (6.07995 + ele) ** -1.6364)
    tau_b = ((1 - ALTITUDE / 7.1) * (0.7 ** (airmass ** 0.678))
             + (ALTITUDE / 7.1))
    tau_d = max(0.0, 0.271 - 0.294 * tau_b)

    # top-of-atmosphere irradiance with eccentricity correction [kW/m^2]
    I0 = 1.360 * (1 + 0.033 * np.cos(2 * np.pi * day_of_year(dt) / 365))

    # component irradiances [kW/m^2]
    dni = I0 * tau_b                 # direct, normal to sun
    dhi = I0 * sind(ele) * tau_d     # diffuse, on horizontal
    ghi = dni * sind(ele) + dhi      # global, on horizontal

    # plane-of-array irradiance (isotropic sky + ground reflection)
    tilt = 90 - ELEVATION
    vf_sky = (1 + cosd(tilt)) / 2
    vf_ground = (1 - cosd(tilt)) / 2
    albedo = 0.2  # grass/urban mix

    beam_poa = dni * cos_aoi_pos * corr_reflection
    diffuse_poa = dhi * vf_sky
    ground_poa = ghi * albedo * vf_ground
    poa = beam_poa + diffuse_poa + ground_poa
    stray_poa = diffuse_poa + ground_poa

    # cell temperature from plane-of-array irradiance
    # http://crossmark.crossref.org/dialog/?doi=10.1016/j.egypro.2014.10.282&domain=pdf
    t_a = get_average_temp(day_of_year(dt), dt.hour + dt.minute / 60)
    t_c = t_a + (NOCT - 20) * 1000 * poa / 800
    temperature_fac = 1 + BETA * (t_c - 25) / 100

    # system efficiency derate then inverter nameplate AC cap
    power = min(MAX_POWER * temperature_fac * poa * SYSTEM_EFFICIENCY, INVERTER_AC_MAX)

    if stray:
        straypower = min(MAX_POWER * temperature_fac * stray_poa * SYSTEM_EFFICIENCY, INVERTER_AC_MAX)
        return power, straypower
    return power


def compute_power_pvlib(dt, stray=False):
    """
    Computes power output using pvlib.  Uses the Ineichen clear-sky model
    (location/time-specific Linke turbidity) instead of the simple
    Meinel/Liu-Jordan parameterisation; everything else follows the same
    algorithm as compute_power().
    """
    import pvlib
    import pandas as pd

    tilt = 90 - ELEVATION  # tilt from horizontal [deg]
    times = pd.DatetimeIndex([dt])

    # Solar position; apparent_zenith (refraction-corrected) is appropriate
    # for irradiance calculations and for the Ineichen model.
    solar_pos = pvlib.solarposition.get_solarposition(times, LATITUDE, LONGITUDE)
    apparent_zenith = float(solar_pos['apparent_zenith'].iloc[0])
    azimuth = float(solar_pos['azimuth'].iloc[0])

    if apparent_zenith >= 90:
        return (0, 0) if stray else 0

    # Fresnel IAM – physical model with n=1.526, K=0 (no absorption loss)
    aoi_deg = float(pvlib.irradiance.aoi(tilt, AZIMUTH, apparent_zenith, azimuth))
    iam = float(pvlib.iam.physical(aoi_deg, n=1.526, K=0))

    # Ineichen clear-sky model: uses pressure-corrected airmass and a
    # monthly Linke turbidity factor looked up for this location/date.
    rel_am = pvlib.atmosphere.get_relative_airmass(
        apparent_zenith, model='kastenyoung1989')
    abs_am = pvlib.atmosphere.get_absolute_airmass(
        rel_am, pvlib.atmosphere.alt2pres(ALTITUDE * 1000))  # ALTITUDE km → m
    tl = float(pvlib.clearsky.lookup_linke_turbidity(times, LATITUDE, LONGITUDE).iloc[0]) * LINKE_TURBIDITY_SCALE
    dni_extra = float(pvlib.irradiance.get_extra_radiation(day_of_year(dt)))
    cs = pvlib.clearsky.ineichen(apparent_zenith, abs_am, tl,
                                  altitude=ALTITUDE * 1000, dni_extra=dni_extra)
    dni_W, dhi_W, ghi_W = cs['dni'], cs['dhi'], cs['ghi']

    # Plane-of-array irradiance: isotropic sky diffuse + ground reflection
    poa_comp = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=AZIMUTH,
        solar_zenith=apparent_zenith, solar_azimuth=azimuth,
        dni=dni_W, ghi=ghi_W, dhi=dhi_W,
        albedo=0.2, model='isotropic')

    # Apply IAM only to the beam component (poa_direct already clips to 0
    # when the sun is behind the panel)
    beam_W = float(poa_comp['poa_direct']) * iam
    diffuse_W = float(poa_comp['poa_sky_diffuse'])
    ground_W = float(poa_comp['poa_ground_diffuse'])
    poa_total_W = beam_W + diffuse_W + ground_W
    stray_poa_W = diffuse_W + ground_W

    # Cell temperature – simple NOCT model (no direct pvlib equivalent)
    t_a = get_average_temp(day_of_year(dt), dt.hour + dt.minute / 60)
    t_c = t_a + (NOCT - 20) * poa_total_W / 800

    # PVwatts DC power with system efficiency derate then inverter AC cap
    def _pvwatts_kw(g_W):
        return min(
            float(pvlib.pvsystem.pvwatts_dc(
                g_W, t_c, MAX_POWER * 1000, BETA / 100)) / 1000 * SYSTEM_EFFICIENCY,
            INVERTER_AC_MAX)

    power = _pvwatts_kw(poa_total_W)
    if stray:
        return power, _pvwatts_kw(stray_poa_W)
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

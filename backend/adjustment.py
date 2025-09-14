import pandas as pd
import pvlib
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


# --- Reference System Configuration ---
@dataclass
class ReferenceSystem:
    """Parameters describing the reference PV system used during model training."""
    latitude: float = 35.0       # deg, location of reference system
    longitude: float = -120.0    # deg
    tz: str = "Etc/GMT+8"        # timezone
    tilt: float = 23.0           # deg
    azimuth: float = 180.0       # deg (180 = south facing in N hemisphere)
    area: float = 10.0           # m²
    efficiency: float = 0.18     # fraction (0–1)


REF_SYSTEM = ReferenceSystem()


# --- Adjustment Layer ---
def adjust_prediction(P_base: float,
                      user_area: float,
                      user_eff: float,
                      tilt: float,
                      azimuth: float,
                      timestamp: pd.Timestamp,
                      location: Dict[str, Any] = None,
                      ref_system: ReferenceSystem = REF_SYSTEM) -> float:
    """
    Adjust the base solar power prediction from the ML model to reflect a user's system.

    Parameters
    ----------
    P_base : float
        Base model prediction [W], already conditioned on irradiation and module temperature.
    user_area : float
        User system panel area [m²].
    user_eff : float
        User system efficiency (fraction, 0–1).
    tilt : float
        User panel tilt angle from horizontal [degrees].
    azimuth : float
        User panel azimuth [degrees] (0 = north, 90 = east, 180 = south in N hemisphere).
    timestamp : pd.Timestamp
        Localized timestamp of the prediction (with timezone).
    location : dict, optional
        Dictionary with 'latitude', 'longitude', 'tz' for the user's site.
        Defaults to reference system’s location if not provided.
    ref_system : ReferenceSystem
        Reference system configuration used during model training.

    Returns
    -------
    float
        Adjusted solar power prediction [W], scaled to user system.
    
    Notes
    -----
    - Temperature effects are NOT reapplied (they are already in P_base).
    - Orientation factor is computed by comparing Plane-of-Array (POA) irradiance
      between the user system and the reference system using pvlib.
    - Assumes efficiency and area scale linearly with power.
    """

    # ---------------------------
    # 1. Location setup
    # ---------------------------
    if location is None:
        latitude, longitude, tz = ref_system.latitude, ref_system.longitude, ref_system.tz
    else:
        latitude, longitude, tz = location["latitude"], location["longitude"], location["tz"]

    site = pvlib.location.Location(latitude, longitude, tz=tz)

    # ---------------------------
    # 2. Solar position
    # ---------------------------
    solpos = site.get_solarposition(timestamp)

    # ---------------------------
    # 3. Clear-sky GHI/DNI/DHI
    # ---------------------------
    # Convert single timestamp to DatetimeIndex (required by pvlib >= 0.10)
    times = pd.DatetimeIndex([timestamp])
    clearsky = site.get_clearsky(times)

    # Extract scalar values (floats) from Series
    dni = clearsky["dni"].iloc[0]
    ghi = clearsky["ghi"].iloc[0]
    dhi = clearsky["dhi"].iloc[0]

    # ---------------------------
    # 4. POA irradiance - user vs reference
    # ---------------------------
    # Pass scalar values for sun position (not Series)
    poa_user = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=solpos["apparent_zenith"].iloc[0],
        solar_azimuth=solpos["azimuth"].iloc[0],
    )["poa_global"]

    poa_ref = pvlib.irradiance.get_total_irradiance(
        surface_tilt=ref_system.tilt,
        surface_azimuth=ref_system.azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=solpos["apparent_zenith"].iloc[0],
        solar_azimuth=solpos["azimuth"].iloc[0],
    )["poa_global"]

    # ---------------------------
    # 5. Compute orientation factor as scalar (NO .fillna() or .clip() on float!)
    # ---------------------------
    # Avoid division by zero
    if poa_ref <= 1e-8:  # Near-zero or invalid irradiance
        orientation_factor = 0.0
    else:
        orientation_factor = poa_user / poa_ref
        orientation_factor = max(orientation_factor, 0.0)  # Clip to 0

    # ---------------------------
    # 6. Apply scaling factors
    # ---------------------------
    P_scaled = P_base
    P_scaled *= (user_area / ref_system.area)
    P_scaled *= (user_eff / ref_system.efficiency)
    P_scaled *= orientation_factor

    return max(P_scaled, 0.0)
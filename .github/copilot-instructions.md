# Copilot Instructions for AI Agents

## Project Overview
This project models and adjusts solar power generation predictions using Python. It includes:
- `adjustment.py`: Contains the core adjustment logic for scaling and orienting PV system predictions to user-specific parameters using `pvlib` and `pandas`.
- `solar_model.joblib`: Pre-trained model artifact (binary, not human-readable).
- `dataset/`: Contains CSVs with generation and weather sensor data for two plants.

## Key Patterns & Workflows
- **Adjustment Logic**: Use the `adjust_prediction` function in `adjustment.py` to scale model outputs for different PV system configurations. It factors in area, efficiency, tilt, azimuth, and location/timezone.
- **Reference System**: The `ReferenceSystem` dataclass in `adjustment.py` defines the baseline PV system for all adjustments.
- **Data Handling**: All time and location calculations use `pandas.Timestamp` and `pvlib.location.Location`.
- **Model Usage**: The main model is loaded from `solar_model.joblib`. (Loading code is not present in the current files, but this is the expected artifact.)

## Conventions
- All physical units are SI (e.g., area in m², angles in degrees).
- Timezone handling is explicit via the `tz` field in system/location objects.
- Orientation factors are computed using clear-sky irradiance for relative scaling.
- All scaling factors are multiplicative and applied in sequence.

## Integration Points
- **pvlib**: Used for all solar position and irradiance calculations.
- **pandas**: Used for timestamp and data manipulation.
- **Model artifact**: `solar_model.joblib` is expected to be loaded (e.g., with `joblib.load`).
- **Data**: Input data is expected in the format of the provided CSVs in `dataset/`.

## Example Usage
```python
from adjustment import adjust_prediction
# ...
P_user = adjust_prediction(P_base, user_area, user_eff, tilt, azimuth, timestamp, location)
```

## Directory Structure
- `adjustment.py`, `app.py`: Core backend logic
- `dataset/`: Input data
- `solar_model.joblib`: Model artifact

## Missing/To-Do
- No explicit build/test scripts or requirements.txt found in this directory. If present, follow those in the backend folder as per the alternate structure in `file structure.txt`.
- No frontend/backend split in this directory, but the project may be structured that way elsewhere.

---
For further conventions, check for a `requirements.txt` or additional backend/frontend folders if present in your environment.

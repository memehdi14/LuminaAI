# Lumina.AI - Solar Intelligence Platform

Lumina.AI is a full-stack solar power prediction platform that leverages a machine learning backend and a sophisticated, futuristic user interface. It provides real-time solar generation forecasts based on environmental data and user-specific panel configurations, offering valuable insights for energy optimization.

---

## Features

-   **AI-Powered Predictions**: Utilizes a pre-trained `scikit-learn` model to forecast solar power output with high accuracy.
-   **Dynamic & Interactive Dashboard**: A sleek, futuristic UI built with modern HTML, CSS, and JavaScript for a premium user experience.
-   **User-Specific Adjustments**: Power predictions are tailored to the user's unique setup, including panel area, efficiency, tilt, and azimuth.
-   **Physically-Accurate Modeling**: Incorporates the `pvlib-python` library to accurately model how panel orientation affects energy generation, moving beyond simple approximations.
-   **Real-time Visualization**: Features an interactive chart (using Chart.js) to visualize the 24-hour solar generation forecast.
-   **"What-If" Scenario Lab**: Allows users to experiment with different configurations to understand their impact on energy output.

---

## Tech Stack

A breakdown of the major technologies and libraries used in this project.

| Backend            | Frontend      |
| ------------------ | ------------- |
| Python             | HTML5         |
| Flask              | CSS3          |
| scikit-learn       | JavaScript    |
| Pandas             | Chart.js      |
| pvlib-python       | Font Awesome  |

---

## Getting Started

Follow these instructions to get a local copy of Lumina.AI up and running.

### Prerequisites

-   Python 3.8+
-   `pip` and `venv`

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/lumina-ai.git](https://github.com/your-username/lumina-ai.git)
    cd lumina-ai
    ```

2.  **Create and activate a Python virtual environment:**
    * **macOS / Linux:**
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **Windows:**
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install backend dependencies:**
    Make sure you have a `requirements.txt` file with the necessary packages.
    ```
    # requirements.txt
    flask
    pandas
    scikit-learn
    joblib
    pvlib
    ```
    Then, run:
    ```sh
    pip install -r requirements.txt
    ```

4.  **Run the backend server:**
    ```sh
    python app.py
    ```
    The Flask server will start, typically on `http://1227.0.0.1:5000`.

5.  **Launch the frontend:**
    Simply open the `futuristic_dashboard.html` file in your favorite web browser.

---

## Usage

1.  Navigate to the **"AI Prediction"** or **"What-If Lab"** section in the sidebar.
2.  Input your geographic coordinates (Latitude/Longitude for Ahmedabad: `23.0225`, `72.5714`).
3.  Adjust the sliders and input fields to match your solar panel configuration (Tilt, Azimuth, Area, Efficiency).
4.  Click the **"Activate Neural Core"** button.
5.  The dashboard will update with the predicted power generation, daily total, peak power, and a 24-hour forecast chart based on your inputs.

---

## API Endpoint

The backend provides a single endpoint for predictions.

-   **URL:** `/predict`
-   **Method:** `POST`
-   **Body (JSON):** The endpoint expects a JSON payload with `weather_data` for the base model and `user_config` for adjustments.

    **Example Request:**
    ```json
    {
        "weather_data": {
            "IRRADIATION": 0.85,
            "AMBIENT_TEMPERATURE": 32.5,
            "hour": 13,
            "day_of_year": 257
        },
        "user_config": {
            "latitude": 23.02,
            "longitude": 72.57,
            "area_m2": 15.0,
            "efficiency": 0.21,
            "tilt": 25,
            "azimuth": 180
        }
    }
    ```

---

## Project Structure

```

lumina-ai/
├── models/
│   ├── solar\_model.joblib
│   └── feature\_names.joblib
├── venv/
├── app.py
├── futuristic\_dashboard.html
├── requirements.txt
└── README.md
```


## Future Improvements

-   Integrate a live weather API (e.g., OpenWeatherMap) to fetch real-time data instead of using static inputs.
-   Implement a database (like SQLite or PostgreSQL) to store historical prediction data for trend analysis.
-   Add user authentication to save different panel configurations.
-   Expand the "Analytics Core" to generate and download PDF reports.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

-   A project inspired by the need for accessible and accurate renewable energy forecasting.
-   Special thanks to the developers of `pvlib-python` for their incredible solar modeling library.
-   This project is prepared by Team VectorX for their submission in LJ HACKOVATE-2025
```

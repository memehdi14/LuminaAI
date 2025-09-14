import os
import requests
import joblib
import pandas as pd
import ollama
import sqlite3
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, render_template, make_response, Response
from flask_cors import CORS
from adjustment import adjust_prediction, ReferenceSystem
import uuid
from werkzeug.utils import secure_filename
import csv
import io
import numpy as np
import pdfkit  # NEW: Simple, reliable PDF generator for Windows

# --- 1. ENV + APP SETUP ---
load_dotenv()
app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app, origins=["*"], methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type"])

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('exports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- 2. LOAD MODEL & FEATURE NAMES ---
try:
    model = joblib.load("models/solar_model.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    metadata = joblib.load("models/model_metadata.joblib")
    BASELINE_ACCURACY = round(metadata['r2'] * 100, 1)
    print(f"✅ Loaded LightGBM model and {len(feature_names)} features from models/")
    print(f"📊 Baseline Model Accuracy (R²): {BASELINE_ACCURACY}%")
except FileNotFoundError:
    print("❌ ERROR: Missing model files! Please run train_model.py to generate:")
    print("   - models/solar_model.joblib")
    print("   - models/feature_names.joblib")
    print("   - models/model_metadata.joblib")
    exit(1)

# 🌍 WEATHER API KEY
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")
if not WEATHERAPI_KEY:
    print("❌ ERROR: WEATHERAPI_KEY is required in .env file.")
    exit(1)

DEFAULT_TZ = "Asia/Kolkata"

# --- 3. DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('solar_predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            latitude REAL,
            longitude REAL,
            timestamp DATETIME,
            prediction_data TEXT,
            ai_insight TEXT,
            user_params TEXT,
            accuracy REAL DEFAULT 85.0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id TEXT PRIMARY KEY,
            filename TEXT,
            upload_date DATETIME,
            file_type TEXT,
            status TEXT DEFAULT 'processed'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY,
            price REAL DEFAULT 7.0,
            area REAL DEFAULT 10.0,
            efficiency REAL DEFAULT 0.9,
            tilt INTEGER DEFAULT 20
        )
    ''')
    
    cursor.execute('SELECT COUNT(*) FROM user_settings')
    if cursor.fetchone()[0] == 0:
        cursor.execute('INSERT INTO user_settings (price, area, efficiency, tilt) VALUES (7.0, 10.0, 0.9, 20)')
    
    conn.commit()
    conn.close()

init_db()

# --- 4. USER SETTINGS ---
@app.route("/settings", methods=["GET", "POST"])
def settings():
    conn = sqlite3.connect('solar_predictions.db')
    cursor = conn.cursor()
    
    if request.method == "POST":
        data = request.get_json()
        cursor.execute('''UPDATE user_settings SET price = ?, area = ?, efficiency = ?, tilt = ? WHERE id = 1''',
                       (data.get('price', 7.0), data.get('area', 10.0), data.get('efficiency', 0.9), data.get('tilt', 20)))
        conn.commit()
        
        cursor.execute('SELECT * FROM user_settings WHERE id = 1')
        row = cursor.fetchone()
        conn.close()
        
        return jsonify({
            "status": "success", 
            "settings": {
                "price": row[1], "area": row[2], 
                "efficiency": row[3], "tilt": row[4]
            }
        })
    else:
        cursor.execute('SELECT * FROM user_settings WHERE id = 1')
        row = cursor.fetchone()
        conn.close()
        
        return jsonify({
            "price": row[1], "area": row[2], 
            "efficiency": row[3], "tilt": row[4]
        })

# --- 5. WEATHER API ---
def get_weather_forecast(lat, lon, api_key):
    try:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={lat},{lon}&days=1&aqi=no&alerts=no"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        hourly_forecasts = data['forecast']['forecastday'][0]['hour']
        cleaned = []
        for hour_data in hourly_forecasts[:8]:
            cleaned.append({
                "dt": hour_data['time_epoch'],
                "temp": round(hour_data['temp_c'], 1),
                "clouds": hour_data['cloud']
            })
        return cleaned
    except Exception as e:
        print(f"⚠ WeatherAPI.com failed: {str(e)}")
        return get_simulated_weather(lat, lon)

def get_simulated_weather(lat, lon):
    now = datetime.now()
    day_of_year = now.timetuple().tm_yday
    forecasts = []
    for i in range(8):
        hour_offset = i * 3
        dt = now + timedelta(hours=hour_offset)
        dt_timestamp = int(dt.timestamp())
        
        base_temp = 30 + 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temp = base_temp + 8 * np.sin(2 * np.pi * (hour_offset - 12) / 24)
        
        if 6 <= hour_offset <= 18:
            clouds = 40 + np.random.randint(0, 50)
        else:
            clouds = np.random.randint(10, 30)
        
        clouds = min(100, max(0, clouds))
        
        forecasts.append({
            "dt": dt_timestamp,
            "temp": round(temp, 1),
            "clouds": clouds
        })
    
    return forecasts

def generate_sandbox_weather(override_params):
    """
    Generates synthetic 24-hour (8-point) weather data based on preset or custom conditions.
    Returns list of dicts: [{"dt": epoch, "temp": float, "clouds": int}, ...]
    """
    now = datetime.now()
    day_of_year = now.timetuple().tm_yday
    base_temp = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    condition = override_params.get("condition", "custom")
    
    if condition == "sunny":
        cloud_profile = [5, 10, 5, 0, 5, 10, 15, 20]
        temp_mod = 5
    elif condition == "partly_cloudy":
        cloud_profile = [40, 55, 70, 60, 50, 65, 75, 80]
        temp_mod = 0
    elif condition == "overcast":
        cloud_profile = [90, 95, 100, 100, 95, 90, 85, 80]
        temp_mod = -5
    else:  # Custom
        cloud_profile = [int(override_params.get("cloud_cover", 30))] * 8
        base_temp = int(override_params.get("temperature", 30))
        temp_mod = 0

    forecasts = []
    for hour_offset in range(0, 24, 3):
        dt = now.replace(hour=hour_offset, minute=0, second=0, microsecond=0)
        temp = (base_temp + temp_mod) - 8 * np.cos(2 * np.pi * hour_offset / 24)
        forecasts.append({
            "dt": int(dt.timestamp()),
            "temp": round(temp, 1),
            "clouds": min(100, max(0, cloud_profile[hour_offset // 3]))
        })
    return forecasts

# --- 6. CORE PREDICTION ---
def make_prediction(live_weather_data):
    df = pd.DataFrame(live_weather_data)
    df["timestamp"] = pd.to_datetime(df["dt"], unit="s").dt.tz_localize("UTC").dt.tz_convert(DEFAULT_TZ)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["AMBIENT_TEMPERATURE"] = df["temp"]
    df["MODULE_TEMPERATURE"] = df["temp"] + 12
    df["IRRADIATION"] = (100 - df["clouds"]) / 100
    df["TEMP_DIFF"] = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]
    df["is_daylight"] = [1 if 6 < h < 18 else 0 for h in df["hour"]]

    features = df[feature_names].values
    prediction = model.predict(features)
    prediction[prediction < 0] = 0
    return prediction, df["timestamp"].tolist(), df["clouds"].tolist()

# --- 7. AI INSIGHTS ---
def get_ai_insights(final_prediction_watts, price_per_kwh, location_key, detailed_insights):
    try:
        import ollama
    except ImportError:
        print("❌ Ollama Python package not installed.")
        return f"Peak generation expected around {12}:00 with {(max(final_prediction_watts)/1000 if final_prediction_watts else 0):.1f}kW. Run high-power appliances like AC or washing machine during peak. Clean panels weekly for optimal performance."

    total_kwh = sum(final_prediction_watts) / 1000
    total_savings = total_kwh * price_per_kwh
    peak_power = max(final_prediction_watts) if final_prediction_watts else 0
    peak_idx = final_prediction_watts.index(peak_power) if peak_power > 0 else -1
    peak_hour = (datetime.now().hour + peak_idx * 3) % 24 if peak_idx >= 0 else None

    prompt = f"""
As Lumina.AI, an expert solar energy analyst, provide a concise, professional analysis based on the following 24-hour forecast data for a user in {location_key}:

- **Total Predicted Generation:** {total_kwh:.2f} kWh
- **Peak Power Output:** {peak_power:.0f} Watts, occurring around {peak_hour or 'mid-day'}:00
- **Estimated Financial Savings:** ₹{total_savings:.2f} for the day
- **Weather Conditions:** {detailed_insights.get('weatherSummary', 'Not available')}
- **System Efficiency Rating:** {detailed_insights.get('efficiencyRating', 'Fair')}

Generate a response in three distinct parts:
1.  **Key Takeaway (Bold):** Start with a single, impactful summary sentence.
2.  **Actionable Advice:** Provide 2-3 specific, timed recommendations. Mention the peak generation window (e.g., "Between 12:00 and 15:00...") and suggest which high-power appliances (EV chargers, water heaters, AC units) to use during that time.
3.  **Optimization Tip:** Offer one practical tip related to the weather or system efficiency (e.g., "Given the partly cloudy forecast, consider cleaning your panels to maximize absorption of diffuse sunlight.").

Do not use markdown lists or bullet points.
"""
    try:
        response = ollama.generate(model='gemma:2b', prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        return "AI insights unavailable due to service error."

# --- 8. PREDICTION ENDPOINT — FULLY UPGRADED FOR SANDBOX + SCENARIOS ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_data = request.get_json()
        
        # Fetch current settings from DB
        conn = sqlite3.connect('solar_predictions.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user_settings WHERE id = 1')
        settings = cursor.fetchone()
        conn.close()
        
        # Use user input OR fallback to settings
        area = user_data.get("area", settings[2])
        eff = user_data.get("efficiency", settings[3])
        tilt = user_data.get("tilt", settings[4])
        price = user_data.get("price", settings[1])

        lat = user_data.get("latitude")
        lon = user_data.get("longitude")
        
        if lat is None or lon is None:
            return jsonify({"error": "Latitude and longitude are required"}), 400

        # 🔥 NEW: Check for sandbox weather override
        if "weather_override" in user_data:
            weather = generate_sandbox_weather(user_data["weather_override"])
        else:
            weather = get_weather_forecast(lat, lon, WEATHERAPI_KEY)

        base_pred, timestamps, clouds = make_prediction(weather)

        final_pred = []
        for i, p_base in enumerate(base_pred):
            adjusted = adjust_prediction(
                P_base=p_base,
                user_area=area,
                user_eff=eff,
                tilt=tilt,
                azimuth=180,
                timestamp=timestamps[i],
                location={"latitude": lat, "longitude": lon, "tz": DEFAULT_TZ}
            )
            final_pred.append(max(0, adjusted))

        location_key = f"{lat}_{lon}"
        detailed_insights = calculate_detailed_insights(final_pred, clouds)
        ai_summary = get_ai_insights(final_pred, price, location_key, detailed_insights)
        
        avg_irradiance = sum((100 - c)/100 for c in clouds) / len(clouds) if clouds else 0.5
        accuracy = max(70.0, min(BASELINE_ACCURACY + (avg_irradiance - 0.5) * 20, 98.0))

        prediction_id = str(uuid.uuid4())
        conn = sqlite3.connect('solar_predictions.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO predictions (id, latitude, longitude, timestamp, prediction_data, ai_insight, user_params, accuracy)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                       (prediction_id, lat, lon, datetime.now(), json.dumps(final_pred), ai_summary, json.dumps(user_data), accuracy))
        conn.commit()
        conn.close()

        peak_power = max(final_pred) if final_pred else 0
        peak_idx = final_pred.index(peak_power) if peak_power > 0 else -1

        return jsonify({
            "prediction": [round(p, 2) for p in final_pred],
            "hours": [ts.hour for ts in timestamps],
            "aiInsight": ai_summary,
            "detailedInsights": detailed_insights,
            "accuracy": round(accuracy, 1),
            "peakPower": peak_power,
            "peakHourIndex": peak_idx,
            "latitude": lat,
            "longitude": lon
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

def calculate_detailed_insights(prediction_watts, cloud_data):
    insights = {}
    
    productive_hours = sum(1 for w in prediction_watts if w > 100) * 3
    insights["sunlightHours"] = f"{productive_hours} hours"
    insights["productiveHours"] = productive_hours
    
    morning_watts = sum(prediction_watts[:4]) if len(prediction_watts) >= 4 else 0
    afternoon_watts = sum(prediction_watts[4:]) if len(prediction_watts) > 4 else sum(prediction_watts)
    total = morning_watts + afternoon_watts
    insights["morningGenerationPercent"] = int((morning_watts / total) * 100) if total > 0 else 0
    
    peak = max(prediction_watts) if prediction_watts else 0
    if peak > 5000:
        insights["applianceEquivalent"] = "Central AC + Water Heater"
        insights["applianceCount"] = "3-4 major appliances"
    elif peak > 3000:
        insights["applianceEquivalent"] = "Room AC + Washing Machine"
        insights["applianceCount"] = "2-3 appliances"
    elif peak > 1500:
        insights["applianceEquivalent"] = "Washing Machine + Refrigerator"
        insights["applianceCount"] = "2 appliances"
    else:
        insights["applianceEquivalent"] = "LED Lights + Basic Electronics"
        insights["applianceCount"] = "Basic appliances only"
    
    avg_clouds = sum(cloud_data) / len(cloud_data) if cloud_data else 0
    if avg_clouds < 25:
        insights["weatherSummary"] = "Excellent: Clear Skies"
        insights["weatherRating"] = "excellent"
    elif avg_clouds < 60:
        insights["weatherSummary"] = "Good: Partly Cloudy"
        insights["weatherRating"] = "good"
    else:
        insights["weatherSummary"] = "Fair: Overcast Conditions"
        insights["weatherRating"] = "fair"
    
    total_generation = sum(prediction_watts)
    max_theoretical = len(prediction_watts) * 1000
    efficiency_ratio = (total_generation / max_theoretical) * 100 if max_theoretical > 0 else 0
    
    if efficiency_ratio > 70:
        insights["efficiencyRating"] = "Excellent"
        insights["efficiencyColor"] = "#10b981"
    elif efficiency_ratio > 50:
        insights["efficiencyRating"] = "Good"
        insights["efficiencyColor"] = "#fbbf24"
    else:
        insights["efficiencyRating"] = "Fair"
        insights["efficiencyColor"] = "#f59e0b"
    
    insights["efficiencyPercent"] = round(efficiency_ratio, 1)
    
    return insights

# --- 9. HISTORY ---
@app.route("/history", methods=["GET"])
def get_history():
    conn = sqlite3.connect('solar_predictions.db')
    cursor = conn.cursor()
    
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    offset = (page - 1) * per_page
    
    cursor.execute('''SELECT id, latitude, longitude, timestamp, prediction_data, ai_insight, accuracy
                      FROM predictions ORDER BY timestamp DESC LIMIT ? OFFSET ?''', (per_page, offset))
    
    predictions = []
    for row in cursor.fetchall():
        pred_data = json.loads(row[4])
        total_generation = sum(pred_data) / 1000
        peak_power = max(pred_data) / 1000
        
        predictions.append({
            "id": row[0],
            "latitude": row[1],
            "longitude": row[2],
            "timestamp": row[3],
            "totalGeneration": round(total_generation, 2),
            "peakPower": round(peak_power, 2),
            "aiInsight": row[5],
            "accuracy": round(row[6], 1),
            "location": f"{row[1]:.3f}, {row[2]:.3f}"
        })
    
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total_count = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        "predictions": predictions,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total_count,
            "pages": (total_count + per_page - 1) // per_page
        }
    })

# --- 10. FILE UPLOAD ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        file_ext = filename.rsplit('.', 1)[1].lower()
        saved_filename = f"{file_id}.{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        
        file.save(file_path)
        
        try:
            processed_data = process_uploaded_file(file_path, file_ext)
            
            conn = sqlite3.connect('solar_predictions.db')
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO uploaded_files (id, filename, upload_date, file_type, status)
                              VALUES (?, ?, ?, ?, ?)''', (file_id, filename, datetime.now(), file_ext, 'processed'))
            conn.commit()
            conn.close()
            
            return jsonify({
                "success": True,
                "fileId": file_id,
                "filename": filename,
                "processedData": processed_data
            })
            
        except Exception as e:
            return jsonify({"error": f"File processing failed: {str(e)}"}), 400
    
    return jsonify({"error": "Invalid file type"}), 400

def process_uploaded_file(file_path, file_type):
    if file_type == 'csv':
        try:
            df = pd.read_csv(file_path)
            analysis = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "sample_data": df.head(3).to_dict('records') if len(df) > 0 else [],
                "summary": {
                    "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                    "date_columns": df.select_dtypes(include=['datetime']).columns.tolist(),
                    "missing_values": df.isnull().sum().to_dict()
                }
            }
            return analysis
        except Exception as e:
            raise Exception(f"CSV processing error: {str(e)}")
    
    elif file_type in ['txt']:
        with open(file_path, 'r') as f:
            content = f.read()
        return {
            "type": "text",
            "size": len(content),
            "preview": content[:500] + "..." if len(content) > 500 else content
        }
    
    else:
        raise Exception(f"Unsupported file type: {file_type}")

@app.route("/files", methods=["GET"])
def get_files():
    conn = sqlite3.connect('solar_predictions.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM uploaded_files ORDER BY upload_date DESC')
    
    files = []
    for row in cursor.fetchall():
        files.append({
            "id": row[0],
            "filename": row[1],
            "uploadDate": row[2],
            "fileType": row[3],
            "status": row[4]
        })
    
    conn.close()
    return jsonify({"files": files})

# --- 11. REPORTS ---
@app.route("/reports/summary", methods=["GET"])
def get_reports_summary():
    conn = sqlite3.connect('solar_predictions.db')
    cursor = conn.cursor()
    
    days = int(request.args.get('days', 30))
    start_date = datetime.now() - timedelta(days=days)
    
    cursor.execute('''SELECT prediction_data, timestamp, latitude, longitude, accuracy 
                      FROM predictions WHERE timestamp >= ? ORDER BY timestamp DESC''', (start_date,))
    
    predictions = cursor.fetchall()
    
    if not predictions:
        return jsonify({
            "summary": {
                "totalPredictions": 0,
                "avgDailyGeneration": 0,
                "totalEstimatedGeneration": 0,
                "avgAccuracy": round(BASELINE_ACCURACY, 1),
                "topLocation": "No data"
            },
            "dailyTrends": [],
            "locationStats": [],
            "predictions": []
        })
    
    total_predictions = len(predictions)
    all_generation = []
    location_stats = {}
    daily_data = {}
    
    for pred in predictions:
        pred_data = json.loads(pred[0])
        total_gen = sum(pred_data) / 1000
        all_generation.append(total_gen)
        
        loc_key = f"{pred[2]:.2f},{pred[3]:.2f}"
        if loc_key not in location_stats:
            location_stats[loc_key] = {"count": 0, "total_gen": 0}
        location_stats[loc_key]["count"] += 1
        location_stats[loc_key]["total_gen"] += total_gen
        
        day_key = pred[1][:10]
        if day_key not in daily_data:
            daily_data[day_key] = []
        daily_data[day_key].append(total_gen)
    
    avg_daily_generation = sum(all_generation) / len(all_generation)
    total_estimated = sum(all_generation)
    
    top_location = max(location_stats.items(), key=lambda x: x[1]["count"])[0] if location_stats else "Unknown"
    
    daily_trends = []
    for day, generations in sorted(daily_data.items()):
        daily_trends.append({
            "date": day,
            "avgGeneration": sum(generations) / len(generations),
            "predictions": len(generations)
        })
    
    location_list = []
    for loc, stats in sorted(location_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:5]:
        location_list.append({
            "location": loc,
            "predictions": stats["count"],
            "avgGeneration": stats["total_gen"] / stats["count"]
        })
    
    conn.close()
    
    return jsonify({
        "summary": {
            "totalPredictions": total_predictions,
            "avgDailyGeneration": round(avg_daily_generation, 2),
            "totalEstimatedGeneration": round(total_estimated, 2),
            "avgAccuracy": round(BASELINE_ACCURACY, 1),
            "topLocation": top_location
        },
        "dailyTrends": daily_trends[-14:],
        "locationStats": location_list,
        "predictions": predictions
    })

# --- 12. EXPORTS — CSV ONLY (SIMPLE) ---
@app.route("/export/<export_type>", methods=["GET"])
def export_data(export_type):
    conn = sqlite3.connect('solar_predictions.db')
    cursor = conn.cursor()
    
    if export_type == "predictions":
        cursor.execute('''SELECT id, latitude, longitude, timestamp, prediction_data, ai_insight, accuracy
                          FROM predictions ORDER BY timestamp DESC''')
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Latitude', 'Longitude', 'Timestamp', 'Total_Generation_kWh', 'Peak_Power_kW', 'AI_Insight', 'Accuracy'])
        
        for row in cursor.fetchall():
            pred_data = json.loads(row[4])
            total_gen = sum(pred_data) / 1000
            peak_power = max(pred_data) / 1000
            writer.writerow([row[0], row[1], row[2], row[3], round(total_gen, 2), round(peak_power, 2), row[5], round(row[6], 1)])
        
        output.seek(0)
        
        return jsonify({
            "success": True,
            "data": output.getvalue(),
            "filename": f"luminaai_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        })
    
    elif export_type == "summary" or export_type == "report":
        cursor.execute('''SELECT prediction_data, timestamp, latitude, longitude, ai_insight, accuracy, user_params
                          FROM predictions ORDER BY timestamp DESC''')
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            'Date', 'Location', 'Total Generation (kWh)', 'Peak Power (kW)', 
            'Estimated Savings (₹)', 'AI Insight', 'Accuracy (%)', 'Area (m²)', 'Efficiency', 'Tilt Angle'
        ])
        
        for row in cursor.fetchall():
            pred_data = json.loads(row[0])
            total_gen = sum(pred_data) / 1000
            peak_power = max(pred_data) / 1000
            savings = total_gen * 7.0
            user_params = json.loads(row[6]) if row[6] else {}
            area = user_params.get('area', 10.0)
            efficiency = user_params.get('efficiency', 0.9)
            tilt = user_params.get('tilt', 20)
            
            writer.writerow([
                row[1][:10],
                f"{row[2]:.3f}, {row[3]:.3f}",
                round(total_gen, 2),
                round(peak_power, 2),
                round(savings, 2),
                row[5],
                round(row[6], 1),
                area,
                efficiency,
                tilt
            ])
        
        output.seek(0)
        
        return jsonify({
            "success": True,
            "data": output.getvalue(),
            "filename": f"luminaai_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        })
    
    conn.close()
    return jsonify({"error": "Invalid export type"}), 400

# --- 13. PDF EXPORT — SIMPLE, WINDOWS-FRIENDLY USING PDFKIT ---
@app.route("/export/report.pdf")
def export_report_pdf():
    conn = sqlite3.connect('solar_predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''SELECT prediction_data, timestamp, latitude, longitude, ai_insight, accuracy, user_params
                      FROM predictions ORDER BY timestamp DESC LIMIT 10''')
    predictions = cursor.fetchall()
    
    recent_data = []
    for row in predictions:
        pred_data = json.loads(row[0])
        total_gen = sum(pred_data) / 1000
        peak_power = max(pred_data) / 1000
        savings = total_gen * 7.0
        user_params = json.loads(row[6]) if row[6] else {}
        area = user_params.get('area', 10.0)
        efficiency = user_params.get('efficiency', 0.9)
        tilt = user_params.get('tilt', 20)
        
        recent_data.append({
            "date": row[1][:10],
            "location": f"{row[2]:.3f}, {row[3]:.3f}",
            "generation": round(total_gen, 2),
            "peak": round(peak_power, 2),
            "savings": round(savings, 2),
            "accuracy": round(row[5], 1),
            "ai_insight": row[5],
            "area": area,
            "efficiency": efficiency,
            "tilt": tilt
        })
    
    # HTML Template as string
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Lumina.ai Solar Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
            h1 {{ color: #fbbf24; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ccc; padding: 10px; text-align: left; }}
            th {{ background-color: #f1f5f9; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            .footer {{ margin-top: 40px; text-align: center; color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Lumina.ai Solar Generation Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <h2>Recent Predictions</h2>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Location</th>
                    <th>Total Gen (kWh)</th>
                    <th>Peak Power (kW)</th>
                    <th>Accuracy (%)</th>
                    <th>Savings (₹)</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for item in recent_data:
        html_content += f"""
                <tr>
                    <td>{item['date']}</td>
                    <td>{item['location']}</td>
                    <td>{item['generation']}</td>
                    <td>{item['peak']}</td>
                    <td>{item['accuracy']}</td>
                    <td>₹{item['savings']}</td>
                </tr>
        """
    
    html_content += """
            </tbody>
        </table>
        
        <div class="footer">
            © 2025 Lumina.ai — Smarter Solar Insights
        </div>
    </body>
    </html>
    """

    # Configure pdfkit to use wkhtmltopdf executable path
    config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')

    # Generate PDF
    pdf = pdfkit.from_string(html_content, False, configuration=config)

    # Return PDF as download
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=lumina_ai_report.pdf'
    return response

# --- 14. HEALTH CHECK ---
@app.route("/health", methods=["GET"])
def health_check():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = response.json().get("models", [])
        gemma_available = any(m["name"] == "gemma:2b" for m in models)
        
        if gemma_available and WEATHERAPI_KEY:
            return jsonify({"status": "online", "message": "Backend ready"}), 200
        else:
            return jsonify({"status": "offline", "message": "AI/Weather service down"}), 503
            
    except Exception:
        return jsonify({"status": "offline", "message": "Connection error"}), 503

# --- 15. SERVE FRONTEND ---
@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)

# --- 16. START SERVER ---
if __name__ == "__main__":
    print("\n🚀 Lumina.ai Enhanced Backend Starting (WEATHERAPI.COM + GEMMA:2B)...")
    print(f"📊 Model Baseline Accuracy: {BASELINE_ACCURACY}%")
    print(f"🌐 WeatherAPI.com Key: {'✅ Present' if WEATHERAPI_KEY else '❌ Missing (Set in .env)'}")
    
    import subprocess
    result = subprocess.run("ollama list | grep gemma:2b", shell=True, capture_output=True, text=True)
    if result.returncode == 0 and "gemma:2b" in result.stdout:
        print("🧠 Ollama gemma:2b: ✅ Pulled and ready")
    else:
        print("🧠 Ollama gemma:2b: ❌ Not installed. Run: ollama pull gemma:2b")
    
    print("📡 Weather API: WeatherAPI.com (Free Tier — No Limits, Works in India)")
    print("Features: Predictions, History, Uploads, Reports, What-If Analysis, PDF Export, Sandbox")
    print("Server running on http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
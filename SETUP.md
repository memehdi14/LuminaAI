# Lumina.AI Setup Guide

## Quick Start

1. **Start the application:**
   ```bash
   ./start_lumina.sh
   ```

2. **Access the dashboard:**
   - Frontend: http://localhost:8080
   - Backend API: http://localhost:5000

## Manual Setup

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- pip package manager

### Installation

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   - Edit `backend/.env` file
   - Add your WeatherAPI key (optional, will use simulated data if not provided)

4. **Start backend:**
   ```bash
   cd backend
   python app.py
   ```

5. **Start frontend (in another terminal):**
   ```bash
   cd frontend
   python3 -m http.server 8080
   ```

## Features Working

✅ **Core Functionality:**
- Solar power predictions with ML model
- Real-time weather data integration
- Interactive dashboard with charts
- Settings management
- File upload and processing
- Historical data tracking
- Export functionality (CSV)
- What-if scenario analysis

⚠️ **Optional Features:**
- AI insights (requires Ollama installation)
- PDF export (requires wkhtmltopdf)

## API Endpoints

- `POST /predict` - Generate solar predictions
- `GET /settings` - Get user settings
- `POST /settings` - Update user settings
- `GET /history` - Get prediction history
- `POST /upload` - Upload data files
- `GET /files` - List uploaded files
- `GET /reports/summary` - Get analytics summary
- `GET /export/{type}` - Export data (CSV)
- `GET /health` - Health check

## Troubleshooting

### Backend Issues
- Check if port 5000 is available
- Verify all dependencies are installed
- Check backend/.env file for API keys

### Frontend Issues
- Ensure backend is running on port 5000
- Check browser console for errors
- Verify frontend server is running on port 8080

### Model Issues
- The app includes pre-trained models
- If models fail to load, a fallback model will be used
- Check backend logs for model loading warnings

## Default Settings

- **Location:** Ahmedabad, India (23.0225, 72.5714)
- **Panel Area:** 10 m²
- **Efficiency:** 90%
- **Tilt Angle:** 20°
- **Electricity Price:** ₹7.0/kWh

## Support

For issues or questions, check the console logs for detailed error messages.
#!/bin/bash

# Lumina.AI Startup Script
echo "🚀 Starting Lumina.AI Solar Prediction Platform..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start backend in background
echo "📡 Starting backend server..."
cd backend
python app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend server in background
echo "🌐 Starting frontend server..."
cd ../frontend
python3 -m http.server 8080 &
FRONTEND_PID=$!

echo ""
echo "✅ Lumina.AI is now running!"
echo "📊 Backend: http://localhost:5000"
echo "🌐 Frontend: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait
#!/bin/bash

echo "Starting Medical Report Assistant..."

# Start Backend in background
echo "Starting Backend (FastAPI)..."
cd backend
# Check if venv exists, if not just run directly or warn
# Assuming user has python installed.
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Start Frontend
echo "Starting Frontend (React/Vite)..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "Services started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Access Frontend at: http://localhost:5173"
echo "Access Backend docs at: http://localhost:8000/docs"
echo "Press CTRL+C to stop both."

# Trap SIGINT to kill both processes
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT

wait

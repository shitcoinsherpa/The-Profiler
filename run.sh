#!/bin/bash
# Behavioral Profiling System - Run Script
# Activates virtual environment and runs the Gradio app

echo "======================================================================"
echo "BEHAVIORAL PROFILING SYSTEM"
echo "======================================================================"
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo
    echo "Please run ./setup.sh first to create the environment and install dependencies."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi
echo

# Add current directory to PATH for ffmpeg (if local)
export PATH="$(pwd):$PATH"

# Run the application
echo "Starting Gradio application..."
echo
echo "======================================================================"
echo "The application URL will be displayed below."
echo "Press Ctrl+C to stop the server when done."
echo "======================================================================"
echo

python app.py

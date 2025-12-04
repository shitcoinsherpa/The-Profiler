#!/bin/bash
# Behavioral Profiling System - Setup Script
# Creates virtual environment and installs dependencies

echo "======================================================================"
echo "BEHAVIORAL PROFILING SYSTEM - SETUP"
echo "======================================================================"
echo

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo
    echo "Install Python 3.10+ from your package manager or https://www.python.org/downloads/"
    exit 1
fi

echo "[1/5] Checking Python version..."
python3 --version
echo

# Create virtual environment
echo "[2/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
    echo "Virtual environment created successfully."
fi
echo

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi
echo "Virtual environment activated."
echo

# Upgrade pip
echo "[4/5] Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo "Pip upgraded successfully."
echo

# Install dependencies
echo "[5/5] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo "Dependencies installed successfully."
echo

echo "======================================================================"
echo "SETUP COMPLETE!"
echo "======================================================================"
echo
echo "Next steps:"
echo "1. Run: ./run.sh"
echo "2. Configure your API key in the Settings panel (in the web UI)"
echo "3. Get your API key from: https://openrouter.ai/keys"
echo

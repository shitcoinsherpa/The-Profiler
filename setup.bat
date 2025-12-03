@echo off
REM FBI Behavioral Profiling System - Setup Script
REM This script creates a virtual environment and installs dependencies

echo ======================================================================
echo FBI BEHAVIORAL PROFILING SYSTEM - SETUP
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.11 or higher from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo Pip upgraded successfully.
echo.

REM Install dependencies
echo [5/5] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

echo ======================================================================
echo SETUP COMPLETE!
echo ======================================================================
echo.
echo Next steps:
echo 1. Run: run.bat
echo 2. Configure your API key in the Settings panel (in the web UI)
echo 3. Get your API key from: https://openrouter.ai/keys
echo.
pause

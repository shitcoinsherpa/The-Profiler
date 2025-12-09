@echo off
REM FBI Behavioral Profiling System - Setup Script
REM This script creates a virtual environment, installs dependencies, and downloads ffmpeg

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

echo [1/6] Checking Python version...
python --version
echo.

REM Download FFmpeg if not present
echo [2/6] Checking for FFmpeg...
if exist ffmpeg.exe (
    echo FFmpeg already present. Skipping download.
) else (
    echo Downloading FFmpeg from gyan.dev...
    echo This may take a minute...
    
    REM Use PowerShell to download and extract ffmpeg
    powershell -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; $url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'; $zip = 'ffmpeg-temp.zip'; Write-Host 'Downloading...'; Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing; Write-Host 'Extracting...'; Expand-Archive -Path $zip -DestinationPath 'ffmpeg-temp' -Force; $exe = Get-ChildItem -Path 'ffmpeg-temp' -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1; if ($exe) { Copy-Item $exe.FullName -Destination 'ffmpeg.exe'; Write-Host 'Done.' } else { Write-Host 'ERROR: ffmpeg.exe not found'; exit 1 }; Remove-Item $zip -Force -ErrorAction SilentlyContinue; Remove-Item 'ffmpeg-temp' -Recurse -Force -ErrorAction SilentlyContinue"
    
    if not exist ffmpeg.exe (
        echo ERROR: Failed to download or extract FFmpeg
        echo Please download manually from: https://www.gyan.dev/ffmpeg/builds/
        echo Download ffmpeg-release-essentials.zip, extract, and copy ffmpeg.exe here.
        pause
        exit /b 1
    )
    echo FFmpeg downloaded and installed successfully.
)
echo.

REM Create virtual environment
echo [3/6] Creating virtual environment...
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
echo [4/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

REM Upgrade pip
echo [5/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo Pip upgraded successfully.
echo.

REM Install dependencies
echo [6/6] Installing dependencies...
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

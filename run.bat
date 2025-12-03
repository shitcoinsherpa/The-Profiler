@echo off
REM FBI Behavioral Profiling System - Run Script
REM This script activates the virtual environment and runs the Gradio app

echo ======================================================================
echo FBI BEHAVIORAL PROFILING SYSTEM
echo ======================================================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo.
    echo Please run setup.bat first to create the environment and install dependencies.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo.

REM Add ffmpeg to PATH for this session
set PATH=%CD%;%PATH%

REM Run the application
echo Starting Gradio application...
echo.
echo ======================================================================
echo The application URL will be displayed below.
echo Press Ctrl+C to stop the server when done.
echo ======================================================================
echo.

python app.py

if errorlevel 1 (
    echo.
    echo ======================================================================
    echo ERROR: Application failed to start
    echo ======================================================================
    echo.
    pause
    exit /b 1
)

REM Deactivate on exit
deactivate
pause

@echo off
setlocal ENABLEDELAYEDEXPANSION

cls
echo =============================
echo  Hand Gesture Mouse Control
echo =============================
echo.

REM Detect Python
where python >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Setup venv
set VENV_DIR=.venv
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate venv
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Upgrade pip and install requirements
echo Installing dependencies...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if exist requirements.txt (
    "%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
) else (
    echo requirements.txt not found. Installing core packages...
    "%VENV_DIR%\Scripts\python.exe" -m pip install opencv-python mediapipe pyautogui numpy pygetwindow
)

REM Quick diagnostics
echo Running dependency diagnostics...
"%VENV_DIR%\Scripts\python.exe" -c "import cv2, mediapipe, pyautogui, numpy, pygetwindow; print('Dependencies OK')"
if errorlevel 1 (
    echo Some dependencies failed to import. Attempting reinstall...
    "%VENV_DIR%\Scripts\python.exe" -m pip install --force-reinstall -r requirements.txt
)

REM Run loop: keep program running; restart on exit. Ctrl+C to stop.
echo.
echo Launching gesture control (auto-restart enabled). Press Ctrl+C to stop.
echo.
:RUN_LOOP
"%VENV_DIR%\Scripts\python.exe" gesture_control_main.py
if errorlevel 1 (
    echo Program exited with error. Restarting in 2 seconds...
) else (
    echo Program exited. Restarting in 2 seconds...
)
timeout /t 2 /nobreak >nul
goto RUN_LOOP

endlocal

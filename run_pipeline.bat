@echo off
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Please follow README to setup.
    pause
    exit /b 1
)
python execute_pipeline.py
pause

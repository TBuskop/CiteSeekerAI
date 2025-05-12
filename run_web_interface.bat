@echo off
echo Starting CiteSeekerAI Web Interface...
echo.

REM Activate the conda environment
call C:\Users\buskop\AppData\Local\miniconda3\Scripts\activate.bat academic_lit_search

REM Start the Flask server in a new terminal window
start "CiteSeekerAI Server" python src/web_interface/app.py

REM Wait a bit to ensure the server has time to start
timeout /t 15 > nul

REM Open the web interface in the default browser
start http://127.0.0.1:5000

echo.
echo CiteSeekerAI is running in a new window. You can close this window.
pause > nul

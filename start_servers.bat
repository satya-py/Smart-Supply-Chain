@echo off
echo ================================================
echo Starting IntelliSupply - AI Supply Chain Platform
echo ================================================
echo.

echo [1/3] Activating virtual environment...
call env\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

echo [2/3] Installing required packages (if needed)...
pip install fastapi uvicorn pandas numpy scikit-learn xgboost requests >nul 2>&1
echo ✓ Dependencies checked
echo.

echo [3/3] Starting servers...
echo.
echo ┌─────────────────────────────────────────────────┐
echo │ ML FastAPI Server: http://localhost:8000        │
echo │ Flask Backend: http://localhost:5000            │
echo │ Frontend: http://localhost:5000                 │
echo └─────────────────────────────────────────────────┘
echo.

echo Starting ML FastAPI Server (port 8000)...
start "ML FastAPI Server" cmd /k "cd \"d:\d_drive_project\googke solution challange\NOTE BOOKS\" && uvicorn fast_api_server:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo Starting Flask Backend Server (port 5000)...
start "Flask Backend Server" cmd /k "cd \"d:\d_drive_project\googke solution challange\Full_project\supply\" && python run.py"

echo.
echo ✓ Both servers starting...
echo.
echo Access the application at: http://localhost:5000
echo ML API Docs at: http://localhost:8000/docs
echo.
echo Press any key to stop all servers...
pause >nul

echo.
echo Stopping servers...
taskkill /FI "WindowTitle eq ML FastAPI Server*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq Flask Backend Server*" /T /F >nul 2>&1
echo ✓ Servers stopped

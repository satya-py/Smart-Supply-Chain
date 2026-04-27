"""
Start Both Servers (ML FastAPI + Flask Backend)
Run: python start_both_servers.py
"""
import subprocess
import sys
import os
import time
import signal

def install_packages():
    """Install required packages if not present."""
    print("Checking and installing required packages...")
    packages = [
        "fastapi", "uvicorn", "pandas", "numpy", 
        "scikit-learn", "xgboost", "requests"
    ]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def start_ml_server():
    """Start the ML FastAPI server."""
    print("\n" + "="*60)
    print("Starting ML FastAPI Server (port 8000)...")
    print("="*60)
    
    os.chdir(r"d:\d_drive_project\googke solution challange\NOTE BOOKS")
    
    # Start uvicorn as a background process
    ml_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "fast_api_server:app", 
        "--reload", "--port", "8000"
    ])
    
    return ml_process

def start_flask_server():
    """Start the Flask backend server."""
    print("\n" + "="*60)
    print("Starting Flask Backend Server (port 5000)...")
    print("="*60)
    
    os.chdir(r"d:\d_drive_project\googke solution challange\Full_project\supply")
    
    # Start Flask as a background process
    flask_process = subprocess.Popen([
        sys.executable, "run.py"
    ])
    
    return flask_process

def main():
    print("="*60)
    print("IntelliSupply - AI Supply Chain Platform")
    print("="*60)
    
    # Install packages
    install_packages()
    
    # Start ML server
    ml_process = start_ml_server()
    
    # Wait a bit for ML server to load models
    print("\nWaiting for ML server to initialize...")
    time.sleep(3)
    
    # Start Flask server
    flask_process = start_flask_server()
    
    print("\n" + "="*60)
    print("✅ Both servers are starting!")
    print("="*60)
    print("\nAccess points:")
    print("  🌐 Frontend: http://localhost:5000")
    print("  🤖 ML API Docs: http://localhost:8000/docs")
    print("  🏥 ML Health: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop both servers...")
    
    try:
        # Wait for both processes
        ml_process.wait()
        flask_process.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        ml_process.terminate()
        flask_process.terminate()
        
        # Wait for graceful shutdown
        try:
            ml_process.wait(timeout=5)
            flask_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ml_process.kill()
            flask_process.kill()
        
        print("✓ Servers stopped")

if __name__ == "__main__":
    main()

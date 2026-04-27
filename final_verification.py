"""
Final Verification Test - Complete System Check
Run: python final_verification.py
"""
import os
import sys
import subprocess
import time

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_file_exists(filepath, name):
    exists = os.path.exists(filepath)
    status = "[OK]" if exists else "[FAIL]"
    print(f"{status} {name}: {filepath}")
    return exists

def check_package_installed(package_name):
    try:
        import_name = package_name.replace("-", "_")
        if package_name == "scikit-learn":
            import_name = "sklearn"
        __import__(import_name)
        print(f"[OK] {package_name}")
        return True
    except ImportError:
        print(f"[FAIL] {package_name} - NOT INSTALLED")
        return False

def main():
    print_section("FINAL VERIFICATION TEST")
    print("Complete System Check for IntelliSupply Platform")
    
    # Check 1: File Existence
    print_section("1. MODEL FILES CHECK")
    files = [
        (r"d:\d_drive_project\googke solution challange\Models\model_scratch_xgb.pkl", "XGBoost Model"),
        (r"d:\d_drive_project\googke solution challange\AUTO ML\isolation_forest.pkl", "Isolation Forest"),
        (r"d:\d_drive_project\googke solution challange\Models\scaler.pkl", "Scaler"),
        (r"d:\d_drive_project\googke solution challange\feature_importance.csv", "Feature Importance"),
    ]
    
    all_files_ok = all(check_file_exists(f, n) for f, n in files)
    
    # Check 2: Python Packages
    print_section("2. PYTHON PACKAGES CHECK")
    packages = ["pandas", "numpy", "scikit-learn", "xgboost", "fastapi", "uvicorn", "requests"]
    all_packages_ok = all(check_package_installed(p) for p in packages)
    
    # Check 3: Test Scripts
    print_section("3. TEST SCRIPTS CHECK")
    test_scripts = [
        "test_xgb_model.py",
        "test_isolation_forest.py", 
        "test_scaler.py",
        "test_all_models.py",
        "test_fastapi_endpoint.py",
        "start_both_servers.py",
        "start_servers.bat"
    ]
    
    all_scripts_ok = True
    for script in test_scripts:
        path = os.path.join(r"d:\d_drive_project\googke solution challange", script)
        exists = os.path.exists(path)
        status = "[OK]" if exists else "[FAIL]"
        print(f"{status} {script}")
        if not exists:
            all_scripts_ok = False
    
    # Check 4: Run Model Test
    print_section("4. RUNNING MODEL LOADING TEST")
    try:
        result = subprocess.run(
            [sys.executable, "test_all_models.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("[OK] All models loaded and tested successfully!")
            # Show key output
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['✓', '[OK]', 'PASSED']):
                    print(f"   {line.strip()}")
        else:
            print("❌ Model test failed!")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Failed to run model test: {e}")
    
    # Final Summary
    print_section("VERIFICATION SUMMARY")
    
    checks = [
        ("Model Files", all_files_ok),
        ("Python Packages", all_packages_ok),
        ("Test Scripts", all_scripts_ok),
    ]
    
    all_passed = all(status for _, status in checks)
    
    for name, status in checks:
        symbol = "[OK]" if status else "[FAIL]"
        print(f"{symbol} {name}")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("[OK] ALL CHECKS PASSED - System is ready!")
        print("\nNext steps:")
        print("  1. Start servers: .\\start_servers.bat")
        print("  2. Or manually: python start_both_servers.py")
        print("  3. Access app: http://localhost:5000")
        print("  4. ML API docs: http://localhost:8000/docs")
    else:
        print("[FAIL] SOME CHECKS FAILED - Please review above")
        print("\nTo fix issues:")
        print("  1. Activate virtual environment: .\\env\\Scripts\\activate")
        print("  2. Install packages: pip install fastapi uvicorn pandas numpy scikit-learn xgboost requests")
        print("  3. Run test: python test_all_models.py")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

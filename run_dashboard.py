#!/usr/bin/env python3
"""
Battery Lifecycle Analysis Dashboard Runner
Run this script to launch the Streamlit dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Battery Lifecycle Analysis Dashboard...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "battery_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main function"""
    print("🔋 Battery Lifecycle Analysis Dashboard")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Check if Excel file exists
    if not os.path.exists("Battery Data Final (3).xlsx"):
        print("❌ Battery Data Final (3).xlsx not found!")
        print("Please ensure the Excel file is in the same directory as this script.")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Run dashboard
    run_dashboard()

if __name__ == "__main__":
    main()

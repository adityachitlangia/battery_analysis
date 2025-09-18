#!/usr/bin/env python3
"""
Modern Battery Dashboard Runner
Launch the commercial-grade BatteryLife Pro dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing requirements for BatteryLife Pro...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_modern_dashboard():
    """Run the modern Streamlit dashboard"""
    print("🚀 Starting BatteryLife Pro Dashboard...")
    print("=" * 60)
    print("🔋 BatteryLife Pro - Advanced Analytics Platform")
    print("📊 Commercial-Grade Battery Analysis Dashboard")
    print("🤖 AI-Powered Lifecycle Predictions")
    print("=" * 60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "battery_dashboard_modern.py",
            "--server.port", "8502",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light",
            "--theme.primaryColor", "#667eea",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f8fafc",
            "--theme.textColor", "#1e293b"
        ])
    except KeyboardInterrupt:
        print("\n👋 BatteryLife Pro Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main function"""
    print("🔋 BatteryLife Pro - Advanced Battery Analytics")
    print("=" * 60)
    
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
    
    # Run modern dashboard
    run_modern_dashboard()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Professional Battery Dashboard Runner
Launch the enterprise-grade BatteryTech Pro dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing requirements for BatteryTech Pro...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_professional_dashboard():
    """Run the professional Streamlit dashboard"""
    print("🚀 Starting BatteryTech Pro Dashboard...")
    print("=" * 60)
    print("🔋 BatteryTech Pro - Enterprise Analytics Platform")
    print("📊 Professional Battery Analysis Dashboard")
    print("🤖 AI-Powered Lifecycle Predictions")
    print("🎨 Modern Dark Theme with Glassmorphism")
    print("=" * 60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "battery_dashboard_professional.py",
            "--server.port", "8503",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "dark",
            "--theme.primaryColor", "#3b82f6",
            "--theme.backgroundColor", "#0f172a",
            "--theme.secondaryBackgroundColor", "#1e293b",
            "--theme.textColor", "#f8fafc"
        ])
    except KeyboardInterrupt:
        print("\n👋 BatteryTech Pro Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main function"""
    print("🔋 BatteryTech Pro - Professional Battery Analytics")
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
    
    # Run professional dashboard
    run_professional_dashboard()

if __name__ == "__main__":
    main()

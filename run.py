#!/usr/bin/env python3
"""
Startup script for Data Analysis AI Agent
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import streamlit
        import openai
        import pandas
        import plotly
        import numpy
        import sklearn
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment setup"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env file not found")
        print("Please copy .env.template to .env and configure your settings")
        return False
    
    # Load and check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not found in .env file")
        print("Please add your OpenAI API key to the .env file")
        return False
    
    print("✅ Environment configuration looks good")
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ["data", "config", "src", "tests"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ Directories created")

def main():
    """Main startup function"""
    print("🤖 Data Analysis AI Agent - Startup Script")
    print("=" * 50)
    
    # Check current directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        print("\nTo set up your environment:")
        print("1. Copy .env.template to .env")
        print("2. Add your OpenAI API key to the .env file")
        print("3. Run this script again")
        sys.exit(1)
    
    # Start the Streamlit app
    print("\n🚀 Starting Data Analysis AI Agent...")
    print("The web interface will open at: http://localhost:8501")
    print("\nTo stop the application, press Ctrl+C")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Data Analysis AI Agent stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting the application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Setup script for the Agentic System Backend
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return None

def main():
    """Main setup function"""
    print("🚀 Setting up Agentic System Backend...")
    
    # Check if virtual environment is activated
    if sys.prefix == sys.base_prefix:
        print("\n⚠️  Warning: No virtual environment detected.")
        print("It's recommended to create and activate a virtual environment first:")
        print("  python -m venv venv")
        print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies. Please check your Python environment.")
        return
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            run_command("cp .env.example .env", "Creating .env file from template")
            print("📝 Please edit .env file with your configuration")
        else:
            print("⚠️  .env.example not found")
    else:
        print("✅ .env file already exists")
    
    # Create uploads directory
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        print("✅ Created uploads directory")
    
    # Run a quick test to verify setup
    print("\n🧪 Running quick verification test...")
    try:
        # Try to import main modules
        import flask
        import pydantic
        import motor
        print("✅ Core dependencies verified")
        
        # Try to create app
        from app import create_app
        app = create_app('testing')
        print("✅ Flask app creation verified")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check your installation")
        return
    except Exception as e:
        print(f"⚠️  Setup verification warning: {e}")
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your MongoDB URI and other configuration")
    print("2. Start MongoDB if using local instance")
    print("3. Run the application: python app.py")
    print("4. Run tests: pytest")
    print("\n📖 See README.md for detailed documentation")

if __name__ == "__main__":
    main()
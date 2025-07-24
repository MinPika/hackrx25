#!/usr/bin/env python3
"""
HackRx 6.0 Quick Setup Script
Prepares your system for running the application
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("📋 Required: Python 3.8 or higher")
        print("🔗 Download from: https://python.org/downloads/")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_system_dependencies():
    """Install system-level dependencies"""
    print("🔧 Installing system dependencies...")
    
    system = platform.system().lower()
    
    if system == "linux":
        # Ubuntu/Debian
        commands = [
            ["sudo", "apt-get", "update"],
            ["sudo", "apt-get", "install", "-y", "tesseract-ocr", "poppler-utils", 
             "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev"]
        ]
        
        for cmd in commands:
            try:
                print(f"   Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️ Command failed: {e}")
                print("   📝 You may need to install these manually:")
                print("   sudo apt-get install tesseract-ocr poppler-utils")
                
    elif system == "darwin":  # macOS
        commands = [
            ["brew", "install", "tesseract", "poppler"]
        ]
        
        for cmd in commands:
            try:
                print(f"   Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️ Command failed: {e}")
                print("   📝 Install Homebrew first: https://brew.sh/")
                print("   Then run: brew install tesseract poppler")
                
    elif system == "windows":
        print("   📝 Windows detected - Manual installation required:")
        print("   1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Install Poppler: https://blog.alivate.com.au/poppler-windows/")
        print("   3. Add both to your PATH environment variable")
        
    else:
        print(f"   ⚠️ Unknown system: {system}")
        print("   📝 Please install tesseract-ocr and poppler-utils manually")

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("🐍 Setting up virtual environment...")
    
    venv_name = "hackrx_env"
    
    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
        print(f"✅ Created virtual environment: {venv_name}")
        
        # Get activation command
        if platform.system().lower() == "windows":
            activate_cmd = f"{venv_name}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_name}/bin/activate"
        
        print(f"📝 To activate: {activate_cmd}")
        return venv_name
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return None

def install_python_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Python dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def verify_installation():
    """Verify that all components are working"""
    print("🧪 Verifying installation...")
    
    # Test critical imports
    test_imports = [
        ("fastapi", "FastAPI"),
        ("doclayout_yolo", "DocLayout-YOLO"),
        ("cv2", "OpenCV"),
        ("pytesseract", "Tesseract"),
        ("groq", "Groq"),
        ("langchain", "LangChain"),
        ("chromadb", "ChromaDB")
    ]
    
    failed_imports = []
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"❌ Failed imports: {', '.join(failed_imports)}")
        return False
    
    print("✅ All critical components verified")
    return True

def create_project_structure():
    """Create necessary project directories"""
    print("📁 Creating project structure...")
    
    directories = [
        "temp_docs",
        "chroma_db",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created: {directory}/")

def run_quick_test():
    """Run a quick test to ensure everything works"""
    print("🚀 Running quick system test...")
    
    try:
        # Test basic import and initialization
        test_code = """
import sys
import os
sys.path.append('.')

# Test basic imports
from main import app
from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)

# Test health endpoint
response = client.get("/")
if response.status_code == 200:
    print("✅ Basic system test passed")
else:
    print(f"❌ Basic system test failed: {response.status_code}")
"""
        
        # Write test to temporary file
        with open("temp_test.py", "w") as f:
            f.write(test_code)
        
        # Run test
        result = subprocess.run([sys.executable, "temp_test.py"], 
                              capture_output=True, text=True, timeout=30)
        
        # Clean up
        os.remove("temp_test.py")
        
        if result.returncode == 0:
            print("✅ Quick test passed")
            return True
        else:
            print(f"❌ Quick test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Quick test error: {e}")
        return False

def display_next_steps():
    """Display next steps for the user"""
    print("\n🎯 SETUP COMPLETE - Next Steps:")
    print("=" * 50)
    print("1. 🚀 Start the application:")
    print("   python main.py")
    print("")
    print("2. 🧪 Test the system:")
    print("   python test_system.py")
    print("")
    print("3. 📦 Deploy to production:")
    print("   python deploy.py")
    print("")
    print("4. 🌐 Access endpoints:")
    print("   Health: http://localhost:8000/health")
    print("   API: http://localhost:8000/hackrx/run")
    print("")
    print("5. 🏆 Submit to HackRx:")
    print("   - Deploy to public platform")
    print("   - Submit webhook URL")
    print("   - Test with sample data")
    print("=" * 50)

def main():
    """Main setup function"""
    print("🏆 HackRx 6.0 - Quick Setup Script")
    print("=" * 50)
    print("🎯 This script will prepare your system for HackRx 6.0")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install system dependencies
    install_system_dependencies()
    
    # Create project structure
    create_project_structure()
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        print("📝 Make sure all project files are in the current directory")
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        print("📝 Try running manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        print("📝 Some components may not be working correctly")
        print("📝 Check error messages above and resolve issues")
        sys.exit(1)
    
    # Run quick test
    print("\n" + "=" * 50)
    if run_quick_test():
        print("🎉 SETUP SUCCESSFUL!")
        display_next_steps()
    else:
        print("⚠️ Setup completed with warnings")
        print("📝 System may work but some tests failed")
        display_next_steps()

if __name__ == "__main__":
    main()
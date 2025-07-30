#!/usr/bin/env python3
"""
HackRx 6.0 Deployment Helper Script
Automates deployment to various platforms
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "main.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def create_env_file():
    """Create .env file with configuration"""
    env_content = """# HackRx 6.0 Configuration
GROQ_API_KEY=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
HACKRX_AUTH_TOKEN=95f763f2e367cc7e5f72304cb9e9b84229f97f2a5b2b08f14b5034e8328596ec
PORT=8000
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env file")

def create_dockerfile():
    """Create Dockerfile for containerized deployment"""
    dockerfile_content = """FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    tesseract-ocr \\
    poppler-utils \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p temp_docs chroma_db

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Created Dockerfile")

def create_heroku_files():
    """Create Heroku-specific deployment files"""
    
    # Procfile
    procfile_content = "web: python main.py"
    with open("Procfile", "w") as f:
        f.write(procfile_content)
    
    # runtime.txt
    runtime_content = "python-3.9.18"
    with open("runtime.txt", "w") as f:
        f.write(runtime_content)
    
    # app.json for Heroku
    app_json = {
        "name": "hackrx-6-intelligent-qa",
        "description": "HackRx 6.0 - AI-powered document QA system",
        "image": "heroku/python",
        "stack": "heroku-20",
        "buildpacks": [
            {
                "url": "heroku/python"
            }
        ],
        "env": {
            "GROQ_API_KEY": {
                "description": "Groq API key for LLM processing",
                "value": "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"
            },
            "HACKRX_AUTH_TOKEN": {
                "description": "HackRx authentication token",
                "value": "95f763f2e367cc7e5f72304cb9e9b84229f97f2a5b2b08f14b5034e8328596ec"
            }
        },
        "formation": {
            "web": {
                "quantity": 1,
                "size": "standard-1x"
            }
        }
    }
    
    with open("app.json", "w") as f:
        json.dump(app_json, f, indent=2)
    
    print("✅ Created Heroku deployment files")

def create_railway_files():
    """Create Railway-specific deployment files"""
    
    # railway.json
    railway_config = {
        "build": {
            "builder": "DOCKERFILE"
        },
        "deploy": {
            "startCommand": "python main.py",
            "healthcheckPath": "/health",
            "healthcheckTimeout": 100,
            "restartPolicyType": "ON_FAILURE"
        }
    }
    
    with open("railway.json", "w") as f:
        json.dump(railway_config, f, indent=2)
    
    print("✅ Created Railway deployment files")

def create_render_files():
    """Create Render-specific deployment files"""
    
    # render.yaml
    render_config = {
        "services": [
            {
                "type": "web",
                "name": "hackrx-6-qa-system",
                "env": "python",
                "buildCommand": "pip install -r requirements.txt",
                "startCommand": "python main.py",
                "healthCheckPath": "/health",
                "envVars": [
                    {
                        "key": "GROQ_API_KEY",
                        "value": "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"
                    },
                    {
                        "key": "HACKRX_AUTH_TOKEN", 
                        "value": "95f763f2e367cc7e5f72304cb9e9b84229f97f2a5b2b08f14b5034e8328596ec"
                    }
                ]
            }
        ]
    }
    
    with open("render.yaml", "w") as f:
        json.dump(render_config, f, indent=2)
    
    print("✅ Created Render deployment files")

def test_local_deployment():
    """Test the application locally"""
    print("Testing local deployment...")
    
    try:
        # Install requirements
        print("Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        
        print("✅ Requirements installed successfully")
        
        # Test import
        print("Testing imports...")
        test_code = """
import sys
sys.path.append('.')
try:
    from main import app
    print("✅ Main application imports successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
"""
        
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Local test passed")
            return True
        else:
            print(f"❌ Local test failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Local test failed: {e}")
        return False

def create_deployment_guide():
    """Create deployment guide README"""
    guide_content = """# HackRx 6.0 Deployment Guide

## Quick Deployment Options

### 1. Railway (Recommended - Fastest)
```bash
# 1. Push to GitHub
git init
git add .
git commit -m "HackRx 6.0 submission"
git push origin main

# 2. Connect to Railway
# - Go to railway.app
# - Connect GitHub repo
# - Deploy automatically
```

### 2. Render (Free Tier Available)
```bash
# 1. Push to GitHub
git init  
git add .
git commit -m "HackRx 6.0 submission"
git push origin main

# 2. Connect to Render
# - Go to render.com
# - Connect GitHub repo
# - Use render.yaml config
```

### 3. Heroku (Reliable)
```bash
# Install Heroku CLI first
heroku create hackrx-6-qa-system
git add .
git commit -m "HackRx 6.0 submission"
git push heroku main
```

### 4. Docker (Any Platform)
```bash
docker build -t hackrx-6-qa .
docker run -p 8000:8000 hackrx-6-qa
```

## Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr poppler-utils

# Run locally
python main.py
```

## System Requirements
- Python 3.9+
- 2GB+ RAM (for AI models)
- Tesseract OCR
- Poppler utilities

## Testing Your Deployment
```bash
# Test health endpoint
curl https://your-app-url.com/health

# Test HackRx endpoint
curl -X POST https://your-app-url.com/hackrx/run \\
  -H "Authorization: Bearer 95f763f2e367cc7e5f72304cb9e9b84229f97f2a5b2b08f14b5034e8328596ec" \\
  -H "Content-Type: application/json" \\
  -d '{
    "documents": "https://example.com/test.pdf",
    "questions": ["What is covered?"]
  }'
```

## HackRx Submission
1. Deploy to any platform above
2. Get your public HTTPS URL
3. Submit webhook URL: `https://your-app.com/hackrx/run`
4. Test with sample data

## Performance Tips
- Use Railway/Render for fastest cold starts
- Heroku has 30s sleep time (might be slower)
- Docker deployments are most reliable

## Security
- API keys are pre-configured
- HTTPS is required for submission
- Bearer token authentication enabled

## Monitoring
- Health check: `/health`
- Root endpoint: `/` 
- Detailed logs in application output
"""
    
    # Fix for Windows encoding issue - use UTF-8 explicitly
    with open("DEPLOYMENT.md", "w", encoding='utf-8') as f:
        f.write(guide_content)
    
    print("✅ Created deployment guide")

def main():
    """Main deployment preparation function"""
    print("HackRx 6.0 - Deployment Preparation")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create all deployment files
    print("\nCreating deployment files...")
    create_env_file()
    create_dockerfile()
    create_heroku_files()
    create_railway_files()
    create_render_files()
    create_deployment_guide()
    
    # Test local deployment
    print("\nTesting local setup...")
    if test_local_deployment():
        print("✅ Local setup successful")
    else:
        print("⚠️ Local test failed - check dependencies")
    
    print("\nNext Steps:")
    print("1. Choose deployment platform (Railway recommended)")
    print("2. Push code to GitHub repository")
    print("3. Connect repository to deployment platform")
    print("4. Submit webhook URL to HackRx platform")
    print("5. Test with sample data")
    
    print("\nRead DEPLOYMENT.md for detailed instructions")
    print("Good luck with HackRx 6.0!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Install essential dependencies for UFC Predictor system
"""

import subprocess
import sys

# Essential packages for basic functionality
ESSENTIAL_PACKAGES = [
    "numpy>=1.21.0",
    "pandas>=1.3.0", 
    "scikit-learn>=1.0.0",
    "joblib>=1.1.0"
]

# Enhanced packages for full system
ENHANCED_PACKAGES = [
    "xgboost>=1.5.0",
    "lightgbm>=3.3.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "jupyter>=1.0.0",
    "plotly>=5.0.0"
]

# Web scraping packages (optional)
SCRAPING_PACKAGES = [
    "selenium>=4.0.0",
    "lxml>=4.6.0",
    "fake-useragent>=1.1.0"
]

def install_packages(packages, description):
    """Install packages using pip"""
    print(f"\n📦 Installing {description}...")
    
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
    
    return True

def test_imports():
    """Test if key packages can be imported"""
    print(f"\n🔍 Testing package imports...")
    
    test_packages = [
        ("numpy", "import numpy as np; print(f'numpy {numpy.__version__}')"),
        ("pandas", "import pandas as pd; print(f'pandas {pd.__version__}')"),
        ("sklearn", "import sklearn; print(f'scikit-learn {sklearn.__version__}')"),
        ("joblib", "import joblib; print(f'joblib {joblib.__version__}')")
    ]
    
    for name, import_test in test_packages:
        try:
            exec(import_test)
            print(f"   ✅ {name} working")
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            return False
    
    return True

def main():
    print("🚀 UFC PREDICTOR DEPENDENCY INSTALLER")
    print("=" * 50)
    
    # Install essential packages first
    if not install_packages(ESSENTIAL_PACKAGES, "essential packages"):
        print("❌ Failed to install essential packages")
        return False
    
    # Test essential imports
    if not test_imports():
        print("❌ Essential package testing failed")
        return False
    
    print("\n✅ ESSENTIAL DEPENDENCIES INSTALLED SUCCESSFULLY!")
    print("\nYou can now run:")
    print("   • python3 main.py --mode predict --fighter1 'Jon Jones' --fighter2 'Stipe Miocic'")
    print("   • python3 enhanced_system_demo.py --benchmark")
    print("   • python3 run_profitability_analysis.py --sample")
    
    # Ask about enhanced packages
    print(f"\n🤔 Install enhanced packages for full functionality? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        if install_packages(ENHANCED_PACKAGES, "enhanced ML packages"):
            print("\n✅ ENHANCED PACKAGES INSTALLED!")
            print("\nFull system capabilities now available:")
            print("   • XGBoost and LightGBM models")
            print("   • Advanced visualization")
            print("   • Jupyter notebook support")
        else:
            print("\n⚠️  Some enhanced packages failed - basic system still works")
    
    # Ask about scraping packages
    print(f"\n🕷️  Install web scraping packages? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        if install_packages(SCRAPING_PACKAGES, "web scraping packages"):
            print("\n✅ WEB SCRAPING PACKAGES INSTALLED!")
            print("   • Live odds scraping capabilities")
            print("   • UFC stats data collection")
        else:
            print("\n⚠️  Some scraping packages failed - prediction system still works")
    
    print(f"\n🏁 INSTALLATION COMPLETE!")
    print(f"Next step: python3 simple_demo.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
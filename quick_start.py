#!/usr/bin/env python3
"""
Quick Start Script for UFC Predictor System
===========================================

This script helps you get the UFC prediction system running with minimal effort.
It detects what's available and guides you through the setup process.
"""

import os
import sys
from pathlib import Path
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def check_dependencies():
    """Check what dependencies are installed"""
    print("🔍 CHECKING DEPENDENCIES")
    print("-" * 40)
    
    deps_status = {}
    
    # Essential dependencies
    essential_deps = [
        ("numpy", "import numpy"),
        ("pandas", "import pandas"), 
        ("sklearn", "import sklearn"),
        ("joblib", "import joblib")
    ]
    
    # Enhanced dependencies
    enhanced_deps = [
        ("xgboost", "import xgboost"),
        ("lightgbm", "import lightgbm"),
        ("matplotlib", "import matplotlib"),
        ("seaborn", "import seaborn")
    ]
    
    print("Essential Dependencies:")
    essential_ok = True
    for name, import_cmd in essential_deps:
        try:
            exec(import_cmd)
            print(f"   ✅ {name}")
            deps_status[name] = True
        except ImportError:
            print(f"   ❌ {name} (missing)")
            deps_status[name] = False
            essential_ok = False
    
    print("\nEnhanced Dependencies:")
    enhanced_ok = True
    for name, import_cmd in enhanced_deps:
        try:
            exec(import_cmd)
            print(f"   ✅ {name}")
            deps_status[name] = True
        except ImportError:
            print(f"   ⚠️  {name} (missing)")
            deps_status[name] = False
            enhanced_ok = False
    
    return essential_ok, enhanced_ok, deps_status

def check_data_and_models():
    """Check for existing data and trained models"""
    print(f"\n📊 CHECKING DATA & MODELS")
    print("-" * 40)
    
    # Check for data files
    data_dir = PROJECT_ROOT / "data"
    model_dir = PROJECT_ROOT / "model"
    
    has_data = False
    has_models = False
    
    print("Data Files:")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            has_data = True
            for csv_file in csv_files[:5]:  # Show first 5
                print(f"   ✅ {csv_file.name}")
        else:
            print("   ❌ No CSV data files found")
    else:
        print("   ❌ No data directory")
    
    print("\nTrained Models:")
    if model_dir.exists():
        # Check for actual model files
        model_files = list(model_dir.glob("*.joblib")) + list(model_dir.glob("*.pkl"))
        if model_files:
            has_models = True
            for model_file in model_files[:3]:  # Show first 3
                print(f"   ✅ {model_file.name}")
        else:
            # Check for model metadata
            json_files = list(model_dir.glob("*.json"))
            training_dirs = list(model_dir.glob("training_*"))
            if json_files or training_dirs:
                print("   ⚠️  Model metadata found but no trained models")
                for json_file in json_files[:3]:
                    print(f"      📄 {json_file.name}")
                for training_dir in training_dirs:
                    print(f"      📁 {training_dir.name}")
            else:
                print("   ❌ No model files found")
    else:
        print("   ❌ No model directory")
    
    return has_data, has_models

def get_runnable_demos():
    """Find which demos can be run with current setup"""
    print(f"\n🎮 AVAILABLE DEMOS")
    print("-" * 40)
    
    demos = []
    
    # Always available (no dependencies)
    simple_demo = PROJECT_ROOT / "simple_demo.py"
    if simple_demo.exists():
        demos.append(("Simple Strategy Demo", "python3 simple_demo.py", "Shows enhancement strategy"))
        print(f"   ✅ Simple Strategy Demo (no dependencies needed)")
    
    # Check if we can run basic analysis
    try:
        import pandas
        # Basic demos that need pandas
        
        quick_analysis = PROJECT_ROOT / "quick_analysis.sh"
        if quick_analysis.exists():
            demos.append(("Quick Analysis", "./quick_analysis.sh", "Interactive analysis menu"))
            print(f"   ✅ Quick Analysis Menu")
            
    except ImportError:
        print(f"   ⚠️  Analysis tools need pandas")
    
    # Check for enhanced demos
    try:
        import numpy, pandas, sklearn
        
        enhanced_demo = PROJECT_ROOT / "enhanced_system_demo.py"
        if enhanced_demo.exists():
            demos.append(("Enhanced System Demo", "python3 enhanced_system_demo.py --benchmark", "Performance benchmarks"))
            print(f"   ✅ Enhanced System Demo")
            
        main_script = PROJECT_ROOT / "main.py"
        if main_script.exists():
            demos.append(("Main Prediction Script", "python3 main.py --mode predict --fighter1 'Jon Jones' --fighter2 'Stipe Miocic'", "Single fight prediction"))
            print(f"   ✅ Main Prediction System")
            
    except ImportError:
        print(f"   ⚠️  Advanced demos need numpy/pandas/sklearn")
    
    return demos

def show_next_steps(essential_ok, enhanced_ok, has_data, has_models):
    """Show recommended next steps based on current state"""
    print(f"\n🎯 RECOMMENDED NEXT STEPS")
    print("=" * 50)
    
    if not essential_ok:
        print("1. INSTALL ESSENTIAL DEPENDENCIES")
        print("   Run: python3 install_dependencies.py")
        print("   This will install numpy, pandas, scikit-learn")
        print()
        
    if essential_ok and not has_data and not has_models:
        print("1. GET SAMPLE DATA OR CREATE DEMO DATA")
        print("   Option A: Run enhanced demo with synthetic data:")
        print("             python3 enhanced_system_demo.py --quick")
        print("   Option B: Check if webscraping can collect data:")
        print("             python3 webscraper/scraping.py")
        print()
        
    if essential_ok and not enhanced_ok:
        print("2. INSTALL ENHANCED DEPENDENCIES (Optional)")
        print("   Run: python3 install_dependencies.py")
        print("   Choose 'yes' for enhanced packages (XGBoost, LightGBM, etc.)")
        print()
        
    if essential_ok:
        print("3. RUN AVAILABLE DEMOS")
        print("   Start with: python3 simple_demo.py")
        print("   Then try:   python3 enhanced_system_demo.py --benchmark")
        print()
        
        print("4. TEST PROFITABILITY ANALYSIS")
        print("   Try: python3 run_profitability_analysis.py --sample")
        print("   This will show betting analysis capabilities")
        print()

def main():
    """Main quick start routine"""
    print("🚀 UFC PREDICTOR QUICK START")
    print("=" * 60)
    print("This script will help you get the system running with minimal effort.")
    print()
    
    # Check dependencies
    essential_ok, enhanced_ok, deps_status = check_dependencies()
    
    # Check data and models
    has_data, has_models = check_data_and_models()
    
    # Show available demos
    demos = get_runnable_demos()
    
    # Show next steps
    show_next_steps(essential_ok, enhanced_ok, has_data, has_models)
    
    # Offer to run something immediately
    if demos:
        print("🎮 READY TO RUN SOMETHING NOW?")
        print("-" * 40)
        
        for i, (name, command, description) in enumerate(demos, 1):
            print(f"{i}. {name}")
            print(f"   Command: {command}")
            print(f"   Description: {description}")
            print()
        
        try:
            choice = input("Choose a demo to run (1-{}, or press Enter to exit): ".format(len(demos)))
            if choice.strip() and choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(demos):
                    name, command, description = demos[choice_idx]
                    print(f"\n🚀 Running {name}...")
                    print(f"Command: {command}")
                    print("-" * 40)
                    
                    # Run the command
                    if command.startswith("python3"):
                        # Python script
                        script_path = command.replace("python3 ", "").split()[0]
                        script_args = command.replace("python3 " + script_path, "").strip().split()
                        
                        try:
                            subprocess.run([sys.executable, script_path] + script_args, cwd=PROJECT_ROOT)
                        except Exception as e:
                            print(f"Error running {name}: {e}")
                    else:
                        # Shell command
                        try:
                            subprocess.run(command, shell=True, cwd=PROJECT_ROOT)
                        except Exception as e:
                            print(f"Error running {name}: {e}")
                            
        except KeyboardInterrupt:
            print("\n\nExiting...")
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n✅ Quick start complete! Check the recommendations above for next steps.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test Current UFC Prediction System Setup
=======================================

Quick test to see what works with your current dependencies after running:
uv pip install requirements

This will test what's working RIGHT NOW without installing anything else.
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test what Python packages are available"""
    print("🔍 TESTING CURRENT PYTHON ENVIRONMENT")
    print("=" * 50)
    
    # Basic packages that should be available
    packages_to_test = [
        ('pandas', '📊 Data manipulation'),
        ('numpy', '🔢 Numerical computing'),
        ('matplotlib', '📈 Plotting'),
        ('seaborn', '📊 Statistical plotting'),
        ('sklearn', '🤖 Machine learning'),
        ('requests', '🌐 HTTP requests'),
        ('beautifulsoup4', '🍜 Web scraping'),
        ('joblib', '💾 Model persistence'),
        ('xgboost', '🚀 Gradient boosting'),
        ('lightgbm', '⚡ Light gradient boosting')
    ]
    
    available = []
    missing = []
    
    for package, description in packages_to_test:
        try:
            if package == 'beautifulsoup4':
                import bs4
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package:15} - {description}")
            available.append(package)
        except ImportError:
            print(f"❌ {package:15} - {description} (Missing)")
            missing.append(package)
    
    print(f"\n📊 SUMMARY: {len(available)}/{len(packages_to_test)} packages available")
    return available, missing

def test_enhanced_system_imports():
    """Test if our enhanced system can be imported"""
    print("\n🚀 TESTING ENHANCED SYSTEM IMPORTS")
    print("=" * 50)
    
    # Add src to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / "src"))
    
    enhanced_modules = [
        ('ufc_elo_system', '🥊 ELO Rating System'),
        ('advanced_feature_engineering', '🔧 Advanced Features'),
        ('advanced_ensemble_methods', '🎯 Ensemble Methods'),
        ('enhanced_ufc_predictor', '🚀 Enhanced Predictor'),
        ('model_training', '🎓 Model Training'),
        ('feature_engineering', '⚙️ Feature Engineering'),
        ('prediction', '🎯 Prediction Engine')
    ]
    
    working_modules = []
    failed_modules = []
    
    for module_name, description in enhanced_modules:
        try:
            module = __import__(f"src.{module_name}", fromlist=[module_name])
            print(f"✅ {module_name:25} - {description}")
            working_modules.append(module_name)
        except ImportError as e:
            print(f"❌ {module_name:25} - {description} (Import Error: {str(e)[:50]}...)")
            failed_modules.append(module_name)
        except Exception as e:
            print(f"⚠️  {module_name:25} - {description} (Other Error: {str(e)[:50]}...)")
            failed_modules.append(module_name)
    
    print(f"\n📊 ENHANCED MODULES: {len(working_modules)}/{len(enhanced_modules)} working")
    return working_modules, failed_modules

def test_project_structure():
    """Test project file structure"""
    print("\n📁 TESTING PROJECT STRUCTURE")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    
    expected_structure = [
        ('main.py', '🎯 Main CLI Interface'),
        ('src/', '📁 Source Code Directory'),
        ('config/', '⚙️ Configuration Files'),
        ('webscraper/', '🌐 Web Scraping Tools'),
        ('data/', '📊 Data Directory (may not exist yet)'),
        ('model/', '🤖 Model Directory (may not exist yet)'),
        ('requirements.txt', '📋 Dependencies List'),
        ('enhanced_system_demo.py', '🚀 Enhanced Demo'),
        ('bootstrap_enhanced_system.py', '⚡ Bootstrap Script')
    ]
    
    existing = []
    missing = []
    
    for item, description in expected_structure:
        item_path = project_root / item
        if item_path.exists():
            if item.endswith('/'):
                file_count = len(list(item_path.glob('*')))
                print(f"✅ {item:25} - {description} ({file_count} files)")
            else:
                print(f"✅ {item:25} - {description}")
            existing.append(item)
        else:
            print(f"❌ {item:25} - {description} (Missing)")
            missing.append(item)
    
    print(f"\n📊 PROJECT STRUCTURE: {len(existing)}/{len(expected_structure)} items found")
    return existing, missing

def test_basic_functionality():
    """Test basic functionality that should work now"""
    print("\n🧪 TESTING BASIC FUNCTIONALITY")
    print("=" * 50)
    
    tests = []
    
    # Test 1: Can we create synthetic data?
    try:
        import pandas as pd
        import numpy as np
        
        # Simple test dataframe
        test_df = pd.DataFrame({
            'fighter': ['Jon Jones', 'Stipe Miocic'],
            'rating': [1500, 1450],
            'wins': [27, 20]
        })
        
        print("✅ Basic DataFrame creation - Working")
        tests.append("dataframe")
    except Exception as e:
        print(f"❌ Basic DataFrame creation - Failed: {e}")
    
    # Test 2: Can we do basic ML?
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Simple test model
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        print("✅ Basic ML model training - Working")
        tests.append("ml_basic")
    except Exception as e:
        print(f"❌ Basic ML model training - Failed: {e}")
    
    # Test 3: Can we run enhanced system demo?
    try:
        project_root = Path(__file__).parent
        demo_script = project_root / "enhanced_system_demo.py"
        
        if demo_script.exists():
            print("✅ Enhanced demo script - Available")
            tests.append("demo_available")
        else:
            print("❌ Enhanced demo script - Missing")
    except Exception as e:
        print(f"❌ Enhanced demo script check - Failed: {e}")
    
    return tests

def show_next_steps(available_packages, working_modules, tests):
    """Show recommended next steps based on current state"""
    print("\n🎯 RECOMMENDED NEXT STEPS")
    print("=" * 50)
    
    # Determine current capability level
    has_basic_ml = 'sklearn' in available_packages and 'pandas' in available_packages
    has_enhanced_ml = 'xgboost' in available_packages and 'lightgbm' in available_packages
    has_working_modules = len(working_modules) >= 5
    
    if not has_basic_ml:
        print("❗ CRITICAL: Missing basic ML packages")
        print("   → Run: pip install scikit-learn pandas numpy")
        print("   → This is required for any ML functionality")
    
    elif not has_enhanced_ml:
        print("⚡ IMMEDIATE ACTION: Install enhanced ML libraries")
        print("   → Run: pip install xgboost lightgbm")
        print("   → This unlocks the advanced ensemble methods")
        print("   → Then run: python3 bootstrap_enhanced_system.py --quick")
    
    elif has_working_modules:
        print("🚀 READY FOR ADVANCED TESTING")
        print("   → Run: python3 bootstrap_enhanced_system.py --quick")
        print("   → This creates synthetic data and tests full system")
        print("   → Then run: python3 enhanced_system_demo.py")
    
    else:
        print("🔧 DEBUG NEEDED: System partially working")
        print("   → Some modules failing to import")
        print("   → Check Python version and package versions")
    
    print("\n📋 STEP-BY-STEP PROGRESSION:")
    print("1. ✅ Check current setup (you just did this!)")
    
    if has_basic_ml:
        print("2. ✅ Basic ML packages available")
    else:
        print("2. ❌ Install basic ML: pip install scikit-learn pandas numpy")
    
    if has_enhanced_ml:
        print("3. ✅ Enhanced ML packages available")
    else:
        print("3. ❌ Install enhanced ML: pip install xgboost lightgbm")
    
    print("4. ⏳ Create test data: python3 bootstrap_enhanced_system.py --quick")
    print("5. ⏳ Run full demo: python3 enhanced_system_demo.py")
    print("6. ⏳ Get real data: python3 webscraper/scraping.py")
    print("7. ⏳ Train models: python3 main.py --mode pipeline --tune")
    
    print("\n🎯 IMMEDIATE ACTION:")
    if not has_enhanced_ml:
        print("   pip install xgboost lightgbm joblib")
        print("   python3 bootstrap_enhanced_system.py --quick")
    else:
        print("   python3 bootstrap_enhanced_system.py --quick")

def main():
    """Run all tests and show results"""
    print("🚀 UFC PREDICTOR SYSTEM - CURRENT STATUS CHECK")
    print("=" * 60)
    print("Testing what works with your current setup after: uv pip install requirements")
    print()
    
    # Run all tests
    available_packages, missing_packages = test_basic_imports()
    working_modules, failed_modules = test_enhanced_system_imports()
    existing_files, missing_files = test_project_structure()
    working_tests = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 OVERALL STATUS SUMMARY")
    print("=" * 60)
    
    total_score = (
        len(available_packages) * 2 +  # Packages are important
        len(working_modules) +         # Modules matter
        len(existing_files) +          # Structure is good
        len(working_tests) * 2         # Functionality is key
    )
    
    max_score = (10 * 2) + 7 + 9 + (3 * 2)  # Maximum possible
    percentage = (total_score / max_score) * 100
    
    print(f"🎯 System Readiness: {percentage:.1f}%")
    print(f"📦 Python Packages: {len(available_packages)}/10 available")
    print(f"🚀 Enhanced Modules: {len(working_modules)}/7 working")
    print(f"📁 Project Structure: {len(existing_files)}/9 items found")
    print(f"🧪 Basic Functionality: {len(working_tests)}/3 tests passed")
    
    # Next steps
    show_next_steps(available_packages, working_modules, working_tests)
    
    print(f"\n✅ System check complete! Your system is {percentage:.1f}% ready.")

if __name__ == "__main__":
    main()
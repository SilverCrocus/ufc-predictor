#!/usr/bin/env python3
"""
🔧 UFC Enhanced Notebook Diagnostics
===================================

This script helps diagnose and explain common issues when running the 
UFC Enhanced Card Analysis notebook.
"""

import sys
import os
from pathlib import Path

def print_header():
    print("🔧 UFC ENHANCED NOTEBOOK DIAGNOSTICS")
    print("=" * 50)
    print()

def check_environment():
    """Check Python environment and dependencies"""
    print("🐍 PYTHON ENVIRONMENT")
    print("-" * 30)
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check virtual environment
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment (.venv) found")
    else:
        print("⚠️  No .venv directory found")
    
    print()

def check_playwright_installation():
    """Check Playwright installation status"""
    print("🎭 PLAYWRIGHT STATUS")
    print("-" * 30)
    
    try:
        import playwright
        print("✅ Playwright package installed")
        
        # Check browser installation
        from playwright.sync_api import sync_playwright
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                browser.close()
                print("✅ Chromium browser available")
        except Exception as e:
            print(f"❌ Browser not available: {e}")
            print("💡 Fix: Run 'playwright install chromium'")
            
    except ImportError:
        print("❌ Playwright not installed")
        print("💡 Fix: Run 'uv pip install playwright' then 'playwright install'")
    
    print()

def check_model_status():
    """Check UFC model availability"""
    print("🥊 UFC MODEL STATUS")
    print("-" * 30)
    
    try:
        sys.path.append('src')
        from src.ufc_fight_predictor import UFCFightPredictor
        
        predictor = UFCFightPredictor()
        if predictor.is_loaded:
            print(f"✅ Models loaded successfully")
            print(f"   Accuracy: {predictor.model_accuracy:.1%}")
            print(f"   Location: {predictor.latest_model_dir}")
        else:
            print("❌ Models not loaded")
            print("💡 Fix: Run 'python main.py --mode train' to train models")
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
    
    print()

def explain_scraping_behavior():
    """Explain how the Polymarket scraping works"""
    print("🌐 POLYMARKET SCRAPING BEHAVIOR")
    print("-" * 30)
    
    print("How it works:")
    print("1. 🔄 Attempt to scrape live odds from Polymarket")
    print("2. ⏱️  If it times out (common), fall back to simulated data")
    print("3. ✅ Continue with betting analysis using either data source")
    print()
    
    print("Common outcomes:")
    print("• ✅ Live scraping succeeds → Use real odds")
    print("• ⏰ Live scraping times out → Use realistic simulated odds")
    print("• 🚫 Network error → Use simulated odds as fallback")
    print()
    
    print("💡 Key point: The notebook works either way!")
    print("   Simulated odds are realistic and match the actual fight card.")
    print()

def provide_solutions():
    """Provide step-by-step solutions"""
    print("🛠️  SOLUTIONS & NEXT STEPS")
    print("-" * 30)
    
    print("If the notebook 'isn't working', try these steps:")
    print()
    
    print("1️⃣  Test your setup:")
    print("   python test_notebook_execution.py")
    print()
    
    print("2️⃣  If Playwright errors:")
    print("   source .venv/bin/activate")
    print("   playwright install chromium")
    print()
    
    print("3️⃣  If models missing:")
    print("   python main.py --mode train")
    print()
    
    print("4️⃣  Expected notebook behavior:")
    print("   • All cells should execute without errors")
    print("   • You may see 'timeout' messages (this is normal)")
    print("   • Final output shows betting recommendations")
    print("   • Works with both live and simulated odds")
    print()
    
    print("5️⃣  If you're still having issues:")
    print("   • Check that you're running in the correct directory")
    print("   • Ensure virtual environment is activated")
    print("   • Try restarting Jupyter kernel")
    print()

def check_specific_files():
    """Check for required files"""
    print("📁 REQUIRED FILES CHECK")
    print("-" * 30)
    
    required_files = [
        "UFC_Enhanced_Card_Analysis.ipynb",
        "src/ufc_fight_predictor.py",
        "webscraper/polymarket_scraper.py",
        "model/"
    ]
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    print()

def main():
    """Run diagnostics"""
    print_header()
    
    check_environment()
    check_playwright_installation()
    check_model_status()
    check_specific_files()
    explain_scraping_behavior()
    provide_solutions()
    
    print("🎯 SUMMARY")
    print("-" * 30)
    print("The notebook should work even if live scraping fails.")
    print("Timeout errors are normal - the system falls back to simulated data.")
    print("If all files exist and models are loaded, you're ready to go!")
    print()
    print("💡 Remember: Simulated data produces realistic betting recommendations!")

if __name__ == "__main__":
    main()
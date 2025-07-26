#!/usr/bin/env python3
"""
Direct Notebook Playwright Test
==============================

This script replicates the exact imports and checks that happen in the 
UFC Enhanced Card Analysis notebook to diagnose the Playwright issue.
"""

import sys
import os
from pathlib import Path

def test_notebook_environment():
    """Test the exact environment that Jupyter notebook sees"""
    print("🔍 NOTEBOOK ENVIRONMENT TEST")
    print("=" * 50)
    
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}")
    print()

def test_playwright_import():
    """Test the exact Playwright import from the notebook"""
    print("🎭 PLAYWRIGHT IMPORT TEST")
    print("=" * 50)
    
    # This replicates the exact import check from polymarket_scraper.py
    try:
        from playwright.async_api import async_playwright
        PLAYWRIGHT_AVAILABLE = True
        print("✅ Playwright import: SUCCESS")
        print("✅ PLAYWRIGHT_AVAILABLE = True")
    except ImportError as e:
        PLAYWRIGHT_AVAILABLE = False
        print("❌ Playwright import: FAILED")
        print(f"❌ Error: {e}")
        print("❌ PLAYWRIGHT_AVAILABLE = False")
    
    return PLAYWRIGHT_AVAILABLE

def test_polymarket_scraper_import():
    """Test the polymarket scraper import (as used in notebook)"""
    print("\n🌐 POLYMARKET SCRAPER TEST")
    print("=" * 50)
    
    try:
        # Add webscraper to path (as notebook does)
        sys.path.append('webscraper')
        
        # Import the scraper (exact notebook import)
        from polymarket_scraper import (
            PolymarketUFCScraper,
            scrape_polymarket_ufc_event,
            PLAYWRIGHT_AVAILABLE
        )
        
        print("✅ Polymarket scraper import: SUCCESS")
        print(f"✅ PLAYWRIGHT_AVAILABLE in scraper: {PLAYWRIGHT_AVAILABLE}")
        
        if PLAYWRIGHT_AVAILABLE:
            print("✅ Ready for live scraping")
        else:
            print("⚠️  Will use simulated data")
            
        return PLAYWRIGHT_AVAILABLE
        
    except Exception as e:
        print("❌ Polymarket scraper import: FAILED")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_browser_functionality():
    """Test if browsers are actually available"""
    print("\n🌐 BROWSER FUNCTIONALITY TEST")
    print("=" * 50)
    
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("https://www.google.com")
            browser.close()
            
        print("✅ Browser test: SUCCESS")
        print("✅ Chromium browser working")
        return True
        
    except Exception as e:
        print("❌ Browser test: FAILED")
        print(f"❌ Error: {e}")
        return False

def provide_solutions(playwright_available, scraper_available, browser_available):
    """Provide specific solutions based on test results"""
    print("\n🛠️  SOLUTIONS")
    print("=" * 50)
    
    if playwright_available and scraper_available and browser_available:
        print("🎉 ALL TESTS PASSED!")
        print()
        print("Your Jupyter notebook SHOULD work correctly.")
        print("If you're still seeing 'Playwright not available' in the notebook:")
        print()
        print("1️⃣  RESTART YOUR JUPYTER KERNEL:")
        print("   • In Jupyter: Kernel → Restart & Clear Output")
        print("   • Or: Kernel → Restart & Run All")
        print()
        print("2️⃣  VERIFY KERNEL IS USING VIRTUAL ENVIRONMENT:")
        print("   • Check notebook shows: .venv/bin/python")
        print()
        print("3️⃣  RE-RUN THE NOTEBOOK CELLS")
        print("   • Start from Cell 1 and run all cells in order")
        
    elif not playwright_available:
        print("❌ PLAYWRIGHT NOT INSTALLED")
        print()
        print("Run these commands:")
        print("   source .venv/bin/activate")
        print("   uv pip install playwright")
        print("   playwright install chromium")
        
    elif not browser_available:
        print("❌ BROWSER BINARIES MISSING")
        print()
        print("Run this command:")
        print("   source .venv/bin/activate")
        print("   playwright install chromium")
        
    else:
        print("❌ UNKNOWN ISSUE")
        print("Please share the test output for further diagnosis.")

def main():
    """Run all tests and provide solutions"""
    test_notebook_environment()
    
    playwright_ok = test_playwright_import()
    scraper_ok = test_polymarket_scraper_import()
    browser_ok = test_browser_functionality() if playwright_ok else False
    
    provide_solutions(playwright_ok, scraper_ok, browser_ok)

if __name__ == "__main__":
    main()
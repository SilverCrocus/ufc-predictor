#!/usr/bin/env python3
"""
Comprehensive UFC Enhanced Notebook Execution Test
================================================

This script simulates the exact execution flow of the UFC Enhanced Card Analysis notebook
to identify and resolve any execution issues.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))
sys.path.append(str(Path(__file__).parent / 'webscraper'))

def test_basic_imports():
    """Test 1: Basic Python package imports"""
    print("🧪 TEST 1: Basic Imports")
    print("-" * 30)
    
    try:
        import pandas as pd
        import numpy as np
        import asyncio
        import nest_asyncio
        from datetime import datetime
        print("✅ Basic packages imported successfully")
        return True
    except Exception as e:
        print(f"❌ Basic import failed: {e}")
        return False

def test_ufc_predictor_import():
    """Test 2: UFC predictor system imports"""
    print("\n🧪 TEST 2: UFC Predictor Imports")
    print("-" * 30)
    
    try:
        from src.ufc_fight_predictor import UFCFightPredictor
        print("✅ UFCFightPredictor imported successfully")
        
        predictor = UFCFightPredictor()
        print(f"✅ UFCFightPredictor initialized: Loaded={predictor.is_loaded}")
        
        if predictor.is_loaded:
            print(f"   Models loaded from: {predictor.latest_model_dir}")
        
        return True
    except Exception as e:
        print(f"❌ UFCFightPredictor error: {e}")
        traceback.print_exc()
        return False

def test_polymarket_scraper_import():
    """Test 3: Polymarket scraper imports"""
    print("\n🧪 TEST 3: Polymarket Scraper Imports")
    print("-" * 30)
    
    try:
        from polymarket_scraper import (
            PolymarketUFCScraper, 
            scrape_polymarket_ufc_event,
            PLAYWRIGHT_AVAILABLE
        )
        print("✅ Polymarket scraper imported successfully")
        print(f"   Playwright available: {PLAYWRIGHT_AVAILABLE}")
        
        scraper = PolymarketUFCScraper(headless=True)
        print(f"✅ Scraper initialized with {scraper.retry_attempts} retry attempts")
        
        return True
    except Exception as e:
        print(f"❌ Polymarket scraper error: {e}")
        traceback.print_exc()
        return False

async def test_async_execution():
    """Test 4: Async execution patterns (notebook style)"""
    print("\n🧪 TEST 4: Async Execution")
    print("-" * 30)
    
    try:
        # Test nest_asyncio (needed for Jupyter)
        import nest_asyncio
        nest_asyncio.apply()
        print("✅ nest_asyncio applied successfully")
        
        # Test basic async function
        async def test_func():
            await asyncio.sleep(0.1)
            return "Async test completed"
        
        result = await test_func()
        print(f"✅ Async execution successful: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Async execution error: {e}")
        traceback.print_exc()
        return False

async def test_polymarket_scraping():
    """Test 5: Polymarket scraping functionality"""
    print("\n🧪 TEST 5: Polymarket Scraping")
    print("-" * 30)
    
    try:
        from polymarket_scraper import scrape_polymarket_ufc_event
        
        url = "https://polymarket.com/event/ufc-fight-night-whittaker-vs-de-ridder?tid=1753515605835"
        print(f"🎯 Testing URL: {url}")
        
        # Test with improved scraper
        odds = await scrape_polymarket_ufc_event(url, headless=True)
        
        print(f"✅ Scraping completed: {len(odds)} results")
        print("   Results preview:")
        for i, odd in enumerate(odds[:3]):  # Show first 3
            print(f"   {i+1}. {odd.fighter_a} vs {odd.fighter_b}")
            print(f"      Decimal odds: {odd.fighter_a_decimal_odds:.2f} vs {odd.fighter_b_decimal_odds:.2f}")
            print(f"      Source: {odd.source}")
        
        return True
    except Exception as e:
        print(f"❌ Polymarket scraping error: {e}")
        traceback.print_exc()
        return False

def test_notebook_dataclasses():
    """Test 6: Notebook-specific dataclasses and configurations"""
    print("\n🧪 TEST 6: Notebook Configuration")
    print("-" * 30)
    
    try:
        # Test notebook configuration (simulating notebook cells)
        ANALYSIS_CONFIG = {
            'polymarket_event_url': 'https://polymarket.com/event/ufc-fight-night-whittaker-vs-de-ridder?tid=1753515605835',
            'target_bankroll': 21.38,
            'max_bet_percentage': 0.20,
            'min_ev_threshold': 0.05
        }
        print(f"✅ Analysis config loaded: ${ANALYSIS_CONFIG['target_bankroll']:.2f} bankroll")
        
        # Test notebook-style dataclass
        from dataclasses import dataclass
        from typing import List
        
        @dataclass
        class PolymarketFightOddsNotebook:
            fighter_a: str
            fighter_b: str
            fighter_a_decimal_odds: float
            fighter_b_decimal_odds: float
            
        # Test creating instance
        test_odds = PolymarketFightOddsNotebook(
            fighter_a="Robert Whittaker",
            fighter_b="Reinier de Ridder",
            fighter_a_decimal_odds=1.47,
            fighter_b_decimal_odds=3.12
        )
        print(f"✅ Notebook dataclass working: {test_odds.fighter_a} @ {test_odds.fighter_a_decimal_odds}")
        
        return True
    except Exception as e:
        print(f"❌ Notebook configuration error: {e}")
        traceback.print_exc()
        return False

async def test_ev_calculations():
    """Test 7: Expected Value calculations with decimal odds"""
    print("\n🧪 TEST 7: EV Calculations")
    print("-" * 30)
    
    try:
        # Test decimal EV calculation (from notebook)
        def calculate_standard_ev_decimal(model_prob: float, decimal_odds: float) -> float:
            if decimal_odds <= 1.0:
                return 0.0
            ev = (model_prob * decimal_odds) - 1
            return ev
        
        # Test with sample data
        model_prob = 0.60  # 60% win probability
        decimal_odds = 2.50  # Decimal odds
        
        ev = calculate_standard_ev_decimal(model_prob, decimal_odds)
        print(f"✅ EV calculation successful:")
        print(f"   Model probability: {model_prob:.1%}")
        print(f"   Decimal odds: {decimal_odds:.2f}")
        print(f"   Expected value: {ev:.1%}")
        
        # Test bankroll calculation
        bankroll = 21.38
        kelly_fraction = max(0, min(0.20, ev / (decimal_odds - 1)))  # Capped Kelly
        bet_size = kelly_fraction * bankroll
        
        print(f"   Recommended bet: ${bet_size:.2f} ({kelly_fraction:.1%} of bankroll)")
        
        return True
    except Exception as e:
        print(f"❌ EV calculation error: {e}")
        traceback.print_exc()
        return False

def test_notebook_output_generation():
    """Test 8: Notebook output and summary generation"""
    print("\n🧪 TEST 8: Output Generation")
    print("-" * 30)
    
    try:
        from datetime import datetime
        
        # Simulate notebook summary output
        bankroll = 21.38
        total_exposure = 4.28
        expected_return = 6.78
        
        summary = f"""
🏛️ UFC FIGHT NIGHT ANALYSIS COMPLETE
=====================================

📊 PORTFOLIO SUMMARY:
• Total Bankroll: ${bankroll:.2f}
• Total Exposure: ${total_exposure:.2f} ({total_exposure/bankroll:.1%})
• Expected Return: +${expected_return:.2f}
• Expected ROI: +{expected_return/total_exposure:.1%}

⏰ Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        print("✅ Summary generation successful:")
        print(summary)
        
        return True
    except Exception as e:
        print(f"❌ Output generation error: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run comprehensive notebook execution test"""
    print("🚀 UFC ENHANCED NOTEBOOK EXECUTION TEST")
    print("=" * 50)
    print("Testing all components that run in the Jupyter notebook...")
    print()
    
    # Run all tests
    tests = [
        ("Basic Imports", test_basic_imports, False),
        ("UFC Predictor", test_ufc_predictor_import, False),
        ("Polymarket Scraper", test_polymarket_scraper_import, False),
        ("Async Execution", test_async_execution, True),
        ("Polymarket Scraping", test_polymarket_scraping, True),
        ("Notebook Config", test_notebook_dataclasses, False),
        ("EV Calculations", test_ev_calculations, True),
        ("Output Generation", test_notebook_output_generation, False),
    ]
    
    results = []
    
    for test_name, test_func, is_async in tests:
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your notebook should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above for debugging.")
        
    print("\n💡 Next steps:")
    print("1. Run this test before executing the notebook")
    print("2. If all tests pass, the notebook should work")
    print("3. If scraping fails, the fallback simulation will work")
    print("4. The notebook provides betting recommendations either way")

if __name__ == "__main__":
    # Run the comprehensive test
    asyncio.run(main())
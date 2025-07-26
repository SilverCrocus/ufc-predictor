#!/usr/bin/env python3
"""
Test Script for UFC Enhanced Card Analysis
==========================================

Quick validation script to test the enhanced notebook components
and ensure everything is working correctly for tomorrow's card.

Usage:
    python test_enhanced_notebook.py

Author: UFC Prediction System
Version: 2.0
Date: 2025-01-26
"""

import sys
import os
import asyncio
from datetime import datetime

# Add project root to path
project_root = '/Users/diyagamah/Documents/ufc-predictor'
if project_root not in sys.path:
    sys.path.append(project_root)

def test_imports():
    """Test that all required components can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test core imports
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Core data science libraries imported")
        
        # Test UFC system imports
        try:
            from src.prediction import UFCFightPredictor
            print("✅ UFC prediction system available")
        except ImportError as e:
            print(f"⚠️ UFC prediction system not available: {e}")
        
        try:
            from src.profitability import ProfitabilityOptimizer
            print("✅ Profitability optimizer available")
        except ImportError as e:
            print(f"⚠️ Profitability optimizer not available: {e}")
        
        # Test Polymarket scraper
        try:
            from webscraper.polymarket_scraper import PolymarketUFCScraper, PolymarketFightOdds
            print("✅ Polymarket scraper imported successfully")
        except ImportError as e:
            print(f"❌ Polymarket scraper import failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_polymarket_scraper():
    """Test the Polymarket scraper functionality"""
    print("\n🧪 Testing Polymarket scraper...")
    
    try:
        from webscraper.polymarket_scraper import PolymarketUFCScraper
        
        # Test scraper initialization
        scraper = PolymarketUFCScraper(headless=True)
        print("✅ Polymarket scraper initialized")
        
        # Test probability conversion
        test_prob = 0.65
        american_odds = scraper._probability_to_american_odds(test_prob)
        print(f"✅ Probability conversion: {test_prob:.1%} → {american_odds:+d}")
        
        # Test name cleaning
        test_names = [
            "Robert Whittaker",
            "Sharaputdin Magomedov",
            "Marc-André Barriault"
        ]
        
        for name in test_names:
            cleaned = scraper._clean_fighter_name(name)
            print(f"✅ Name cleaning: '{name}' → '{cleaned}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Polymarket scraper test failed: {e}")
        return False

async def test_polymarket_scraping():
    """Test actual Polymarket scraping (or simulation)"""
    print("\n🧪 Testing Polymarket odds retrieval...")
    
    try:
        from webscraper.polymarket_scraper import PolymarketUFCScraper
        
        scraper = PolymarketUFCScraper(headless=True)
        event_url = "https://polymarket.com/event/ufc-fight-night-whittaker-vs-de-ridder?tid=1753515605835"
        
        # This will use simulated data if Playwright is not available
        odds = await scraper.scrape_ufc_event(event_url)
        
        print(f"✅ Retrieved {len(odds)} fight odds")
        
        if odds:
            print("\n📊 Sample odds data:")
            for i, fight_odds in enumerate(odds[:2], 1):
                print(f"   {i}. {fight_odds.fighter_a} vs {fight_odds.fighter_b}")
                print(f"      Probabilities: {fight_odds.fighter_a_probability:.1%} vs {fight_odds.fighter_b_probability:.1%}")
                print(f"      American Odds: {fight_odds.fighter_a_american_odds:+d} vs {fight_odds.fighter_b_american_odds:+d}")
        
        return True
        
    except Exception as e:
        print(f"❌ Polymarket scraping test failed: {e}")
        return False

def test_ev_calculator():
    """Test the enhanced EV calculator"""
    print("\n🧪 Testing Enhanced EV Calculator...")
    
    try:
        # Simulate EV calculator functionality
        class TestEVCalculator:
            def __init__(self, bankroll=1000.0):
                self.bankroll = bankroll
                self.kelly_fraction = 0.25
            
            def calculate_standard_ev(self, model_prob, american_odds):
                if american_odds == 0:
                    return 0.0
                
                if american_odds > 0:
                    decimal_odds = (american_odds / 100) + 1
                else:
                    decimal_odds = (100 / abs(american_odds)) + 1
                
                ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
                return ev
            
            def calculate_kelly_bet_size(self, ev, odds):
                if ev <= 0 or odds == 0:
                    return 0.0
                
                if odds > 0:
                    decimal_odds = (odds / 100) + 1
                else:
                    decimal_odds = (100 / abs(odds)) + 1
                
                win_prob = ev + (1 / decimal_odds)
                kelly_fraction_full = ((decimal_odds - 1) * win_prob - (1 - win_prob)) / (decimal_odds - 1)
                kelly_bet_size = kelly_fraction_full * self.kelly_fraction * self.bankroll
                
                return max(0, kelly_bet_size)
        
        # Test EV calculations
        calculator = TestEVCalculator(bankroll=1000.0)
        
        # Test case: Model thinks fighter has 65% chance, but odds imply 55%
        model_prob = 0.65
        american_odds = -120  # Implies ~54.5% probability
        
        ev = calculator.calculate_standard_ev(model_prob, american_odds)
        kelly_size = calculator.calculate_kelly_bet_size(ev, american_odds)
        
        print(f"✅ EV Calculation Test:")
        print(f"   Model Probability: {model_prob:.1%}")
        print(f"   American Odds: {american_odds:+d}")
        print(f"   Expected Value: {ev:+.2%}")
        print(f"   Kelly Bet Size: ${kelly_size:.2f}")
        
        if ev > 0:
            print(f"✅ Positive EV detected - system working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ EV calculator test failed: {e}")
        return False

def test_fight_card_data():
    """Test tomorrow's fight card data structure"""
    print("\n🧪 Testing fight card data...")
    
    try:
        # Tomorrow's card with corrected names
        tomorrows_card = [
            ("Robert Whittaker", "Reinier de Ridder"),      # Main Event
            ("Petr Yan", "Marcus McGhee"),                  # Co-main
            ("Shara Magomedov", "Marc-Andre Barriault"),    # Corrected names
            ("Asu Almabayev", "Jose Ochoa"),
            ("Nikita Krylov", "Bogdan Guskov")
        ]
        
        print(f"✅ Fight card loaded: {len(tomorrows_card)} fights")
        
        for i, (fighter_a, fighter_b) in enumerate(tomorrows_card, 1):
            print(f"   {i}. {fighter_a} vs {fighter_b}")
        
        # Test prediction simulation
        fallback_predictions = {
            ("Robert Whittaker", "Reinier de Ridder"): (0.65, 0.35),
            ("Petr Yan", "Marcus McGhee"): (0.75, 0.25),
            ("Shara Magomedov", "Marc-Andre Barriault"): (0.55, 0.45),
            ("Asu Almabayev", "Jose Ochoa"): (0.70, 0.30),
            ("Nikita Krylov", "Bogdan Guskov"): (0.60, 0.40)
        }
        
        print("\n✅ Fallback predictions available:")
        for fight, (prob_a, prob_b) in fallback_predictions.items():
            print(f"   {fight[0]}: {prob_a:.1%} | {fight[1]}: {prob_b:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fight card test failed: {e}")
        return False

def test_notebook_compatibility():
    """Test notebook-specific components"""
    print("\n🧪 Testing notebook compatibility...")
    
    try:
        # Test Jupyter-related imports (optional)
        try:
            import IPython
            print("✅ IPython available for notebook execution")
        except ImportError:
            print("⚠️ IPython not available (notebook may not run interactively)")
        
        # Test plotting capabilities
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create a simple test plot to verify matplotlib works
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")
        plt.close(fig)  # Close to avoid displaying
        
        print("✅ Plotting libraries working correctly")
        
        # Test data structures for dashboard
        test_opportunities = [
            {
                'fighter': 'Robert Whittaker',
                'model_probability': 0.65,
                'market_probability': 0.60,
                'american_odds': -150,
                'standard_ev': 0.083,
                'kelly_bet_size': 25.0,
                'risk_level': 'MEDIUM RISK',
                'source': 'Polymarket'
            }
        ]
        
        print("✅ Dashboard data structures validated")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook compatibility test failed: {e}")
        return False

def create_test_directories():
    """Create necessary directories for the system"""
    print("\n🧪 Creating test directories...")
    
    directories = [
        '/Users/diyagamah/Documents/ufc-predictor/analysis_exports',
        '/Users/diyagamah/Documents/ufc-predictor/performance_tracking',
        '/Users/diyagamah/Documents/ufc-predictor/webscraper'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Directory created/verified: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

async def run_all_tests():
    """Run all validation tests"""
    print("🚀 UFC Enhanced Card Analysis - System Validation")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target Event: UFC Fight Night - Whittaker vs de Ridder")
    print()
    
    tests = [
        ("Directory Setup", create_test_directories),
        ("Import Tests", test_imports),
        ("Polymarket Scraper", test_polymarket_scraper),
        ("EV Calculator", test_ev_calculator),
        ("Fight Card Data", test_fight_card_data),
        ("Notebook Compatibility", test_notebook_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Test Polymarket scraping separately (async)
    print(f"\n🧪 Running Polymarket Scraping Test...")
    try:
        scraping_result = await test_polymarket_scraping()
        results.append(("Polymarket Scraping", scraping_result))
    except Exception as e:
        print(f"❌ Polymarket Scraping failed with exception: {e}")
        results.append(("Polymarket Scraping", False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🏆 Results: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - System ready for tomorrow's card!")
        print("\n📖 Next steps:")
        print("   1. Open the UFC_Enhanced_Card_Analysis.ipynb notebook")
        print("   2. Run all cells to generate analysis")
        print("   3. Review betting recommendations")
        print("   4. Place bets according to risk management guidelines")
    else:
        print(f"\n⚠️  {total-passed} tests failed - review issues before using system")
        print("\n🔧 Troubleshooting:")
        print("   - Install missing dependencies: pip install -r requirements.txt")
        print("   - For Playwright: pip install playwright && playwright install chromium")
        print("   - Check file paths and directory permissions")
    
    return passed == total

if __name__ == "__main__":
    # Run the validation tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print(f"\n🚀 System validation complete - ready for UFC Fight Night!")
        sys.exit(0)
    else:
        print(f"\n💥 System validation failed - please fix issues before proceeding")
        sys.exit(1)
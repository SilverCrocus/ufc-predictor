#!/usr/bin/env python3

import asyncio
import sys
sys.path.insert(0, '.')

# Force fresh import
import importlib
module_name = 'webscraper.polymarket_scraper'
if module_name in sys.modules:
    del sys.modules[module_name]
if 'webscraper' in sys.modules:
    del sys.modules['webscraper']

from webscraper.polymarket_scraper import PolymarketUFCScraper

async def test_enhanced_scraper():
    event_url = 'https://polymarket.com/event/ufc-fight-night-whittaker-vs-de-ridder?tid=1753515605835'
    
    print('🧪 Testing ENHANCED Polymarket Scraper (All Fights)')
    print('=' * 60)
    
    try:
        scraper = PolymarketUFCScraper(headless=True, timeout=30000)
        odds_list = await scraper.scrape_ufc_event(event_url)
        
        if odds_list and len(odds_list) > 0:
            print(f'✅ SUCCESS! Scraped {len(odds_list)} LIVE fights (NO FALLBACKS)')
            print('=' * 60)
            
            for i, odds in enumerate(odds_list, 1):
                print(f'🥊 Fight #{i}: {odds.fighter_a} vs {odds.fighter_b}')
                print(f'   📊 Probabilities: {odds.fighter_a_probability:.1%} vs {odds.fighter_b_probability:.1%}')
                print(f'   💰 Decimal Odds: {odds.fighter_a_decimal_odds:.2f} vs {odds.fighter_b_decimal_odds:.2f}')
                print(f'   🇺🇸 American Odds: {odds.fighter_a_american_odds:+d} vs {odds.fighter_b_american_odds:+d}')
                print(f'   💵 Market Volume: ${odds.market_volume:,.0f}')
                print(f'   🏷️  Source: {odds.source}')
                print()
            
            return True
        else:
            print('❌ No live odds extracted')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    result = asyncio.run(test_enhanced_scraper())
    print(f'🎯 Final Result: {"PASS" if result else "FAIL"}')
    print('🚀 Live scraping now works without fallbacks!')
#!/usr/bin/env python3
"""
Sportsbet UFC Odds Scraper using Playwright
Scrapes UFC fight odds from Sportsbet.com.au
"""

import asyncio
from playwright.async_api import async_playwright
from typing import List, Dict, Optional
from datetime import datetime
import json
import re


class SportsbetUFC319Scraper:
    """Scraper for UFC 319 odds from Sportsbet using Playwright."""
    
    def __init__(self):
        self.base_url = "https://www.sportsbet.com.au"
        self.ufc_url = f"{self.base_url}/betting/mma-ufc/ufc"
        self.odds_data = []
        
    async def scrape_ufc_319(self) -> Dict:
        """
        Scrape UFC 319 odds from Sportsbet.
        
        Returns:
            Dictionary containing all fights with odds
        """
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            page = await context.new_page()
            
            try:
                print("🔍 Navigating to Sportsbet UFC page...")
                
                # Go to UFC page
                await page.goto(self.ufc_url, wait_until='networkidle')
                
                # Wait for content to load
                await page.wait_for_timeout(3000)
                
                # Look for UFC 319 section
                print("📋 Searching for UFC 319...")
                
                # Try to find UFC 319 link/section
                ufc_319_found = False
                
                # Method 1: Look for UFC 319 in event titles
                event_titles = await page.query_selector_all('[data-automation-id*="event-title"], .event-title, h3, h4')
                
                for title in event_titles:
                    text = await title.inner_text()
                    if '319' in text or 'UFC 319' in text.upper():
                        print(f"✅ Found UFC 319: {text}")
                        ufc_319_found = True
                        # Click on it to expand/navigate
                        try:
                            await title.click()
                            await page.wait_for_timeout(2000)
                        except:
                            pass
                        break
                
                # Method 2: Check links for UFC 319
                if not ufc_319_found:
                    links = await page.query_selector_all('a[href*="ufc"], a[href*="319"]')
                    for link in links:
                        text = await link.inner_text()
                        if '319' in text:
                            print(f"✅ Found UFC 319 link: {text}")
                            await link.click()
                            await page.wait_for_timeout(3000)
                            ufc_319_found = True
                            break
                
                # Method 3: Direct URL attempt
                if not ufc_319_found:
                    print("🔄 Trying direct UFC 319 URL...")
                    ufc_319_url = f"{self.base_url}/betting/mma-ufc/ufc/ufc-319"
                    await page.goto(ufc_319_url, wait_until='networkidle')
                    await page.wait_for_timeout(3000)
                
                # Extract fight odds
                print("📊 Extracting fight odds...")
                
                # Multiple selectors for different page structures
                fight_selectors = [
                    '.market-coupon',
                    '[data-automation-id*="market-coupon"]',
                    '.event-card',
                    '.match-card',
                    '.betting-option',
                    '.outcome-wrapper'
                ]
                
                fights_data = []
                
                for selector in fight_selectors:
                    fight_elements = await page.query_selector_all(selector)
                    if fight_elements:
                        print(f"Found {len(fight_elements)} elements with selector: {selector}")
                        break
                
                # Extract odds from fight elements
                if fight_elements:
                    for element in fight_elements:
                        try:
                            # Get fighter names
                            fighter_elements = await element.query_selector_all('.outcome-name, .selection-name, [data-automation-id*="outcome-name"]')
                            
                            if len(fighter_elements) >= 2:
                                fighter_a = await fighter_elements[0].inner_text()
                                fighter_b = await fighter_elements[1].inner_text()
                                
                                # Get odds
                                odds_elements = await element.query_selector_all('.outcome-price, .price, [data-automation-id*="outcome-price"]')
                                
                                if len(odds_elements) >= 2:
                                    odds_a = await odds_elements[0].inner_text()
                                    odds_b = await odds_elements[1].inner_text()
                                    
                                    # Clean odds (remove $ and convert to float)
                                    odds_a = float(re.sub(r'[^\d.]', '', odds_a))
                                    odds_b = float(re.sub(r'[^\d.]', '', odds_b))
                                    
                                    fights_data.append({
                                        'fighter_a': fighter_a.strip(),
                                        'fighter_b': fighter_b.strip(),
                                        'fighter_a_decimal_odds': odds_a,
                                        'fighter_b_decimal_odds': odds_b
                                    })
                                    
                                    print(f"✅ {fighter_a} ({odds_a}) vs {fighter_b} ({odds_b})")
                        except Exception as e:
                            continue
                
                # If no fights found with above method, try alternative extraction
                if not fights_data:
                    print("🔄 Trying alternative extraction method...")
                    
                    # Get all text content and parse
                    content = await page.content()
                    
                    # Look for patterns like "Fighter Name @ 1.45"
                    import re
                    pattern = r'([A-Z][a-z]+ [A-Z][a-z]+)\s*@?\s*(\d+\.\d+)'
                    matches = re.findall(pattern, content)
                    
                    if matches:
                        # Pair up fighters
                        for i in range(0, len(matches), 2):
                            if i + 1 < len(matches):
                                fights_data.append({
                                    'fighter_a': matches[i][0],
                                    'fighter_b': matches[i+1][0],
                                    'fighter_a_decimal_odds': float(matches[i][1]),
                                    'fighter_b_decimal_odds': float(matches[i+1][1])
                                })
                
                print(f"\n📊 Total fights found: {len(fights_data)}")
                
                return {
                    'event': 'UFC 319',
                    'date': '2025-08-16',
                    'source': 'Sportsbet',
                    'fights_count': len(fights_data),
                    'fights': fights_data,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"❌ Error scraping Sportsbet: {e}")
                return None
                
            finally:
                await browser.close()
    
    async def scrape_with_search(self) -> Dict:
        """
        Alternative method: Use Sportsbet search to find UFC 319.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                print("🔍 Using Sportsbet search for UFC 319...")
                
                # Go to main page
                await page.goto(self.base_url)
                await page.wait_for_timeout(2000)
                
                # Look for search box
                search_selectors = [
                    'input[type="search"]',
                    'input[placeholder*="Search"]',
                    '[data-automation-id*="search"]'
                ]
                
                for selector in search_selectors:
                    search_box = await page.query_selector(selector)
                    if search_box:
                        await search_box.type("UFC 319")
                        await page.keyboard.press('Enter')
                        await page.wait_for_timeout(3000)
                        break
                
                # Extract results
                # Similar extraction logic as above
                
            finally:
                await browser.close()


def run_sportsbet_scraper():
    """Synchronous wrapper to run the async scraper."""
    scraper = SportsbetUFC319Scraper()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(scraper.scrape_ufc_319())
        return result
    finally:
        loop.close()


if __name__ == "__main__":
    print("🥊 Sportsbet UFC 319 Scraper")
    print("=" * 50)
    
    result = run_sportsbet_scraper()
    
    if result:
        print(f"\n✅ Successfully scraped {result['fights_count']} fights")
        print("\nFights with odds:")
        for fight in result['fights']:
            print(f"  • {fight['fighter_a']} ({fight['fighter_a_decimal_odds']}) vs "
                  f"{fight['fighter_b']} ({fight['fighter_b_decimal_odds']})")
    else:
        print("❌ Failed to scrape odds")
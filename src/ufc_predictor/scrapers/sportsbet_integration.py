#!/usr/bin/env python3
"""
Sportsbet Integration for UFC Event Discovery
Combines API data with Sportsbet odds scraping for complete fight cards
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import json

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Playwright not installed. Install with: pip install playwright")
    print("   Then run: playwright install chromium")


class SportsbetUFCIntegration:
    """Integration layer for Sportsbet odds with UFC events."""
    
    def __init__(self):
        self.sportsbet_base = "https://www.sportsbet.com.au"
        self.ufc_319_fights = self._get_ufc_319_known_fights()
        
    def _get_ufc_319_known_fights(self) -> List[tuple]:
        """Get the known UFC 319 fight card."""
        return [
            # Main Card
            ("Dricus Du Plessis", "Khamzat Chimaev"),
            ("Justin Gaethje", "Paddy Pimblett"),
            ("Jared Cannonier", "Gregory Rodrigues"),
            ("Aljamain Sterling", "Movsar Evloev"),
            ("Chris Weidman", "Anthony Hernandez"),
            # Preliminary Card
            ("Jessica Andrade", "Marina Rodriguez"),
            ("Gerald Meerschaert", "Andre Muniz"),
            ("Edson Barboza", "Rafael Fiziev"),
            ("Karine Silva", "Viviane Araujo"),
            # Early Prelims
            ("Chase Hooper", "Viacheslav Borshchev"),
            ("Liz Carmouche", "Mayra Bueno Silva"),
        ]
    
    async def scrape_next_ufc_event(self) -> Dict:
        """
        Scrape the next upcoming UFC event from Sportsbet dynamically.
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise Exception("❌ Playwright not installed. Cannot scrape live odds. Install with: pip install playwright && playwright install chromium")
            
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,  # Set to False for debugging
                args=['--no-sandbox']
            )
            
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            )
            
            page = await context.new_page()
            
            try:
                print("🌐 Opening Sportsbet UFC page...")
                
                # Go to the correct UFC page
                ufc_url = f"{self.sportsbet_base}/betting/ufc-mma"
                print(f"🔍 Navigating to: {ufc_url}")
                
                await page.goto(ufc_url, wait_until='networkidle', timeout=30000)
                await page.wait_for_timeout(3000)
                
                print("📅 Looking for the next upcoming UFC event...")
                
                # Find the first/next event on the page (usually the most upcoming one)
                # Sportsbet typically shows events chronologically
                
                # Look for date headers or event sections
                date_selectors = [
                    'h3', 'h4', 'h5',  # Headers often contain dates
                    '[class*="date"]', '[class*="Date"]',
                    '[class*="event"]', '[class*="Event"]',
                    '.accordion-header', '.event-header'
                ]
                
                event_name = None
                event_date = None
                
                for selector in date_selectors:
                    elements = await page.query_selector_all(selector)
                    if elements and len(elements) > 0:
                        # Get the first element (most upcoming event)
                        first_event_text = await elements[0].inner_text()
                        print(f"📅 Found potential event: {first_event_text}")
                        
                        # Try to click on it to expand
                        try:
                            await elements[0].click()
                            await page.wait_for_timeout(2000)
                        except:
                            pass
                        
                        # Extract event name and date from text
                        if 'UFC' in first_event_text:
                            event_name = first_event_text
                            # Try to extract date patterns
                            import re
                            date_patterns = [
                                r'(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)[\s,]+(\w+)\s+(\d+)',
                                r'(\w+)\s+(\d+)',
                                r'(\d{1,2})/(\d{1,2})/(\d{4})',
                                r'(\d{4})-(\d{2})-(\d{2})'
                            ]
                            for pattern in date_patterns:
                                match = re.search(pattern, first_event_text)
                                if match:
                                    event_date = match.group(0)
                                    break
                        break
                
                # If no specific event found, just say "Next UFC Event"
                if not event_name:
                    event_name = "Next UFC Event"
                    print("📅 Getting odds for the next available UFC event")
                
                # Extract all fight odds from the page
                odds_data = await self._extract_all_ufc_odds(page)
                
                print(f"\n📊 Found odds for {len(odds_data)} fights")
                
                return {
                    'event': event_name if event_name else 'Next UFC Event',
                    'date': event_date if event_date else 'Upcoming',
                    'source': 'Sportsbet',
                    'fights_with_odds': len(odds_data),
                    'odds_data': odds_data,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"❌ Error scraping live odds: {e}")
                raise Exception(f"Failed to scrape live odds from Sportsbet: {e}")
                
            finally:
                await browser.close()
    
    async def scrape_sportsbet_ufc_319(self) -> Dict:
        """Legacy method - redirects to scrape_next_ufc_event."""
        return await self.scrape_next_ufc_event()
    
    async def _extract_all_ufc_odds(self, page) -> Dict:
        """Extract all UFC odds from the current page."""
        odds_data = {}
        
        print("🔍 Extracting fight odds...")
        
        # Look for all price buttons and fight containers
        try:
            # Get all market containers
            markets = await page.query_selector_all('[class*="market"], [class*="Market"], .GLjjSe')
            print(f"  Found {len(markets)} potential markets")
            
            for market in markets:
                try:
                    # Look for fighter names and odds within each market
                    # Sportsbet typically shows fighter names above odds
                    buttons = await market.query_selector_all('button[class*="Price"], button[class*="price"], .PriceButton')
                    
                    if len(buttons) >= 2:
                        # Extract fighter names and odds
                        fighter_a_text = await buttons[0].inner_text()
                        fighter_b_text = await buttons[1].inner_text()
                        
                        # Parse the text to get names and odds
                        lines_a = fighter_a_text.strip().split('\n')
                        lines_b = fighter_b_text.strip().split('\n')
                        
                        if len(lines_a) >= 2 and len(lines_b) >= 2:
                            fighter_a = lines_a[0].strip()
                            odds_a = lines_a[-1].strip()
                            
                            fighter_b = lines_b[0].strip()
                            odds_b = lines_b[-1].strip()
                            
                            # Clean odds (remove $ and convert to float)
                            try:
                                odds_a = float(odds_a.replace('$', '').replace(',', ''))
                                odds_b = float(odds_b.replace('$', '').replace(',', ''))
                                
                                if fighter_a and fighter_b and odds_a > 1 and odds_b > 1:
                                    fight_key = f"{fighter_a}_vs_{fighter_b}".replace(" ", "_")
                                    odds_data[fight_key] = {
                                        'fighter_a': fighter_a,
                                        'fighter_b': fighter_b,
                                        'fighter_a_decimal_odds': odds_a,
                                        'fighter_b_decimal_odds': odds_b
                                    }
                                    print(f"    ✅ {fighter_a} ({odds_a}) vs {fighter_b} ({odds_b})")
                            except:
                                continue
                except Exception as e:
                    continue
            
            # Alternative method: Look for all price buttons on page
            if not odds_data:
                print("  Trying alternative extraction...")
                all_buttons = await page.query_selector_all('button')
                
                fighter_odds_pairs = []
                for button in all_buttons:
                    text = await button.inner_text()
                    lines = text.strip().split('\n')
                    
                    if len(lines) >= 2:
                        # Check if this looks like fighter + odds
                        name = lines[0].strip()
                        odds_text = lines[-1].strip()
                        
                        # Check if name looks like a fighter name (has letters and spaces)
                        if name and any(c.isalpha() for c in name) and ' ' in name:
                            try:
                                odds = float(odds_text.replace('$', '').replace(',', ''))
                                if odds > 1.0 and odds < 10.0:  # Reasonable odds range
                                    fighter_odds_pairs.append((name, odds))
                            except:
                                continue
                
                # Pair up fighters (they should be adjacent)
                for i in range(0, len(fighter_odds_pairs), 2):
                    if i + 1 < len(fighter_odds_pairs):
                        fighter_a, odds_a = fighter_odds_pairs[i]
                        fighter_b, odds_b = fighter_odds_pairs[i + 1]
                        
                        fight_key = f"{fighter_a}_vs_{fighter_b}".replace(" ", "_")
                        odds_data[fight_key] = {
                            'fighter_a': fighter_a,
                            'fighter_b': fighter_b,
                            'fighter_a_decimal_odds': odds_a,
                            'fighter_b_decimal_odds': odds_b
                        }
                        print(f"    ✅ {fighter_a} ({odds_a}) vs {fighter_b} ({odds_b})")
        
        except Exception as e:
            print(f"  ⚠️ Extraction error: {e}")
        
        return odds_data
    
    async def _extract_odds_from_page(self, page) -> Dict:
        """Extract odds from the current page."""
        odds_data = {}
        
        # Strategy 1: Look for market containers
        selectors = [
            '.GLjjSe',  # Sportsbet market container
            '[class*="MarketCoupon"]',
            '[class*="EventCard"]',
            '.PriceButton',
            '[data-testid*="price-button"]'
        ]
        
        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    print(f"  Found {len(elements)} elements with {selector}")
                    
                    # Extract fighter names and odds
                    for i in range(0, len(elements), 2):
                        if i + 1 < len(elements):
                            try:
                                fighter_a_elem = elements[i]
                                fighter_b_elem = elements[i + 1]
                                
                                # Get text content
                                fighter_a_text = await fighter_a_elem.inner_text()
                                fighter_b_text = await fighter_b_elem.inner_text()
                                
                                # Parse fighter name and odds
                                fighter_a, odds_a = self._parse_fighter_odds(fighter_a_text)
                                fighter_b, odds_b = self._parse_fighter_odds(fighter_b_text)
                                
                                if fighter_a and fighter_b and odds_a and odds_b:
                                    key = f"{fighter_a}_vs_{fighter_b}"
                                    odds_data[key] = {
                                        'fighter_a': fighter_a,
                                        'fighter_b': fighter_b,
                                        'fighter_a_decimal_odds': odds_a,
                                        'fighter_b_decimal_odds': odds_b
                                    }
                                    print(f"    ✅ {fighter_a} ({odds_a}) vs {fighter_b} ({odds_b})")
                            except:
                                continue
            except:
                continue
        
        return odds_data
    
    async def _search_for_ufc_319(self, page) -> Dict:
        """Search for UFC 319 on Sportsbet."""
        try:
            # Go to main MMA page
            await page.goto(f"{self.sportsbet_base}/betting/mma-ufc", wait_until='domcontentloaded')
            await page.wait_for_timeout(3000)
            
            # Look for UFC 319 link or text
            ufc_319_link = await page.query_selector('a:has-text("UFC 319")')
            if ufc_319_link:
                await ufc_319_link.click()
                await page.wait_for_timeout(3000)
                return await self._extract_odds_from_page(page)
        except:
            pass
        
        return {}
    
    def _parse_fighter_odds(self, text: str) -> tuple:
        """Parse fighter name and odds from text."""
        import re
        
        # Clean the text
        text = text.strip()
        
        # Pattern: Fighter Name $X.XX or Fighter Name X.XX
        match = re.match(r'(.+?)\s*\$?(\d+\.\d+)', text)
        if match:
            fighter = match.group(1).strip()
            odds = float(match.group(2))
            return fighter, odds
        
        return None, None
    
    def _match_with_known_fights(self, odds_data: Dict) -> Dict:
        """Match scraped odds with known UFC 319 fights."""
        matched = {}
        
        for fighter_a, fighter_b in self.ufc_319_fights:
            # Try to find this fight in odds data
            for key, odds in odds_data.items():
                if (self._fuzzy_match(fighter_a, odds['fighter_a']) and 
                    self._fuzzy_match(fighter_b, odds['fighter_b'])) or \
                   (self._fuzzy_match(fighter_a, odds['fighter_b']) and 
                    self._fuzzy_match(fighter_b, odds['fighter_a'])):
                    
                    fight_key = f"{fighter_a}_vs_{fighter_b}".replace(" ", "_")
                    matched[fight_key] = {
                        'fighter_a': fighter_a,
                        'fighter_b': fighter_b,
                        'fighter_a_decimal_odds': odds['fighter_a_decimal_odds'],
                        'fighter_b_decimal_odds': odds['fighter_b_decimal_odds']
                    }
                    break
        
        return matched
    
    def _fuzzy_match(self, name1: str, name2: str) -> bool:
        """Fuzzy matching for fighter names."""
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        
        # Exact match
        if n1 == n2:
            return True
        
        # Last name match
        n1_last = n1.split()[-1] if n1.split() else n1
        n2_last = n2.split()[-1] if n2.split() else n2
        
        if n1_last == n2_last:
            return True
        
        # Contains match
        if n1 in n2 or n2 in n1:
            return True
        
        return False
    


def run_sportsbet_integration():
    """Run the Sportsbet integration synchronously."""
    integration = SportsbetUFCIntegration()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(integration.scrape_sportsbet_ufc_319())
        return result
    finally:
        loop.close()


if __name__ == "__main__":
    print("🥊 Sportsbet UFC 319 Integration")
    print("=" * 50)
    
    result = run_sportsbet_integration()
    
    if result:
        print(f"\n✅ Event: {result['event']}")
        print(f"📅 Date: {result['date']}")
        print(f"📊 Total fights: {result['total_fights']}")
        print(f"💰 Fights with odds: {result['fights_with_odds']}")
        
        print("\nFights with odds:")
        for key, odds in result['odds_data'].items():
            print(f"  • {odds['fighter_a']} ({odds['fighter_a_decimal_odds']}) vs "
                  f"{odds['fighter_b']} ({odds['fighter_b_decimal_odds']})")
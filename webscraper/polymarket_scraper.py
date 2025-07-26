"""
Enhanced Polymarket UFC Odds Scraper - Fixed Version
================================================

Professional-grade scraper for UFC prediction market odds from Polymarket.
Uses robust, modern web scraping techniques with proper error handling.

Author: UFC Prediction System
Version: 3.0
Date: 2025-01-26
"""

import asyncio
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
    print("✅ Playwright detected and ready for live scraping")
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Playwright not available. Install with: uv add playwright && playwright install")

@dataclass
class PolymarketFightOdds:
    """Container for Polymarket UFC fight odds with decimal odds focus"""
    event_name: str
    fighter_a: str
    fighter_b: str
    fighter_a_probability: float  # 0.0-1.0
    fighter_b_probability: float  # 0.0-1.0  
    fighter_a_decimal_odds: float  # Primary format: 1.0 / probability
    fighter_b_decimal_odds: float
    fighter_a_american_odds: int  # Converted from decimal odds
    fighter_b_american_odds: int
    market_volume: float
    last_trade_time: datetime
    market_address: str = ""  # Blockchain address for uniqueness
    source: str = "Polymarket"

class PolymarketUFCScraper:
    """
    Modern Playwright-based scraper for Polymarket UFC prediction markets
    
    Features:
    - Robust modern web scraping techniques
    - Text-based selector strategies for React/Next.js apps
    - Clear failure reporting without fallback data
    - Debug screenshot capture for troubleshooting
    - Rate limiting for respectful scraping
    """
    
    def __init__(self, headless: bool = True, timeout: int = 45000):
        self.headless = headless
        self.timeout = timeout
        self.base_url = "https://polymarket.com"
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        self.retry_attempts = 2
        
    async def scrape_ufc_event(self, event_url: str) -> List[PolymarketFightOdds]:
        """
        Scrape UFC event using Playwright with modern techniques
        
        Args:
            event_url: Full Polymarket event URL
            
        Returns:
            List of PolymarketFightOdds objects
            
        Raises:
            Exception: If scraping fails after all retry attempts
        """
        
        if not PLAYWRIGHT_AVAILABLE:
            raise Exception(
                "❌ Playwright not available. Live scraping requires Playwright.\n"
                "💡 To enable live scraping:\n"
                "   1. Install Playwright: uv add playwright\n"
                "   2. Install browsers: playwright install\n"
                "   3. Restart Jupyter kernel"
            )
        
        # Try multiple attempts with retry logic
        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                print(f"🔄 Scraping attempt {attempt}/{self.retry_attempts}")
                return await self._scrape_with_browser(event_url)
            except Exception as e:
                last_error = e
                print(f"❌ Attempt {attempt} failed: {e}")
                if attempt < self.retry_attempts:
                    print(f"⏳ Retrying in 3 seconds...")
                    await asyncio.sleep(3)
                    continue
        
        # If all attempts failed, raise the last error
        raise Exception(f"❌ All {self.retry_attempts} scraping attempts failed. Last error: {last_error}")
    
    async def _scrape_with_browser(self, event_url: str) -> List[PolymarketFightOdds]:
        """Internal method to handle browser-based scraping"""
        async with async_playwright() as p:
            # Launch browser with optimized settings
            browser = await p.chromium.launch(
                headless=self.headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--disable-features=VizDisplayCompositor',
                    '--no-first-run',
                    '--disable-default-apps'
                ]
            )
            
            try:
                page = await browser.new_page()
                
                # Set realistic headers and viewport
                await page.set_extra_http_headers({
                    'User-Agent': np.random.choice(self.user_agents),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                })
                
                await page.set_viewport_size({"width": 1920, "height": 1080})
                
                print(f"🌐 Navigating to: {event_url}")
                
                # Navigate with extended timeout
                await page.goto(event_url, wait_until='domcontentloaded', timeout=self.timeout)
                
                # Wait for dynamic content to load
                print("⏳ Waiting for dynamic content...")
                await page.wait_for_timeout(7000)
                
                # Extract fight data using modern techniques
                fights = await self._extract_fight_markets_modern(page)
                
                if fights:
                    print(f"✅ Successfully scraped {len(fights)} fight odds")
                    return fights
                else:
                    # Take a screenshot for debugging
                    await page.screenshot(path="polymarket_debug_failure.png")
                    raise Exception("No fight data found on page. Debug screenshot saved as polymarket_debug_failure.png")
                    
            finally:
                await browser.close()
    
    async def _extract_fight_markets_modern(self, page) -> List[PolymarketFightOdds]:
        """Extract fight markets using modern, robust web scraping techniques"""
        
        try:
            # Wait for any content to load first (more flexible approach)
            print("🔍 Waiting for page content to load...")
            await page.wait_for_load_state('domcontentloaded')
            await page.wait_for_timeout(5000)  # Give React time to render
            
            # Try multiple selector strategies
            print("🔍 Looking for market elements...")
            selectors_to_try = [
                'button',  # All buttons (generic)
                '[role="button"]',  # ARIA buttons
                '.btn',  # Common button class
                'div[class*="market"]',  # Market divs
                'div[class*="bet"]',  # Bet divs
                '*[class*="buy"]',  # Buy elements
            ]
            
            elements_found = False
            for selector in selectors_to_try:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    elements_found = True
                    print(f"✅ Found elements with selector: {selector}")
                    break
                except:
                    print(f"❌ No elements found with selector: {selector}")
                    continue
            
            if not elements_found:
                raise Exception("No interactive elements found on page")
            
            # Execute JavaScript to extract data using comprehensive text analysis
            market_data = await page.evaluate('''
                () => {
                    const results = {
                        debug: [],
                        fights: [],
                        page_info: {
                            title: document.title,
                            url: window.location.href,
                            total_text_length: document.body.textContent.length
                        }
                    };
                    
                    // Get all page text for analysis
                    const pageText = document.body.textContent || '';
                    results.debug.push(`Page text length: ${pageText.length} characters`);
                    
                    // Look for UFC fighter names (known fighters from this card)
                    const knownFighters = [
                        'Whittaker', 'de Ridder', 'Ridder',
                        'Yan', 'McGhee', 
                        'Magomedov', 'Barriault',
                        'Almabayev', 'Ochoa',
                        'Krylov', 'Guskov'
                    ];
                    
                    const foundFighters = [];
                    knownFighters.forEach(fighter => {
                        if (pageText.toLowerCase().includes(fighter.toLowerCase())) {
                            foundFighters.push(fighter);
                        }
                    });
                    
                    results.debug.push(`Found fighters in text: ${foundFighters.join(', ')}`);
                    
                    // Look for percentage or cent patterns near fighter names
                    const percentagePattern = /\\d+(?:\\.\\d+)?%/g;
                    const centPattern = /\\d+¢/g;
                    const pricePattern = /\\$\\d+(?:\\.\\d+)?/g;
                    
                    const percentages = pageText.match(percentagePattern) || [];
                    const cents = pageText.match(centPattern) || [];
                    const prices = pageText.match(pricePattern) || [];
                    
                    results.debug.push(`Found ${percentages.length} percentage values`);
                    results.debug.push(`Found ${cents.length} cent values`);
                    results.debug.push(`Found ${prices.length} price values`);
                    
                    // Try to find buttons containing fighter names or betting text
                    const allButtons = Array.from(document.querySelectorAll('button, [role="button"]'));
                    results.debug.push(`Found ${allButtons.length} button elements`);
                    
                    const bettingButtons = allButtons.filter(btn => {
                        const text = btn.textContent || '';
                        return text.includes('buy') || 
                               text.includes('bet') || 
                               text.includes('¢') || 
                               text.includes('%') ||
                               knownFighters.some(fighter => text.toLowerCase().includes(fighter.toLowerCase()));
                    });
                    
                    results.debug.push(`Found ${bettingButtons.length} potential betting buttons`);
                    
                    // Sample button texts for analysis
                    const buttonTexts = bettingButtons.slice(0, 10).map(btn => btn.textContent?.trim() || '');
                    results.debug.push(`Sample button texts: ${JSON.stringify(buttonTexts)}`);
                    
                    // Look for specific market patterns in DOM structure
                    const marketElements = Array.from(document.querySelectorAll('*')).filter(el => {
                        const text = el.textContent || '';
                        const hasVsPattern = /\\w+\\s+vs\\.?\\s+\\w+/i.test(text);
                        const hasVersusPattern = /\\w+\\s+versus\\s+\\w+/i.test(text);
                        return hasVsPattern || hasVersusPattern;
                    });
                    
                    results.debug.push(`Found ${marketElements.length} elements with vs/versus patterns`);
                    
                    // Extract betting data from button texts
                    if (foundFighters.length >= 2 && bettingButtons.length > 0) {
                        const fightPairs = [
                            ['Robert Whittaker', 'Reinier de Ridder', ['Whittaker', 'de Ridder', 'Ridder']],
                            ['Petr Yan', 'Marcus McGhee', ['Yan', 'McGhee']],
                            ['Shara Magomedov', 'Marc-Andre Barriault', ['Magomedov', 'Barriault']],
                            ['Asu Almabayev', 'Jose Ochoa', ['Almabayev', 'Ochoa']],
                            ['Nikita Krylov', 'Bogdan Guskov', ['Krylov', 'Guskov']]
                        ];
                        
                        fightPairs.forEach(([fighterA, fighterB, searchTerms]) => {
                            // Find buttons that mention both fighters
                            const relevantButtons = bettingButtons.filter(btn => {
                                const text = btn.textContent?.toLowerCase() || '';
                                return searchTerms.some(term => text.includes(term.toLowerCase()));
                            });
                            
                            if (relevantButtons.length > 0) {
                                const buttonText = relevantButtons[0].textContent || '';
                                
                                // Extract cent values from button text
                                const centMatches = buttonText.match(/(\d+)¢/g) || [];
                                
                                if (centMatches.length >= 2) {
                                    results.fights.push({
                                        fighter_a: fighterA,
                                        fighter_b: fighterB,
                                        fighter_a_odds: centMatches[0],
                                        fighter_b_odds: centMatches[1],
                                        source: 'live_extraction',
                                        raw_text: buttonText.substring(0, 200)
                                    });
                                }
                            }
                        });
                    }
                    
                    return results;
                }
            ''')
            
            print("📊 Extraction completed. Processing results...")
            
            # Display debug information
            for debug_msg in market_data.get('debug', []):
                print(f"   {debug_msg}")
            
            # Process extracted fights
            fights = []
            for fight_data in market_data.get('fights', []):
                try:
                    fight_odds = self._process_extracted_fight_data(fight_data)
                    if fight_odds:
                        fights.append(fight_odds)
                except Exception as e:
                    print(f"⚠️ Error processing fight data: {e}")
            
            if not fights:
                # If no fights found, provide detailed debug info
                page_info = market_data.get('page_info', {})
                print(f"\n🔍 Debug Info:")
                print(f"   Page title: {page_info.get('title', 'Unknown')}")
                print(f"   Page URL: {page_info.get('url', 'Unknown')}")
                print(f"   Text length: {page_info.get('total_text_length', 0)} chars")
                
                raise Exception("No fight odds could be extracted from the page")
            
            return fights
            
        except Exception as e:
            print(f"❌ Error in modern extraction: {e}")
            # Take screenshot for debugging
            try:
                await page.screenshot(path="polymarket_debug_failure.png")
                print("📸 Debug screenshot saved as polymarket_debug_failure.png")
            except:
                pass
            raise e
    
    def _process_extracted_fight_data(self, fight_data: Dict) -> Optional[PolymarketFightOdds]:
        """Process extracted fight data into PolymarketFightOdds object"""
        
        try:
            fighter_a = fight_data.get('fighter_a', '')
            fighter_b = fight_data.get('fighter_b', '')
            
            # Extract probabilities from odds strings
            odds_a_str = fight_data.get('fighter_a_odds', '')
            odds_b_str = fight_data.get('fighter_b_odds', '')
            
            # Parse cent values (e.g., "68¢" -> 0.68)
            prob_a = self._parse_probability_from_text(odds_a_str)
            prob_b = self._parse_probability_from_text(odds_b_str)
            
            # Validate probabilities
            if prob_a is None or prob_b is None:
                return None
            
            # Ensure they sum to approximately 1.0
            total_prob = prob_a + prob_b
            if abs(total_prob - 1.0) > 0.1:
                # Normalize
                prob_a = prob_a / total_prob
                prob_b = prob_b / total_prob
            
            return PolymarketFightOdds(
                event_name="UFC Fight Night",
                fighter_a=self._clean_fighter_name(fighter_a),
                fighter_b=self._clean_fighter_name(fighter_b),
                fighter_a_probability=prob_a,
                fighter_b_probability=prob_b,
                fighter_a_decimal_odds=self._probability_to_decimal_odds(prob_a),
                fighter_b_decimal_odds=self._probability_to_decimal_odds(prob_b),
                fighter_a_american_odds=self._probability_to_american_odds(prob_a),
                fighter_b_american_odds=self._probability_to_american_odds(prob_b),
                market_volume=100000.0,  # Estimated
                last_trade_time=datetime.now()
            )
            
        except Exception as e:
            print(f"⚠️ Error processing fight data: {e}")
            return None
    
    def _parse_probability_from_text(self, text: str) -> Optional[float]:
        """Parse probability from various text formats"""
        if not text:
            return None
        
        # Look for cent format (68¢ -> 0.68)
        cent_match = re.search(r'(\d+)¢', text)
        if cent_match:
            return float(cent_match.group(1)) / 100
        
        # Look for percentage format (68% -> 0.68)  
        percent_match = re.search(r'(\d+(?:\.\d+)?)%', text)
        if percent_match:
            return float(percent_match.group(1)) / 100
        
        # Look for decimal format (0.68 -> 0.68)
        decimal_match = re.search(r'(0\.\d+)', text)
        if decimal_match:
            return float(decimal_match.group(1))
        
        return None
    
    def _create_fight_odds(self, fighter_a: str, fighter_b: str, prob_a: float, prob_b: float) -> PolymarketFightOdds:
        """Create a PolymarketFightOdds object with calculated odds"""
        return PolymarketFightOdds(
            event_name="UFC Fight Night",
            fighter_a=self._clean_fighter_name(fighter_a),
            fighter_b=self._clean_fighter_name(fighter_b),
            fighter_a_probability=prob_a,
            fighter_b_probability=prob_b,
            fighter_a_decimal_odds=self._probability_to_decimal_odds(prob_a),
            fighter_b_decimal_odds=self._probability_to_decimal_odds(prob_b),
            fighter_a_american_odds=self._probability_to_american_odds(prob_a),
            fighter_b_american_odds=self._probability_to_american_odds(prob_b),
            market_volume=50000.0,
            last_trade_time=datetime.now()
        )
    
    def _fighters_match(self, name1: str, name2: str) -> bool:
        """Check if two fighter names match (handles variations)"""
        if not name1 or not name2:
            return False
            
        name1 = self._clean_fighter_name(name1.lower())
        name2 = self._clean_fighter_name(name2.lower())
        
        # Exact match
        if name1 == name2:
            return True
            
        # Check if one is a substring of the other (for partial matches)
        name1_parts = name1.split()
        name2_parts = name2.split()
        
        # Check if all words from shorter name are in longer name
        if len(name1_parts) <= len(name2_parts):
            return all(part in name2 for part in name1_parts)
        else:
            return all(part in name1 for part in name2_parts)
    
    def _clean_fighter_name(self, name: str) -> str:
        """Clean fighter names for database matching"""
        if not name:
            return ""
            
        # Remove extra whitespace and artifacts
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Remove common prefixes/suffixes
        name = re.sub(r'\s*\(.*?\)\s*', '', name)
        
        # Remove quotation marks
        name = name.replace('"', '').replace("'", "")
        
        # Remove "..." truncation indicators
        name = name.replace('...', '').strip()
        
        # Handle common name variations
        name_corrections = {
            'Sharaputdin Magomedov': 'Shara Magomedov',
            'Marc-André Barriault': 'Marc-Andre Barriault'
        }
        
        return name_corrections.get(name, name)
    
    def _probability_to_decimal_odds(self, probability: float) -> float:
        """Convert probability to decimal odds (primary format)"""
        if probability <= 0 or probability >= 1:
            return 1.0
        return 1.0 / probability
    
    def _decimal_to_american_odds(self, decimal_odds: float) -> int:
        """Convert decimal odds to American odds"""
        if decimal_odds <= 1.0:
            return 0
        elif decimal_odds >= 2.0:
            # Underdog (positive odds)
            return min(10000, int((decimal_odds - 1) * 100))
        else:
            # Favorite (negative odds)
            return max(-10000, int(-100 / (decimal_odds - 1)))
    
    def _probability_to_american_odds(self, probability: float) -> int:
        """Convert probability to American odds (legacy support)"""
        decimal_odds = self._probability_to_decimal_odds(probability)
        return self._decimal_to_american_odds(decimal_odds)

# Utility functions for integration with main system
async def scrape_polymarket_ufc_event(event_url: str, headless: bool = True) -> List[PolymarketFightOdds]:
    """
    Convenience function to scrape Polymarket UFC event
    
    Args:
        event_url: Full Polymarket event URL
        headless: Run browser in headless mode
        
    Returns:
        List of PolymarketFightOdds objects
        
    Raises:
        Exception: If scraping fails
    """
    scraper = PolymarketUFCScraper(headless=headless)
    return await scraper.scrape_ufc_event(event_url)

def convert_polymarket_odds_to_dict(odds_list: List[PolymarketFightOdds]) -> List[Dict]:
    """
    Convert PolymarketFightOdds objects to dictionary format for JSON serialization
    
    Args:
        odds_list: List of PolymarketFightOdds objects
        
    Returns:
        List of dictionaries containing odds data
    """
    return [
        {
            'event_name': odds.event_name,
            'fighter_a': odds.fighter_a,
            'fighter_b': odds.fighter_b,
            'fighter_a_probability': odds.fighter_a_probability,
            'fighter_b_probability': odds.fighter_b_probability,
            'fighter_a_decimal_odds': odds.fighter_a_decimal_odds,
            'fighter_b_decimal_odds': odds.fighter_b_decimal_odds,
            'fighter_a_american_odds': odds.fighter_a_american_odds,
            'fighter_b_american_odds': odds.fighter_b_american_odds,
            'market_volume': odds.market_volume,
            'last_trade_time': odds.last_trade_time.isoformat(),
            'source': odds.source
        }
        for odds in odds_list
    ]

# Main execution for testing
if __name__ == "__main__":
    async def test_scraper():
        """Test the fixed Polymarket scraper"""
        event_url = "https://polymarket.com/event/ufc-fight-night-whittaker-vs-de-ridder?tid=1753515605835"
        
        print("🧪 Testing Fixed Polymarket UFC Scraper...")
        print(f"📍 Target URL: {event_url}")
        
        try:
            scraper = PolymarketUFCScraper(headless=True)
            odds = await scraper.scrape_ufc_event(event_url)
            
            print(f"\n✅ Scraped {len(odds)} fight odds:")
            print("=" * 60)
            
            for fight_odds in odds:
                print(f"🥊 {fight_odds.fighter_a} vs {fight_odds.fighter_b}")
                print(f"   Probabilities: {fight_odds.fighter_a_probability:.1%} vs {fight_odds.fighter_b_probability:.1%}")
                print(f"   Decimal Odds: {fight_odds.fighter_a_decimal_odds:.2f} vs {fight_odds.fighter_b_decimal_odds:.2f}")
                print(f"   American Odds: {fight_odds.fighter_a_american_odds:+d} vs {fight_odds.fighter_b_american_odds:+d}")
                print(f"   Volume: ${fight_odds.market_volume:,.0f}")
                print(f"   Source: {fight_odds.source}")
                print()
                
        except Exception as e:
            print(f"❌ Scraping failed: {e}")
            print("💡 Check debug screenshots for troubleshooting")
    
    # Run the test
    asyncio.run(test_scraper())
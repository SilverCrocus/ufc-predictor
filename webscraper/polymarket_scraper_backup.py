"""
Enhanced Polymarket UFC Odds Scraper
=====================================

Professional-grade scraper for UFC prediction market odds from Polymarket.
Integrates with the UFC Enhanced Card Analysis system.

Author: UFC Prediction System
Version: 2.0
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
    print("   Falling back to simulated data mode")

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
    Playwright-based scraper for Polymarket UFC prediction markets
    
    Features:
    - Stealth browsing to avoid detection
    - Robust error handling and retries
    - Fighter name cleaning and matching
    - Probability to American odds conversion
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
        Scrape UFC event using Playwright for dynamic content with retry logic
        
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
                
                # Extract fight data
                fights = await self._extract_fight_markets(page)
                
                if fights:
                    print(f"✅ Successfully scraped {len(fights)} fight odds")
                    return fights
                else:
                    # Take a screenshot for debugging
                    await page.screenshot(path="polymarket_debug_failure.png")
                    raise Exception("No fight data found on page. Debug screenshot saved as polymarket_debug_failure.png")
                    
            finally:
                await browser.close()
    
    async def _extract_fight_markets(self, page) -> List[PolymarketFightOdds]:
        """Extract fight markets from the loaded page"""
        
        try:
            # Wait for betting buttons to load (updated selectors based on actual DOM)
            print("🔍 Looking for betting buttons...")
            await page.wait_for_selector(
                'button:has-text("buy"), button:has-text("¢")', 
                timeout=15000
            )
            
            # Execute JavaScript to extract data from page using actual DOM structure
            market_data = await page.evaluate('''
                () => {
                    const markets = [];
                    
                    // Based on DOM analysis, look for fight container buttons
                    const fightButtons = document.querySelectorAll('button.c-PJLV.c-bWVhnw');
                    
                    console.log(`Found ${fightButtons.length} fight container buttons`);
                    
                    fightButtons.forEach((fightButton, index) => {
                        const textContent = fightButton.textContent || '';
                        
                        // Look for fight pattern: "Fighter vs Fighter" followed by cent values
                        const fightMatch = textContent.match(/(\\w+(?:\\s+\\w+)*)\\s+vs\\.?\\s+(\\w+(?:\\s+\\w+)*)/i);
                        
                        if (fightMatch) {
                            const fighter1 = fightMatch[1].trim();
                            const fighter2 = fightMatch[2].trim();
                            
                            // Extract cent values from this fight container
                            const centMatches = textContent.match(/(\\d+)¢/g) || [];
                            
                            // Also look for individual buy buttons within this container
                            const buyButtons = fightButton.querySelectorAll('button:has([class*="flex-shrink-0"])') || 
                                             Array.from(fightButton.querySelectorAll('button')).filter(btn => 
                                                 btn.textContent && btn.textContent.includes('¢')
                                             );
                            
                            const buyButtonData = [];
                            buyButtons.forEach(btn => {
                                const btnText = btn.textContent || '';
                                const centMatch = btnText.match(/(\\d+)¢/);
                                const fighterMatch = btnText.match(/buy\\s+(\\w+(?:\\s+\\w+)*)/i);
                                if (centMatch && fighterMatch) {
                                    buyButtonData.push({
                                        fighter: fighterMatch[1].trim(),
                                        cents: parseInt(centMatch[1])
                                    });
                                }
                            });
                            
                            markets.push({
                                index: index,
                                fighters: [fighter1, fighter2],
                                centValues: centMatches,
                                buyButtons: buyButtonData,
                                fullText: textContent,
                                element: {
                                    html: fightButton.outerHTML.substring(0, 500),
                                    text: textContent.substring(0, 300)
                                }
                            });
                        }
                    });
                    
                    // Fallback: Also look for individual buy buttons if fight containers missed some
                    if (markets.length === 0) {
                        console.log("No fight containers found, trying individual buy buttons...");
                        
                        const allBuyButtons = document.querySelectorAll('button');
                        const buyButtonsByFight = {};
                        
                        allBuyButtons.forEach(btn => {
                            const btnText = btn.textContent || '';
                            if (btnText.includes('buy') && btnText.includes('¢')) {
                                const centMatch = btnText.match(/(\\d+)¢/);
                                const fighterMatch = btnText.match(/buy\\s+(\\w+(?:\\s+\\w+)*)/i);
                                
                                if (centMatch && fighterMatch) {
                                    const fighter = fighterMatch[1].trim();
                                    const cents = parseInt(centMatch[1]);
                                    
                                    // Try to find the fight this belongs to by looking at surrounding text
                                    let fightContext = btn.closest('*')?.textContent || '';
                                    let contextMatch = fightContext.match(/(\\w+(?:\\s+\\w+)*)\\s+vs\\.?\\s+(\\w+(?:\\s+\\w+)*)/i);
                                    
                                    if (contextMatch) {
                                        const fightKey = `${contextMatch[1]} vs ${contextMatch[2]}`;
                                        if (!buyButtonsByFight[fightKey]) {
                                            buyButtonsByFight[fightKey] = {
                                                fighters: [contextMatch[1].trim(), contextMatch[2].trim()],
                                                buttons: []
                                            };
                                        }
                                        buyButtonsByFight[fightKey].buttons.push({
                                            fighter: fighter,
                                            cents: cents
                                        });
                                    }
                                }
                            }
                        });
                        
                        // Convert buyButtonsByFight to markets format
                        Object.values(buyButtonsByFight).forEach((fight, index) => {
                            if (fight.buttons.length >= 2) {
                                markets.push({
                                    index: index,
                                    fighters: fight.fighters,
                                    centValues: fight.buttons.map(b => `${b.cents}¢`),
                                    buyButtons: fight.buttons,
                                    fullText: `${fight.fighters[0]} vs ${fight.fighters[1]}`,
                                    element: {
                                        html: '',
                                        text: `${fight.fighters[0]} vs ${fight.fighters[1]}`
                                    }
                                });
                            }
                        });
                    }
                    
                    console.log(`Extracted ${markets.length} fight markets`);
                    return markets;
                }
            ''')
            
            print(f"📊 Processing {len(market_data)} potential markets...")
            
            # Process extracted data
            fights = []
            for market in market_data:
                fight_odds = self._process_market_data(market)
                if fight_odds:
                    fights.append(fight_odds)
            
            if not fights:
                print("⚠️ No valid fight markets found, using fallback extraction...")
                fights = await self._fallback_extraction(page)
            
            return fights
            
        except Exception as e:
            print(f"❌ Error extracting markets: {e}")
            return []
    
    async def _fallback_extraction(self, page) -> List[PolymarketFightOdds]:
        """Fallback extraction method using simpler selectors"""
        
        try:
            # Get all text content and parse manually
            page_content = await page.evaluate('() => document.body.textContent')
            
            # Look for fighter names and odds patterns
            fight_patterns = [
                r'(Robert\s+Whittaker).*?(Reinier\s+de\s+Ridder)',
                r'(Petr\s+Yan).*?(Marcus\s+McGhee)',
                r'(Sharaputdin\s+Magomedov).*?(Marc-André\s+Barriault)',
                r'(Asu\s+Almabayev).*?(Jose\s+Ochoa)',
                r'(Nikita\s+Krylov).*?(Bogdan\s+Guskov)'
            ]
            
            fights = []
            for pattern in fight_patterns:
                match = re.search(pattern, page_content, re.IGNORECASE)
                if match:
                    fighter_a, fighter_b = match.groups()
                    
                    # Extract probabilities from surrounding text
                    fight_section = page_content[max(0, match.start()-200):match.end()+200]
                    prob_matches = re.findall(r'(\d+(?:\.\d+)?)%', fight_section)
                    
                    if len(prob_matches) >= 2:
                        prob_a = float(prob_matches[0]) / 100
                        prob_b = float(prob_matches[1]) / 100
                        
                        # Normalize if needed
                        if abs(prob_a + prob_b - 1.0) > 0.1:
                            prob_b = 1.0 - prob_a
                            
                        fight_odds = PolymarketFightOdds(
                            event_name="UFC Fight Night",
                            fighter_a=self._clean_fighter_name(fighter_a),
                            fighter_b=self._clean_fighter_name(fighter_b),
                            fighter_a_probability=prob_a,
                            fighter_b_probability=prob_b,
                            fighter_a_decimal_odds=self._probability_to_decimal_odds(prob_a),
                            fighter_b_decimal_odds=self._probability_to_decimal_odds(prob_b),
                            fighter_a_american_odds=self._probability_to_american_odds(prob_a),
                            fighter_b_american_odds=self._probability_to_american_odds(prob_b),
                            market_volume=50000.0,  # Estimated
                            last_trade_time=datetime.now()
                        )
                        
                        fights.append(fight_odds)
            
            return fights
            
        except Exception as e:
            print(f"❌ Fallback extraction failed: {e}")
            return []
    
    def _process_market_data(self, market_data: Dict) -> Optional[PolymarketFightOdds]:
        """Process raw market data into structured fight odds"""
        
        try:
            fighters = market_data.get('fighters', [])
            if len(fighters) < 2:
                return None
                
            fighter_a, fighter_b = fighters[0], fighters[1]
            
            # First try to extract probabilities from buyButtons (new structure)
            buy_buttons = market_data.get('buyButtons', [])
            probabilities = {}
            
            if buy_buttons:
                # Map fighter names to their cent values (probabilities)
                for button in buy_buttons:
                    fighter_name = button.get('fighter', '').strip()
                    cents = button.get('cents', 0)
                    
                    if cents > 0 and 5 <= cents <= 95:  # Reasonable probability range
                        prob_val = cents / 100.0
                        
                        # Match fighter name to our fighters list
                        for fighter in [fighter_a, fighter_b]:
                            if self._fighters_match(fighter_name, fighter):
                                probabilities[fighter] = prob_val
                                break
                
                # If we got probabilities for both fighters
                if len(probabilities) >= 2:
                    prob_a = probabilities.get(fighter_a, 0)
                    prob_b = probabilities.get(fighter_b, 0)
                    
                    # Use the probabilities as-is (they should already be complementary)
                    if prob_a > 0 and prob_b > 0:
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
                            market_volume=50000.0,  # Estimated volume
                            last_trade_time=datetime.now()
                        )
                
                # If we only got one probability, calculate the complement
                elif len(probabilities) == 1:
                    fighter_with_prob = list(probabilities.keys())[0]
                    prob_value = probabilities[fighter_with_prob]
                    
                    if fighter_with_prob == fighter_a:
                        prob_a, prob_b = prob_value, 1.0 - prob_value
                    else:
                        prob_a, prob_b = 1.0 - prob_value, prob_value
                    
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
            
            # Fallback: Extract from centValues or text (old method)
            full_text = market_data.get('fullText', '')
            cent_values = market_data.get('centValues', [])
            
            # Extract cent values from text
            all_cent_matches = []
            for cent_str in cent_values:
                match = re.search(r'(\d+)¢', cent_str)
                if match:
                    cents = int(match.group(1))
                    if 5 <= cents <= 95:
                        all_cent_matches.append(cents / 100.0)
            
            # Also check full text for cent values
            cent_matches = re.findall(r'(\d+)¢', full_text)
            for match in cent_matches:
                cents = int(match)
                if 5 <= cents <= 95:
                    all_cent_matches.append(cents / 100.0)
            
            # Remove duplicates and sort
            probabilities_list = sorted(list(set(all_cent_matches)), reverse=True)
            
            if len(probabilities_list) >= 2:
                prob_a, prob_b = probabilities_list[0], probabilities_list[1]
                
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
                
        except Exception as e:
            print(f"⚠️ Error processing market data: {e}")
            
        return None
    
    def _fighters_match(self, name1: str, name2: str) -> bool:
        """Check if two fighter names match (handles variations)"""
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
    
    def _get_simulated_odds(self) -> List[PolymarketFightOdds]:
        """
        Simulated Polymarket odds for testing when scraping is unavailable
        These should be replaced with actual scraped data in production
        """
        
        simulated_odds = [
            # Main Event: Whittaker vs de Ridder
            PolymarketFightOdds(
                event_name="UFC Fight Night - Whittaker vs de Ridder",
                fighter_a="Robert Whittaker",
                fighter_b="Reinier de Ridder",
                fighter_a_probability=0.68,
                fighter_b_probability=0.32,
                fighter_a_decimal_odds=self._probability_to_decimal_odds(0.68),
                fighter_b_decimal_odds=self._probability_to_decimal_odds(0.32),
                fighter_a_american_odds=self._probability_to_american_odds(0.68),
                fighter_b_american_odds=self._probability_to_american_odds(0.32),
                market_volume=125000.0,
                last_trade_time=datetime.now()
            ),
            
            # Co-main: Yan vs McGhee
            PolymarketFightOdds(
                event_name="UFC Fight Night - Whittaker vs de Ridder",
                fighter_a="Petr Yan",
                fighter_b="Marcus McGhee",
                fighter_a_probability=0.78,
                fighter_b_probability=0.22,
                fighter_a_decimal_odds=self._probability_to_decimal_odds(0.78),
                fighter_b_decimal_odds=self._probability_to_decimal_odds(0.22),
                fighter_a_american_odds=self._probability_to_american_odds(0.78),
                fighter_b_american_odds=self._probability_to_american_odds(0.22),
                market_volume=95000.0,
                last_trade_time=datetime.now()
            ),
            
            # Magomedov vs Barriault
            PolymarketFightOdds(
                event_name="UFC Fight Night - Whittaker vs de Ridder",
                fighter_a="Shara Magomedov",
                fighter_b="Marc-Andre Barriault",
                fighter_a_probability=0.58,
                fighter_b_probability=0.42,
                fighter_a_decimal_odds=self._probability_to_decimal_odds(0.58),
                fighter_b_decimal_odds=self._probability_to_decimal_odds(0.42),
                fighter_a_american_odds=self._probability_to_american_odds(0.58),
                fighter_b_american_odds=self._probability_to_american_odds(0.42),
                market_volume=75000.0,
                last_trade_time=datetime.now()
            ),
            
            # Almabayev vs Ochoa
            PolymarketFightOdds(
                event_name="UFC Fight Night - Whittaker vs de Ridder",
                fighter_a="Asu Almabayev",
                fighter_b="Jose Ochoa",
                fighter_a_probability=0.72,
                fighter_b_probability=0.28,
                fighter_a_decimal_odds=self._probability_to_decimal_odds(0.72),
                fighter_b_decimal_odds=self._probability_to_decimal_odds(0.28),
                fighter_a_american_odds=self._probability_to_american_odds(0.72),
                fighter_b_american_odds=self._probability_to_american_odds(0.28),
                market_volume=65000.0,
                last_trade_time=datetime.now()
            ),
            
            # Krylov vs Guskov
            PolymarketFightOdds(
                event_name="UFC Fight Night - Whittaker vs de Ridder",
                fighter_a="Nikita Krylov",
                fighter_b="Bogdan Guskov",
                fighter_a_probability=0.62,
                fighter_b_probability=0.38,
                fighter_a_decimal_odds=self._probability_to_decimal_odds(0.62),
                fighter_b_decimal_odds=self._probability_to_decimal_odds(0.38),
                fighter_a_american_odds=self._probability_to_american_odds(0.62),
                fighter_b_american_odds=self._probability_to_american_odds(0.38),
                market_volume=55000.0,
                last_trade_time=datetime.now()
            )
        ]
        
        return simulated_odds

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

def get_simulated_polymarket_odds() -> List[PolymarketFightOdds]:
    """
    Get simulated Polymarket odds for testing purposes
    
    Returns:
        List of PolymarketFightOdds with realistic odds data
    """
    scraper = PolymarketUFCScraper()
    return scraper._get_simulated_odds()

# Main execution for testing
if __name__ == "__main__":
    async def test_scraper():
        """Test the Polymarket scraper with fallback to simulated data"""
        event_url = "https://polymarket.com/event/ufc-fight-night-whittaker-vs-de-ridder?tid=1753515605835"
        
        print("🧪 Testing Polymarket UFC Scraper...")
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
            print(f"❌ Live scraping failed: {e}")
            print("🔄 Falling back to simulated data...")
            
            # Fallback to simulated data
            scraper = PolymarketUFCScraper()
            odds = scraper._get_simulated_odds()
            
            print(f"\n✅ Using {len(odds)} simulated fight odds:")
            print("=" * 60)
            
            for fight_odds in odds:
                print(f"🥊 {fight_odds.fighter_a} vs {fight_odds.fighter_b}")
                print(f"   Probabilities: {fight_odds.fighter_a_probability:.1%} vs {fight_odds.fighter_b_probability:.1%}")
                print(f"   Decimal Odds: {fight_odds.fighter_a_decimal_odds:.2f} vs {fight_odds.fighter_b_decimal_odds:.2f}")
                print(f"   American Odds: {fight_odds.fighter_a_american_odds:+d} vs {fight_odds.fighter_b_american_odds:+d}")
                print(f"   Volume: ${fight_odds.market_volume:,.0f}")
                print()
    
    # Run the test
    asyncio.run(test_scraper())
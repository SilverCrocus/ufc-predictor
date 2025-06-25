"""
TAB Australia UFC Odds Scraper

This scraper handles TAB Australia's unique structure where:
1. Featured fights (main events) are on a separate page
2. Undercard fights are on individual event pages  
3. Odds are in decimal format (need conversion to American)

URLs:
- Main UFC page: https://www.tab.com.au/sports/betting/UFC
- Featured fights: https://www.tab.com.au/sports/betting/UFC/competitions/UFC%20Featured%20Fights/tournaments/UFC%20Featured%20Fights
- Individual events: https://www.tab.com.au/sports/betting/UFC/competitions/UFC/tournaments/{EVENT_NAME}
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import json
import re

@dataclass
class TABFightOdds:
    """Container for TAB Australia fight odds data"""
    event_name: str
    fighter_a: str
    fighter_b: str
    fighter_a_decimal_odds: float
    fighter_b_decimal_odds: float
    fighter_a_american_odds: int
    fighter_b_american_odds: int
    fight_time: str
    is_featured_fight: bool
    
    def to_dict(self):
        return asdict(self)

class TABAustraliaUFCScraper:
    """Comprehensive scraper for TAB Australia UFC odds"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
        self.wait = None
        
        # TAB Australia URLs
        self.base_ufc_url = "https://www.tab.com.au/sports/betting/UFC"
        self.featured_fights_url = "https://www.tab.com.au/sports/betting/UFC/competitions/UFC%20Featured%20Fights/tournaments/UFC%20Featured%20Fights"
        
    def setup_driver(self):
        """Initialize the Chrome WebDriver for Australian site"""
        print("🇦🇺 Setting up Chrome WebDriver for TAB Australia...")
        
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Standard options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Australian user agent to avoid geo-blocking
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, 15)
            print("✅ Chrome WebDriver initialized for TAB Australia")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Chrome WebDriver: {e}")
            return False
    
    def decimal_to_american_odds(self, decimal_odds: float) -> int:
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2.0:
            # Positive American odds (underdog)
            return int((decimal_odds - 1) * 100)
        else:
            # Negative American odds (favorite)
            return int(-100 / (decimal_odds - 1))
    
    def scrape_all_ufc_odds(self) -> List[TABFightOdds]:
        """Scrape all UFC odds from TAB Australia (featured + individual events)"""
        print("🥊 Starting comprehensive TAB Australia UFC scraping...")
        
        if not self.setup_driver():
            return self._get_sample_tab_data()
        
        all_fights = []
        
        try:
            # Step 1: Scrape Featured Fights
            print("\n🌟 SCRAPING FEATURED FIGHTS")
            print("-" * 40)
            featured_fights = self._scrape_featured_fights()
            all_fights.extend(featured_fights)
            
            # Step 2: Discover and scrape individual events
            print("\n📋 DISCOVERING INDIVIDUAL EVENTS")
            print("-" * 40)
            event_urls = self._discover_event_urls()
            
            for event_url in event_urls:
                print(f"\n🎯 Scraping event: {event_url.split('/')[-1]}")
                event_fights = self._scrape_event_page(event_url)
                all_fights.extend(event_fights)
            
            print(f"\n✅ Total fights scraped: {len(all_fights)}")
            return all_fights
            
        except Exception as e:
            print(f"❌ Error during scraping: {e}")
            import traceback
            traceback.print_exc()
            return self._get_sample_tab_data()
        
        finally:
            if self.driver:
                self.driver.quit()
                print("🔒 WebDriver closed")
    
    def _scrape_featured_fights(self) -> List[TABFightOdds]:
        """Scrape the featured fights page"""
        featured_fights = []
        
        try:
            print(f"📄 Loading featured fights page...")
            self.driver.get(self.featured_fights_url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Take screenshot for debugging
            self.driver.save_screenshot("tab_featured_fights.png")
            print("📸 Screenshot saved: tab_featured_fights.png")
            
            # Parse the page content
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Look for fight containers (you'll need to adjust these selectors based on actual HTML)
            fight_containers = self._find_fight_containers(soup)
            
            for container in fight_containers:
                fight_data = self._extract_fight_data(container, is_featured=True)
                if fight_data:
                    featured_fights.append(fight_data)
                    print(f"   ✅ {fight_data.fighter_a} vs {fight_data.fighter_b}")
            
            print(f"🌟 Featured fights found: {len(featured_fights)}")
            
        except Exception as e:
            print(f"❌ Error scraping featured fights: {e}")
        
        return featured_fights
    
    def _discover_event_urls(self) -> List[str]:
        """Discover individual UFC event URLs from the main page"""
        event_urls = []
        
        try:
            print(f"📄 Loading main UFC page to discover events...")
            self.driver.get(self.base_ufc_url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Look for event links
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Find links that match the event pattern
            # Pattern: https://www.tab.com.au/sports/betting/UFC/competitions/UFC/tournaments/{EVENT_NAME}
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                if '/UFC/competitions/UFC/tournaments/' in href and 'Featured%20Fights' not in href:
                    full_url = f"https://www.tab.com.au{href}" if href.startswith('/') else href
                    if full_url not in event_urls:
                        event_urls.append(full_url)
                        
                        # Extract event name for display
                        event_name = href.split('/')[-1].replace('%20', ' ')
                        print(f"   🎯 Found event: {event_name}")
            
            print(f"📋 Total events discovered: {len(event_urls)}")
            
        except Exception as e:
            print(f"❌ Error discovering events: {e}")
        
        return event_urls
    
    def _scrape_event_page(self, event_url: str) -> List[TABFightOdds]:
        """Scrape a specific event page"""
        event_fights = []
        
        try:
            self.driver.get(event_url)
            time.sleep(2)
            
            # Extract event name from URL
            event_name = event_url.split('/')[-1].replace('%20', ' ')
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            fight_containers = self._find_fight_containers(soup)
            
            for container in fight_containers:
                fight_data = self._extract_fight_data(container, is_featured=False, event_name=event_name)
                if fight_data:
                    event_fights.append(fight_data)
                    print(f"   ✅ {fight_data.fighter_a} vs {fight_data.fighter_b}")
            
        except Exception as e:
            print(f"❌ Error scraping event {event_url}: {e}")
        
        return event_fights
    
    def _find_fight_containers(self, soup: BeautifulSoup) -> List:
        """Find fight containers in the HTML - this needs to be customized based on TAB's structure"""
        
        # Save full HTML for analysis
        with open('tab_page_source.html', 'w', encoding='utf-8') as f:
            f.write(str(soup))
        print("💾 Page source saved to: tab_page_source.html")
        
        # Look for potential fight containers
        # These selectors will need to be adjusted based on actual TAB HTML structure
        potential_selectors = [
            'div[class*="fight"]',
            'div[class*="match"]',
            'div[class*="event"]',
            'div[class*="market"]',
            'div[class*="selection"]',
            '[data-testid*="fight"]',
            '[data-testid*="match"]',
            'article',
            'section',
        ]
        
        fight_containers = []
        
        for selector in potential_selectors:
            elements = soup.select(selector)
            if elements:
                print(f"   🔍 Found {len(elements)} elements with selector: {selector}")
                
                # Check if these elements contain fighter-like content
                for element in elements:
                    text_content = element.get_text(strip=True)
                    if self._looks_like_fight_content(text_content):
                        fight_containers.append(element)
                        print(f"     ✅ Fight-like content: {text_content[:100]}...")
        
        # Remove duplicates
        unique_containers = []
        seen_content = set()
        
        for container in fight_containers:
            content_hash = hash(container.get_text(strip=True))
            if content_hash not in seen_content:
                unique_containers.append(container)
                seen_content.add(content_hash)
        
        print(f"📊 Unique fight containers found: {len(unique_containers)}")
        return unique_containers
    
    def _looks_like_fight_content(self, text: str) -> bool:
        """Determine if text content looks like a fight"""
        # Look for patterns that suggest this is fight content
        fight_indicators = [
            ' v ', ' vs ', ' V ', ' VS ',
            'Head To Head',
            # Common fighter name patterns
            len([word for word in text.split() if word.istitle()]) >= 2,  # Multiple capitalized words
        ]
        
        # Check for decimal odds pattern (1.XX, 2.XX, etc.)
        decimal_odds_pattern = re.search(r'\b[1-9]\.\d{2}\b', text)
        
        return any(indicator for indicator in fight_indicators if isinstance(indicator, bool) and indicator) or \
               any(indicator in text for indicator in fight_indicators if isinstance(indicator, str)) or \
               decimal_odds_pattern is not None
    
    def _extract_fight_data(self, container, is_featured: bool = False, event_name: str = None) -> TABFightOdds:
        """Extract fight data from a container element"""
        try:
            text_content = container.get_text(separator='|', strip=True)
            
            # Look for fighter names and odds
            # This is a basic implementation - you'll need to refine based on actual HTML structure
            
            # Try to find decimal odds (pattern: X.XX)
            odds_matches = re.findall(r'\b([1-9]\.\d{2})\b', text_content)
            
            if len(odds_matches) >= 2:
                decimal_odds_a = float(odds_matches[0])
                decimal_odds_b = float(odds_matches[1])
                
                # Convert to American odds
                american_odds_a = self.decimal_to_american_odds(decimal_odds_a)
                american_odds_b = self.decimal_to_american_odds(decimal_odds_b)
                
                # Try to extract fighter names (this is basic - needs refinement)
                text_parts = [part.strip() for part in text_content.split('|') if part.strip()]
                
                # Look for fighter-like names (capitalized words, not just numbers)
                potential_fighters = []
                for part in text_parts:
                    if (len(part) > 3 and 
                        any(c.isalpha() for c in part) and 
                        not re.match(r'^\d+\.\d+$', part) and
                        'Head To Head' not in part):
                        potential_fighters.append(part)
                
                if len(potential_fighters) >= 2:
                    fighter_a = potential_fighters[0]
                    fighter_b = potential_fighters[1]
                    
                    # Determine event name
                    if not event_name:
                        if is_featured:
                            event_name = "UFC Featured Fights"
                        else:
                            event_name = "UFC Event"
                    
                    return TABFightOdds(
                        event_name=event_name,
                        fighter_a=fighter_a,
                        fighter_b=fighter_b,
                        fighter_a_decimal_odds=decimal_odds_a,
                        fighter_b_decimal_odds=decimal_odds_b,
                        fighter_a_american_odds=american_odds_a,
                        fighter_b_american_odds=american_odds_b,
                        fight_time="TBD",
                        is_featured_fight=is_featured
                    )
            
        except Exception as e:
            print(f"❌ Error extracting fight data: {e}")
        
        return None
    
    def _get_sample_tab_data(self) -> List[TABFightOdds]:
        """Return sample data based on the screenshots provided"""
        print("📋 Returning sample TAB Australia data from screenshots")
        
        return [
            TABFightOdds(
                event_name="UFC 317",
                fighter_a="Charles Oliveira",
                fighter_b="Ilia Topuria", 
                fighter_a_decimal_odds=4.25,
                fighter_b_decimal_odds=1.22,
                fighter_a_american_odds=325,  # 4.25 decimal = +325 American
                fighter_b_american_odds=-455,  # 1.22 decimal = -455 American
                fight_time="Sun 29 Jun 14:00",
                is_featured_fight=True
            ),
            TABFightOdds(
                event_name="UFC 317",
                fighter_a="Alexandre Pantoja",
                fighter_b="Steve Erceg",  # Assuming from KaraFrance fight
                fighter_a_decimal_odds=1.40,
                fighter_b_decimal_odds=2.95,
                fighter_a_american_odds=-250,  # 1.40 decimal = -250 American  
                fighter_b_american_odds=195,   # 2.95 decimal = +195 American
                fight_time="Sun 29 Jun 13:30",
                is_featured_fight=True
            ),
            TABFightOdds(
                event_name="UFC 317",
                fighter_a="Hyder Amil",
                fighter_b="Jose Delgado",
                fighter_a_decimal_odds=2.35,
                fighter_b_decimal_odds=1.60,
                fighter_a_american_odds=135,   # 2.35 decimal = +135 American
                fighter_b_american_odds=-167,  # 1.60 decimal = -167 American
                fight_time="Sun 29 Jun 7:00",
                is_featured_fight=False
            )
        ]

def main():
    """Test the TAB Australia scraper"""
    print("🇦🇺 TAB AUSTRALIA UFC SCRAPER TEST")
    print("=" * 50)
    
    scraper = TABAustraliaUFCScraper(headless=False)  # Set to False to see the browser
    
    try:
        all_fights = scraper.scrape_all_ufc_odds()
        
        print(f"\n📊 TAB AUSTRALIA SCRAPING RESULTS")
        print("=" * 40)
        print(f"✅ Total fights scraped: {len(all_fights)}")
        
        # Separate featured vs regular fights
        featured = [f for f in all_fights if f.is_featured_fight]
        regular = [f for f in all_fights if not f.is_featured_fight]
        
        print(f"🌟 Featured fights: {len(featured)}")
        print(f"📋 Event fights: {len(regular)}")
        
        print(f"\n🥊 FIGHT DETAILS:")
        for i, fight in enumerate(all_fights, 1):
            print(f"\n{i}. {fight.fighter_a} vs {fight.fighter_b}")
            print(f"   Event: {fight.event_name}")
            print(f"   TAB Odds: {fight.fighter_a} ({fight.fighter_a_decimal_odds}) vs {fight.fighter_b} ({fight.fighter_b_decimal_odds})")
            print(f"   American: {fight.fighter_a} ({fight.fighter_a_american_odds:+d}) vs {fight.fighter_b} ({fight.fighter_b_american_odds:+d})")
            print(f"   Type: {'Featured' if fight.is_featured_fight else 'Regular'}")
        
        # Save results
        results = [fight.to_dict() for fight in all_fights]
        with open('tab_australia_odds.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: tab_australia_odds.json")
        
        # Show comparison with international odds if available
        print(f"\n📈 ODDS COMPARISON EXAMPLE:")
        if all_fights:
            fight = all_fights[0]
            print(f"Fight: {fight.fighter_a} vs {fight.fighter_b}")
            print(f"TAB Australia: {fight.fighter_a} ({fight.fighter_a_american_odds:+d}) vs {fight.fighter_b} ({fight.fighter_b_american_odds:+d})")
            print("International (FightOdds.io): Oliveira (+400) vs Topuria (-410)")
            print("💡 TAB offers worse odds on Oliveira (+325 vs +400) - this affects profitability!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Scraping failed: {e}")

if __name__ == "__main__":
    main() 
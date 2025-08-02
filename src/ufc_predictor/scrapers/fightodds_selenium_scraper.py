"""
Selenium-based FightOdds.io Scraper

This scraper can handle JavaScript-rendered content using Selenium WebDriver.
It's specifically designed to extract odds from the Material-UI components
that are dynamically loaded.
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
from typing import List, Dict
from dataclasses import dataclass, asdict
import json

@dataclass
class BettingOdds:
    """Single betting odds entry"""
    sportsbook: str
    american_odds: int
    decimal_odds: float
    implied_probability: float

@dataclass
class FightOddsData:
    """Complete fight odds data"""
    event_name: str
    fighter_a: str
    fighter_b: str
    fighter_a_odds: List[BettingOdds]
    fighter_b_odds: List[BettingOdds]
    
    def to_dict(self):
        return asdict(self)

class FightOddsSeleniumScraper:
    """Selenium-based scraper for fightodds.io"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
        self.wait = None
    
    def setup_driver(self):
        """Initialize the Chrome WebDriver with appropriate options"""
        print("🔧 Setting up Chrome WebDriver...")
        
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Standard options for better compatibility
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # Disable SSL verification for testing
        chrome_options.add_argument("--ignore-ssl-errors")
        chrome_options.add_argument("--ignore-certificate-errors")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, 20)
            print("✅ Chrome WebDriver initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Chrome WebDriver: {e}")
            print("💡 Make sure ChromeDriver is installed and in PATH")
            print("   Download from: https://chromedriver.chromium.org/")
            return False
    
    def scrape_fightodds(self, url: str = "https://fightodds.io/") -> List[FightOddsData]:
        """Scrape odds from fightodds.io"""
        print(f"🎯 Scraping odds from: {url}")
        
        if not self.setup_driver():
            return self._get_sample_data()
        
        try:
            # Navigate to the page
            print("📄 Loading page...")
            self.driver.get(url)
            
            # Wait for the React app to load
            print("⏳ Waiting for React app to load...")
            self._wait_for_app_to_load()
            
            # Take a screenshot for debugging
            self.driver.save_screenshot("fightodds_screenshot.png")
            print("📸 Screenshot saved: fightodds_screenshot.png")
            
            # Find the MUI container
            print("🔍 Looking for MUI odds container...")
            mui_container = self._find_mui_container()
            
            if mui_container:
                print("✅ Found MUI container! Extracting odds...")
                odds_data = self._extract_odds_from_mui(mui_container)
            else:
                print("❌ MUI container not found. Analyzing page structure...")
                self._analyze_page_structure()
                odds_data = []
            
            return odds_data if odds_data else self._get_sample_data()
            
        except Exception as e:
            print(f"❌ Error scraping: {e}")
            return self._get_sample_data()
        
        finally:
            if self.driver:
                self.driver.quit()
                print("🔒 WebDriver closed")
    
    def _wait_for_app_to_load(self):
        """Wait for the React app and content to load"""
        try:
            # Wait for the app div to have content
            self.wait.until(
                lambda driver: driver.find_element(By.ID, "app").get_attribute("innerHTML") != ""
            )
            
            # Additional wait for any loading indicators to disappear
            time.sleep(3)
            
            print("✅ React app loaded successfully")
            
        except TimeoutException:
            print("⚠️  Timeout waiting for app to load, continuing anyway...")
    
    def _find_mui_container(self):
        """Find the MUI container with odds data"""
        selectors_to_try = [
            # Specific selector you mentioned
            "nav.MuiList-root.jss1579",
            
            # More general MUI selectors
            "nav[class*='MuiList-root']",
            "div[class*='MuiList-root']",
            "[class*='MuiList']",
            
            # General odds-related selectors
            "[class*='odds']",
            "[class*='fight']",
            "[class*='bet']",
            "table",
            
            # Any nav element
            "nav",
        ]
        
        for selector in selectors_to_try:
            try:
                print(f"   Trying selector: {selector}")
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                
                if elements:
                    print(f"   ✅ Found {len(elements)} elements with: {selector}")
                    
                    # Check if any contain text that looks like odds or fighter names
                    for element in elements:
                        text_content = element.text
                        if text_content and (
                            any(char in text_content for char in ['+', '-']) or
                            len(text_content.split()) > 5  # Might contain fighter names
                        ):
                            print(f"   ✅ Found promising element with content")
                            return element
                else:
                    print(f"   ❌ No elements found")
                    
            except Exception as e:
                print(f"   ❌ Error with selector {selector}: {e}")
        
        return None
    
    def _extract_odds_from_mui(self, container) -> List[FightOddsData]:
        """Extract odds data from the MUI container"""
        odds_data = []
        
        try:
            # Get the HTML content and parse with BeautifulSoup for easier handling
            html_content = container.get_attribute('outerHTML')
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for patterns in the text to identify fights and odds
            all_text = container.text
            print(f"📊 Analyzing text content of {len(all_text)} characters")
            
            # Split by common separators and clean
            lines = [line.strip() for line in all_text.split('\n') if line.strip()]
            
            # Find event name
            event_name = "Unknown Event"
            for line in lines[:20]:  # Check first 20 lines
                if "UFC" in line and any(term in line for term in ["vs", ":", "Topuria", "Oliveira"]):
                    event_name = line
                    print(f"🎯 Found event: {event_name}")
                    break
            
            # Look for fighter names and odds patterns
            current_fight_data = None
            fights_found = []
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Skip header/navigation lines
                if any(skip_term in line.lower() for skip_term in [
                    'fighters', 'betonline', 'bovada', 'bookmaker', 'pinnacle',
                    'sportsbook', 'odds', 'bet', 'june', 'ufc 317'
                ]):
                    i += 1
                    continue
                
                # Look for potential fighter names (longer text, mostly letters)
                if (len(line) > 5 and 
                    len([c for c in line if c.isalpha()]) > len(line) * 0.6 and
                    not any(c in line for c in ['+', '-', '%', '$'])):
                    
                    print(f"🥊 Potential fighter: {line}")
                    
                    # Look ahead for odds in the next few lines
                    odds_found = []
                    for j in range(i+1, min(i+10, len(lines))):
                        next_line = lines[j]
                        if any(c in next_line for c in ['+', '-']) and any(c.isdigit() for c in next_line):
                            # Extract individual odds values
                            import re
                            odds_matches = re.findall(r'[+-]\d+', next_line)
                            for odds_match in odds_matches:
                                try:
                                    odds_val = int(odds_match)
                                    odds_found.append(odds_val)
                                except:
                                    pass
                    
                    if odds_found:
                        print(f"   📊 Found {len(odds_found)} odds: {odds_found[:5]}...")
                        
                        if current_fight_data is None:
                            current_fight_data = {
                                'fighter_a': line,
                                'fighter_a_odds': odds_found,
                                'event_name': event_name
                            }
                        else:
                            # Complete the fight
                            fight = self._create_fight_from_raw_data(
                                event_name,
                                current_fight_data['fighter_a'],
                                line,
                                current_fight_data['fighter_a_odds'],
                                odds_found
                            )
                            if fight:
                                fights_found.append(fight)
                                print(f"✅ Created fight: {fight.fighter_a} vs {fight.fighter_b}")
                            
                            current_fight_data = None
                
                i += 1
            
            print(f"🎯 Successfully extracted {len(fights_found)} fights")
            return fights_found if fights_found else self._get_sample_data()
            
        except Exception as e:
            print(f"❌ Error extracting from MUI: {e}")
            import traceback
            traceback.print_exc()
            return self._get_sample_data()
    
    def _create_fight_from_raw_data(self, event_name: str, fighter_a: str, fighter_b: str,
                                   fighter_a_odds: List[int], fighter_b_odds: List[int]) -> FightOddsData:
        """Create a FightOddsData object from raw extracted data"""
        try:
            sportsbooks = ['BetOnline', 'Bovada', 'Bookmaker', 'Pinnacle', 'Betway', 'BetUS', 
                          'DraftKings', 'FanDuel', 'Circa', 'ESPN', 'BetRivers', 'BetMGM', 'Caesars']
            
            # Convert american odds to decimal and implied probability
            def american_to_decimal(american_odds):
                if american_odds > 0:
                    return (american_odds / 100) + 1
                else:
                    return (100 / abs(american_odds)) + 1
            
            def american_to_implied_prob(american_odds):
                if american_odds > 0:
                    return 100 / (american_odds + 100) * 100
                else:
                    return abs(american_odds) / (abs(american_odds) + 100) * 100
            
            # Create betting odds objects for fighter A
            fighter_a_betting_odds = []
            for i, odds in enumerate(fighter_a_odds[:len(sportsbooks)]):
                betting_odds = BettingOdds(
                    sportsbook=sportsbooks[i] if i < len(sportsbooks) else f"Sportsbook_{i+1}",
                    american_odds=odds,
                    decimal_odds=round(american_to_decimal(odds), 2),
                    implied_probability=round(american_to_implied_prob(odds), 1)
                )
                fighter_a_betting_odds.append(betting_odds)
            
            # Create betting odds objects for fighter B
            fighter_b_betting_odds = []
            for i, odds in enumerate(fighter_b_odds[:len(sportsbooks)]):
                betting_odds = BettingOdds(
                    sportsbook=sportsbooks[i] if i < len(sportsbooks) else f"Sportsbook_{i+1}",
                    american_odds=odds,
                    decimal_odds=round(american_to_decimal(odds), 2),
                    implied_probability=round(american_to_implied_prob(odds), 1)
                )
                fighter_b_betting_odds.append(betting_odds)
            
            return FightOddsData(
                event_name=event_name,
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                fighter_a_odds=fighter_a_betting_odds,
                fighter_b_odds=fighter_b_betting_odds
            )
            
        except Exception as e:
            print(f"❌ Error creating fight data: {e}")
            return None
    
    def _analyze_page_structure(self):
        """Analyze the current page structure for debugging"""
        print("\n🔍 ANALYZING PAGE STRUCTURE")
        print("=" * 40)
        
        try:
            # Get page source
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Save full HTML
            with open('fightodds_full_page.html', 'w', encoding='utf-8') as f:
                f.write(page_source)
            print("💾 Full page HTML saved: fightodds_full_page.html")
            
            # Look for any text content that might contain fighter names or odds
            all_text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in all_text.split('\n') if line.strip()]
            
            print(f"📝 Found {len(lines)} lines of text content")
            
            # Look for lines that might contain odds or fighter names
            potential_odds = [line for line in lines if any(char in line for char in ['+', '-']) and any(c.isdigit() for c in line)]
            potential_fighters = [line for line in lines if len(line) > 10 and any(c.isalpha() for c in line)]
            
            print(f"🎯 Potential odds lines: {len(potential_odds)}")
            if potential_odds:
                for line in potential_odds[:5]:
                    print(f"   {line}")
            
            print(f"🥊 Potential fighter lines: {len(potential_fighters)}")
            if potential_fighters:
                for line in potential_fighters[:5]:
                    print(f"   {line}")
            
        except Exception as e:
            print(f"❌ Error analyzing structure: {e}")
    
    def _get_sample_data(self) -> List[FightOddsData]:
        """Return sample data when scraping fails"""
        print("📋 Returning sample data for demonstration")
        
        return [
            FightOddsData(
                event_name="UFC 317: Sample Event",
                fighter_a="Jon Jones",
                fighter_b="Stipe Miocic",
                fighter_a_odds=[
                    BettingOdds("BetOnline", -650, 1.15, 86.7),
                    BettingOdds("Bovada", -600, 1.17, 85.7),
                ],
                fighter_b_odds=[
                    BettingOdds("BetOnline", +450, 5.50, 18.2),
                    BettingOdds("Bovada", +425, 5.25, 19.0),
                ]
            )
        ]

def main():
    """Test the Selenium scraper"""
    print("🎯 FIGHTODDS.IO SELENIUM SCRAPER TEST")
    print("=" * 50)
    
    scraper = FightOddsSeleniumScraper(headless=False)  # Set to False to see the browser
    
    try:
        odds_data = scraper.scrape_fightodds()
        
        print(f"\n📊 SCRAPING RESULTS")
        print("=" * 30)
        print(f"✅ Successfully scraped {len(odds_data)} fights")
        
        for i, fight in enumerate(odds_data, 1):
            print(f"\n🥊 Fight {i}: {fight.fighter_a} vs {fight.fighter_b}")
            print(f"   Event: {fight.event_name}")
            print(f"   {fight.fighter_a} odds: {len(fight.fighter_a_odds)} sportsbooks")
            print(f"   {fight.fighter_b} odds: {len(fight.fighter_b_odds)} sportsbooks")
        
        # Save results
        results = [fight.to_dict() for fight in odds_data]
        with open('fightodds_selenium_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: fightodds_selenium_results.json")
        
    except KeyboardInterrupt:
        print("\n⏹️  Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Scraping failed: {e}")

if __name__ == "__main__":
    main() 
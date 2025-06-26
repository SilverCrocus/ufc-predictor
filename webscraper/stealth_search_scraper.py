#!/usr/bin/env python3
"""
Stealth Search TAB Scraper - Bot Detection Bypass
==============================================

Uses DuckDuckGo (more bot-friendly) + stealth techniques to find 
specific TAB fight pages and scrape live odds.

Strategy:
1. Use DuckDuckGo instead of Google (no captcha)
2. Stealth Selenium configuration 
3. Human-like delays and actions
4. Search for "Fighter A vs Fighter B tab.com.au"
5. Click TAB link and scrape clean H2H odds
"""

import time
import json
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from typing import Dict, List, Optional
import random

class StealthTABScraper:
    """Stealth scraper that uses DuckDuckGo to find TAB pages without bot detection"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
        self.wait = None
        
    def setup_stealth_driver(self):
        """Initialize Chrome with maximum stealth"""
        print("🕵️  Setting up stealth WebDriver...")
        
        chrome_options = Options()
        
        # Stealth options to avoid detection
        if self.headless:
            chrome_options.add_argument('--headless')
        
        # Essential stealth arguments
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--disable-features=VizDisplayCompositor')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-plugins')
        chrome_options.add_argument('--disable-images')
        chrome_options.add_argument('--disable-javascript')  # We'll enable later if needed
        
        # Realistic window size
        chrome_options.add_argument('--window-size=1366,768')
        
        # Realistic user agent (recent Chrome on Mac)
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Disable automation flags
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Additional stealth prefs
        prefs = {
            "profile.default_content_setting_values": {
                "notifications": 2,
                "media_stream": 2,
            },
            "profile.managed_default_content_settings": {
                "images": 2
            }
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # Execute stealth script to hide webdriver property
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        self.wait = WebDriverWait(self.driver, 15)
        print("✅ Stealth WebDriver initialized")
        
    def human_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """Add random human-like delay"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
        
    def human_type(self, element, text: str):
        """Type text with human-like delays"""
        for char in text:
            element.send_keys(char)
            time.sleep(random.uniform(0.05, 0.15))
    
    def duckduckgo_search_tab_fight(self, fighter1: str, fighter2: str) -> Optional[str]:
        """Search DuckDuckGo for specific TAB fight page"""
        
        search_query = f'"{fighter1} vs {fighter2}" site:tab.com.au'
        print(f"🔍 DuckDuckGo search: {search_query}")
        
        try:
            # Go to DuckDuckGo
            print("   📍 Loading DuckDuckGo...")
            self.driver.get("https://duckduckgo.com")
            self.human_delay(2, 4)
            
            # Find search box
            search_box = self.wait.until(EC.presence_of_element_located((By.NAME, "q")))
            search_box.clear()
            
            # Type search query with human timing
            print(f"   ⌨️  Typing search query...")
            self.human_type(search_box, search_query)
            self.human_delay(0.5, 1.5)
            
            # Press Enter
            search_box.send_keys(Keys.RETURN)
            self.human_delay(3, 5)
            
            # Look for actual search result links (not DuckDuckGo navigation)
            print("   🔍 Looking for search results...")
            
            # Multiple selectors for DuckDuckGo search results
            result_selectors = [
                "a[data-testid='result-title-a']",  # Main DuckDuckGo result links
                ".result__a",                       # Alternative result links
                "h3 a",                            # Header links in results
                "a[href*='tab.com.au']"            # Direct TAB links
            ]
            
            tab_links = []
            
            for selector in result_selectors:
                try:
                    links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for link in links:
                        href = link.get_attribute('href')
                        # Skip DuckDuckGo's own URLs and look for actual TAB links
                        if (href and 
                            'tab.com.au' in href and 
                            'duckduckgo.com' not in href and
                            not href.startswith('https://duckduckgo.com')):
                            tab_links.append(href)
                            print(f"   ✅ Found TAB result: {href}")
                except Exception as e:
                    print(f"   ⚠️  Selector {selector} failed: {e}")
                    continue
            
            if tab_links:
                # Filter for UFC/MMA related links
                ufc_links = [link for link in tab_links if any(term in link.lower() for term in ['ufc', 'mma', 'fight', 'sports/betting'])]
                if ufc_links:
                    return ufc_links[0]
                else:
                    return tab_links[0]  # Return any TAB link
            else:
                print(f"   ❌ No TAB result links found")
                
                # Fallback: Try without site restriction
                print("   🔄 Trying fallback search without site restriction...")
                return self.fallback_search_methods(fighter1, fighter2)
                
        except Exception as e:
            print(f"   ❌ DuckDuckGo search failed: {e}")
            return None
    
    def fallback_search_methods(self, fighter1: str, fighter2: str) -> Optional[str]:
        """Fallback methods to find TAB pages"""
        
        # Method 1: Try different search engines
        search_engines = [
            ("https://www.bing.com/search?q=", "Bing"),
            ("https://search.yahoo.com/search?p=", "Yahoo")
        ]
        
        for base_url, engine_name in search_engines:
            try:
                print(f"   🔄 Trying {engine_name}...")
                
                search_query = f'"{fighter1} vs {fighter2}" tab.com.au betting odds'
                full_url = f"{base_url}{search_query.replace(' ', '+')}"
                
                self.driver.get(full_url)
                self.human_delay(3, 5)
                
                # Look for TAB links in results
                all_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='tab.com.au']")
                
                for link in all_links:
                    href = link.get_attribute('href')
                    if href and 'sports/betting' in href.lower():
                        print(f"   ✅ {engine_name} found: {href}")
                        return href
                        
            except Exception as e:
                print(f"   ❌ {engine_name} failed: {e}")
                continue
        
        # Method 2: Try direct TAB site search
        return self.try_tab_site_search(fighter1, fighter2)
    
    def try_tab_site_search(self, fighter1: str, fighter2: str) -> Optional[str]:
        """Try searching directly on TAB's website"""
        
        try:
            print("   🔄 Trying TAB site search...")
            
            self.driver.get("https://www.tab.com.au/sports/betting/MMA")
            self.human_delay(3, 5)
            
            # Look for search box or UFC section
            search_selectors = [
                "input[type='search']",
                "[placeholder*='search']",
                "[class*='search']"
            ]
            
            for selector in search_selectors:
                try:
                    search_box = self.driver.find_element(By.CSS_SELECTOR, selector)
                    search_box.clear()
                    self.human_type(search_box, f"{fighter1} {fighter2}")
                    search_box.send_keys(Keys.RETURN)
                    self.human_delay(3, 5)
                    
                    # Look for result links
                    result_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='competitions/UFC']")
                    if result_links:
                        href = result_links[0].get_attribute('href')
                        print(f"   ✅ TAB site search found: {href}")
                        return href
                        
                except:
                    continue
            
            # If no search box, try to find UFC section and browse
            ufc_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='UFC'], a[href*='ufc']")
            for link in ufc_links:
                link_text = link.text.lower()
                if any(name.lower() in link_text for name in [fighter1, fighter2]):
                    href = link.get_attribute('href')
                    print(f"   ✅ TAB browse found: {href}")
                    return href
            
        except Exception as e:
            print(f"   ❌ TAB site search failed: {e}")
        
        return None
    
    def scrape_tab_fight_page_stealth(self, tab_url: str) -> Dict[str, float]:
        """Scrape H2H odds from TAB page with stealth techniques"""
        
        print(f"🎯 Stealthily accessing: {tab_url}")
        
        try:
            # Navigate to TAB page
            self.driver.get(tab_url)
            self.human_delay(5, 8)  # Let page fully load
            
            # Check if we hit a blocking page
            page_title = self.driver.title.lower()
            if any(term in page_title for term in ['blocked', 'access denied', 'not found', 'error']):
                print(f"   ❌ Page blocked or not found: {page_title}")
                return {}
            
            print(f"   📄 Page title: {self.driver.title}")
            
            odds_data = {}
            
            # Strategy 1: Look for visible elements with odds
            print("   🔍 Strategy 1: Looking for visible odds elements...")
            
            # Common selectors for betting odds
            odds_selectors = [
                "[class*='odds']",
                "[class*='price']", 
                "[data-testid*='odds']",
                "[data-testid*='price']",
                "button[class*='odd']",
                ".price",
                ".odds"
            ]
            
            for selector in odds_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        text = elem.get_attribute('textContent') or elem.text
                        if text and re.match(r'^\d+\.\d+$', text.strip()):
                            odds_value = float(text.strip())
                            if 1.0 <= odds_value <= 50.0:
                                # Try to find associated fighter name
                                parent = elem.find_element(By.XPATH, "..")
                                parent_text = parent.get_attribute('textContent') or parent.text
                                odds_data[f"Found odds {odds_value}"] = odds_value
                                print(f"   💰 Found odds: {odds_value} in context: {parent_text[:50]}...")
                except:
                    continue
            
            # Strategy 2: Look for text patterns in page source
            print("   🔍 Strategy 2: Text pattern analysis...")
            page_source = self.driver.page_source
            
            # Pattern for H2H with names and odds
            h2h_pattern = r'H2H\s+([A-Z]+)\s+([A-Za-z]+)[\s\S]*?(\d+\.\d+)'
            h2h_matches = re.findall(h2h_pattern, page_source)
            
            for match in h2h_matches:
                lastname, firstname, odds_str = match
                try:
                    odds_value = float(odds_str)
                    if 1.0 <= odds_value <= 50.0:
                        fighter_name = f"{firstname.title()} {lastname.title()}"
                        market_name = f"H2H {lastname.upper()} {firstname.title()}"
                        odds_data[market_name] = odds_value
                        print(f"   💰 H2H Pattern: {fighter_name} = {odds_value}")
                except ValueError:
                    continue
            
            # Strategy 3: Look for fight names and nearby odds
            print("   🔍 Strategy 3: Fighter name proximity search...")
            
            # Target fighter names from your card
            target_names = ['Topuria', 'Oliveira', 'Pantoja', 'Kara-France', 'Royval', 'Van', 'Dariush', 'Moicano', 'Talbott', 'Lima']
            
            for name in target_names:
                name_pattern = rf'{name}[\s\S]*?(\d+\.\d+)'
                name_matches = re.findall(name_pattern, page_source, re.IGNORECASE)
                for odds_str in name_matches:
                    try:
                        odds_value = float(odds_str)
                        if 1.0 <= odds_value <= 50.0:
                            odds_data[f"Fighter {name.title()}"] = odds_value
                            print(f"   🎯 Name proximity: {name} = {odds_value}")
                            break  # Take first reasonable odds for this fighter
                    except ValueError:
                        continue
            
            # Save page for debugging if we found odds
            if odds_data:
                with open('stealth_tab_page.html', 'w', encoding='utf-8') as f:
                    f.write(page_source)
                print(f"   💾 Saved successful page source")
            
            return odds_data
            
        except Exception as e:
            print(f"   ❌ Error scraping {tab_url}: {e}")
            return {}
    
    def scrape_fight_stealth(self, fighter1: str, fighter2: str) -> Dict[str, float]:
        """Complete stealth workflow for one fight"""
        
        print(f"\n🥊 STEALTH SCRAPING: {fighter1} vs {fighter2}")
        print("=" * 50)
        
        # Search DuckDuckGo for TAB page
        tab_url = self.duckduckgo_search_tab_fight(fighter1, fighter2)
        
        if not tab_url:
            return {}
        
        # Stealth scrape the TAB page
        odds_data = self.scrape_tab_fight_page_stealth(tab_url)
        
        return odds_data
    
    def scrape_multiple_fights_stealth(self, fight_list: List[str]) -> Dict[str, Dict[str, float]]:
        """Scrape multiple fights with stealth"""
        
        print("🕵️  STEALTH TAB SCRAPER - MULTIPLE FIGHTS")
        print("=" * 60)
        print("💡 Using DuckDuckGo + stealth techniques to avoid detection!")
        
        if not self.driver:
            self.setup_stealth_driver()
        
        all_results = {}
        
        for i, fight_string in enumerate(fight_list, 1):
            try:
                print(f"\n🎯 Fight {i}/{len(fight_list)}: {fight_string}")
                
                if " vs. " in fight_string:
                    fighter1, fighter2 = fight_string.split(" vs. ")
                elif " vs " in fight_string:
                    fighter1, fighter2 = fight_string.split(" vs ")
                else:
                    print(f"   ⚠️  Invalid format: {fight_string}")
                    continue
                
                fighter1 = fighter1.strip()
                fighter2 = fighter2.strip()
                
                odds_data = self.scrape_fight_stealth(fighter1, fighter2)
                
                if odds_data:
                    all_results[fight_string] = odds_data
                    print(f"   ✅ Successfully scraped {fight_string}")
                else:
                    print(f"   ❌ No odds found for {fight_string}")
                
                # Human-like delay between searches
                if i < len(fight_list):
                    print(f"   ⏳ Waiting before next search...")
                    self.human_delay(8, 15)
                
            except Exception as e:
                print(f"   ❌ Error with {fight_string}: {e}")
                continue
        
        return all_results
    
    def close(self):
        """Close WebDriver"""
        if self.driver:
            self.driver.quit()
            print("🔒 Stealth WebDriver closed")

def test_stealth_scraper():
    """Test the stealth TAB scraper"""
    
    test_fights = [
        "Ilia Topuria vs. Charles Oliveira",
        "Alexandre Pantoja vs. Kai Kara-France"
    ]
    
    print("🕵️  TESTING STEALTH TAB SCRAPER")
    print("=" * 40)
    print("💡 Strategy: DuckDuckGo search + stealth techniques!")
    print("🚫 No more 'I am not a robot' captchas!")
    
    scraper = StealthTABScraper(headless=False)  # Show browser for testing
    
    try:
        # Test first couple fights
        results = scraper.scrape_multiple_fights_stealth(test_fights)
        
        print(f"\n📊 STEALTH RESULTS:")
        print(f"✅ Successfully scraped: {len(results)}/{len(test_fights)} fights")
        
        for fight, odds_data in results.items():
            print(f"\n🥊 {fight}:")
            for market, odds_value in odds_data.items():
                print(f"   {market}: {odds_value}")
        
        # Save results
        with open('stealth_tab_odds.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to stealth_tab_odds.json")
        
        print(f"\n🎯 STEALTH SOLUTION SUMMARY:")
        print("✅ Uses DuckDuckGo (no captcha)")
        print("✅ Stealth Selenium configuration")
        print("✅ Human-like delays and typing")
        print("✅ Multiple scraping strategies")
        print("✅ Finds exact TAB pages via search")
        print("\n💡 This should bypass bot detection!")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    test_stealth_scraper() 
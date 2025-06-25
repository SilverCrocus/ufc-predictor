"""
Live Test for FightOdds.io Scraper

This script tests the scraper with enhanced error handling and debugging
for the Material-UI structure.
"""

import requests
from bs4 import BeautifulSoup
import ssl
import urllib3
from typing import List, Dict
import json

# Disable SSL warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class FightOddsLiveTester:
    """Enhanced tester for fightodds.io with better error handling"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def test_connection_methods(self, url: str = "https://fightodds.io/"):
        """Test different methods to connect to fightodds.io"""
        print("🔗 TESTING CONNECTION METHODS")
        print("=" * 40)
        
        methods = [
            ("Standard HTTPS", self._try_standard_https),
            ("Ignore SSL Verification", self._try_ignore_ssl),
            ("Custom SSL Context", self._try_custom_ssl),
        ]
        
        for method_name, method_func in methods:
            print(f"\n🔍 Trying: {method_name}")
            try:
                response = method_func(url)
                if response and response.status_code == 200:
                    print(f"✅ SUCCESS with {method_name}")
                    return response
                else:
                    print(f"❌ Failed: Status {response.status_code if response else 'None'}")
            except Exception as e:
                print(f"❌ Failed: {str(e)[:100]}...")
        
        print(f"\n❌ All connection methods failed")
        return None
    
    def _try_standard_https(self, url: str):
        """Try standard HTTPS connection"""
        return self.session.get(url, timeout=10)
    
    def _try_ignore_ssl(self, url: str):
        """Try ignoring SSL verification"""
        return self.session.get(url, verify=False, timeout=10)
    
    def _try_custom_ssl(self, url: str):
        """Try with custom SSL context"""
        # Create custom SSL context
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        # Create new session with custom SSL
        session = requests.Session()
        session.headers.update(self.headers)
        session.mount('https://', requests.adapters.HTTPAdapter())
        
        return session.get(url, verify=False, timeout=10)
    
    def analyze_html_structure(self, html_content: str):
        """Analyze the HTML structure to find MUI components"""
        print("\n🔍 ANALYZING HTML STRUCTURE")
        print("=" * 40)
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for MUI components
        mui_elements = [
            ('nav.MuiList-root.jss1579', 'nav', 'MuiList-root jss1579'),
            ('nav with MuiList-root class', 'nav', lambda x: x and 'MuiList-root' in x),
            ('Any MUI List component', None, lambda x: x and 'MuiList' in x),
            ('Table elements', 'table', None),
            ('Div with odds-related classes', 'div', lambda x: x and any(term in x for term in ['odds', 'bet', 'fight']) if x else False),
        ]
        
        findings = []
        
        for description, tag, class_filter in mui_elements:
            if tag and class_filter:
                if callable(class_filter):
                    elements = soup.find_all(tag, class_=class_filter)
                else:
                    elements = soup.find_all(tag, class_=class_filter)
            elif tag:
                elements = soup.find_all(tag)
            else:
                elements = soup.find_all(class_=class_filter)
            
            if elements:
                print(f"✅ Found {len(elements)} elements: {description}")
                findings.append((description, elements))
                
                # Show first element structure
                if elements:
                    first_elem = elements[0]
                    print(f"   First element: <{first_elem.name} class='{first_elem.get('class')}'>")
                    
                    # Show children
                    children = first_elem.find_all(recursive=False)[:3]  # First 3 children
                    for i, child in enumerate(children):
                        print(f"   Child {i+1}: <{child.name} class='{child.get('class')}'>")
            else:
                print(f"❌ Not found: {description}")
        
        return findings
    
    def extract_sample_fighter_odds(self, soup: BeautifulSoup):
        """Try to extract sample fighter and odds data"""
        print("\n🥊 EXTRACTING FIGHTER AND ODDS DATA")
        print("=" * 40)
        
        # Try the specific MUI selector first
        mui_container = soup.find('nav', class_='MuiList-root jss1579')
        
        if mui_container:
            print("✅ Found MUI container!")
            self._extract_from_mui_container(mui_container)
        else:
            print("❌ MUI container not found, trying alternatives...")
            
            # Try alternative selectors
            alternatives = [
                soup.find('nav', class_=lambda x: x and 'MuiList-root' in x),
                soup.find('div', class_=lambda x: x and 'MuiList' in x),
                soup.find('table'),
            ]
            
            for i, container in enumerate(alternatives):
                if container:
                    print(f"✅ Found alternative container {i+1}: {container.name}")
                    self._extract_from_container(container)
                    break
            else:
                print("❌ No suitable containers found")
    
    def _extract_from_mui_container(self, container):
        """Extract data from MUI container"""
        print("📋 Analyzing MUI container structure...")
        
        # Look for list items
        list_items = container.find_all('li', class_=lambda x: x and 'MuiListItem' in x)
        print(f"   Found {len(list_items)} MuiListItem elements")
        
        if not list_items:
            # Look for direct div children
            list_items = container.find_all('div', recursive=False)
            print(f"   Found {len(list_items)} direct div children")
        
        # Analyze first few items
        for i, item in enumerate(list_items[:5]):  # First 5 items
            print(f"\n   Item {i+1}:")
            
            # Look for text content
            text_elements = item.find_all(string=True)
            text_content = [text.strip() for text in text_elements if text.strip()]
            
            print(f"     Text content: {text_content[:10]}...")  # First 10 text elements
            
            # Look for potential fighter names (longer text, not just numbers)
            potential_fighters = [text for text in text_content 
                                if len(text) > 5 and not text.replace('+', '').replace('-', '').replace('.', '').isdigit()]
            
            if potential_fighters:
                print(f"     Potential fighters: {potential_fighters[:3]}")
            
            # Look for odds (text with + or -)
            potential_odds = [text for text in text_content 
                            if any(c in text for c in ['+', '-']) and any(c.isdigit() for c in text)]
            
            if potential_odds:
                print(f"     Potential odds: {potential_odds[:5]}")
    
    def _extract_from_container(self, container):
        """Extract data from generic container"""
        print(f"📋 Analyzing {container.name} container...")
        
        # Get all text content
        all_text = container.get_text(separator='|', strip=True)
        text_parts = [part.strip() for part in all_text.split('|') if part.strip()]
        
        print(f"   Total text elements: {len(text_parts)}")
        print(f"   Sample text: {text_parts[:10]}")
        
        # Look for fighter-like names and odds
        potential_fighters = [text for text in text_parts 
                            if len(text) > 5 and any(c.isalpha() for c in text) 
                            and not text.replace('+', '').replace('-', '').replace('.', '').isdigit()]
        
        potential_odds = [text for text in text_parts 
                        if any(c in text for c in ['+', '-']) and any(c.isdigit() for c in text)]
        
        print(f"   Potential fighters found: {len(potential_fighters)}")
        print(f"   Potential odds found: {len(potential_odds)}")
        
        if potential_fighters:
            print(f"   Sample fighters: {potential_fighters[:5]}")
        
        if potential_odds:
            print(f"   Sample odds: {potential_odds[:10]}")
    
    def generate_scraper_template(self, findings):
        """Generate custom scraper code based on findings"""
        print("\n📝 GENERATING CUSTOM SCRAPER TEMPLATE")
        print("=" * 40)
        
        template = """
# Custom scraper template based on site analysis

def scrape_fightodds_custom(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Primary selector (update based on findings)
    container = soup.find('nav', class_='MuiList-root jss1579')
    
    if not container:
        # Fallback selectors
        container = (
            soup.find('nav', class_=lambda x: x and 'MuiList-root' in x) or
            soup.find('div', class_=lambda x: x and 'MuiList' in x) or
            soup.find('table')
        )
    
    if container:
        # Extract fighter and odds data
        # TODO: Customize based on actual structure found
        pass
    
    return []
"""
        
        print(template)
        
        # Save template to file
        with open('custom_scraper_template.py', 'w') as f:
            f.write(template)
        
        print("💾 Template saved to: custom_scraper_template.py")

def main():
    """Main testing function"""
    print("🎯 FIGHTODDS.IO LIVE TESTING")
    print("=" * 50)
    
    tester = FightOddsLiveTester()
    
    # Try to connect
    response = tester.test_connection_methods()
    
    if response:
        print(f"\n✅ Successfully connected! Status: {response.status_code}")
        print(f"📄 Content length: {len(response.text)} characters")
        
        # Analyze structure
        findings = tester.analyze_html_structure(response.text)
        
        # Try to extract sample data
        soup = BeautifulSoup(response.text, 'html.parser')
        tester.extract_sample_fighter_odds(soup)
        
        # Generate template
        tester.generate_scraper_template(findings)
        
        # Save raw HTML for inspection
        with open('fightodds_raw.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("\n💾 Raw HTML saved to: fightodds_raw.html")
        
    else:
        print("\n❌ Could not connect to fightodds.io")
        print("💡 You can still use the scraper with manual HTML or sample data")
        print("📋 To customize the scraper:")
        print("   1. Save the page HTML manually")
        print("   2. Look for the nav.MuiList-root.jss1579 element")
        print("   3. Update the selector in fightodds_scraper.py")

if __name__ == "__main__":
    main() 
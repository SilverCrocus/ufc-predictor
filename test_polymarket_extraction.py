#!/usr/bin/env python3
"""
Test Polymarket Extraction Approaches
====================================

This script tests different approaches to extract UFC betting odds from Polymarket
without requiring Playwright, to verify our extraction logic.
"""

import requests
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

def test_static_extraction(url: str) -> Dict:
    """Test static HTML extraction to understand page structure"""
    
    results = {
        'page_loaded': False,
        'fighters_found': [],
        'betting_patterns': [],
        'market_indicators': [],
        'json_data': [],
        'recommendations': []
    }
    
    try:
        # Fetch page with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        results['page_loaded'] = True
        
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text()
        
        # Look for known fighters
        known_fighters = [
            'Whittaker', 'de Ridder', 'Ridder',
            'Yan', 'McGhee', 
            'Magomedov', 'Barriault',
            'Almabayev', 'Ochoa',
            'Krylov', 'Guskov'
        ]
        
        for fighter in known_fighters:
            if fighter.lower() in page_text.lower():
                results['fighters_found'].append(fighter)
        
        # Look for betting patterns
        cent_pattern = re.findall(r'\d+¢', page_text)
        percent_pattern = re.findall(r'\d+(?:\.\d+)?%', page_text)
        decimal_pattern = re.findall(r'0\.\d+', page_text)
        
        results['betting_patterns'] = {
            'cents': cent_pattern[:10],
            'percentages': percent_pattern[:10], 
            'decimals': decimal_pattern[:10]
        }
        
        # Look for market indicators in HTML
        market_classes = soup.find_all(attrs={'class': re.compile(r'market|Market|bet|Bet|outcome|Outcome', re.I)})
        results['market_indicators'] = [elem.get('class') for elem in market_classes[:5]]
        
        # Look for JSON data in script tags
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string and ('market' in script.string.lower() or 'bet' in script.string.lower()):
                # Extract small snippet
                snippet = script.string[:200] + '...' if len(script.string) > 200 else script.string
                results['json_data'].append(snippet)
        
        # Analyze what extraction approaches would work
        if results['fighters_found'] and results['betting_patterns']['cents']:
            results['recommendations'].append("✅ Fighter names and cent patterns found - text extraction viable")
        elif results['fighters_found']:
            results['recommendations'].append("⚠️ Fighter names found but no betting patterns - may need JavaScript rendering")
        else:
            results['recommendations'].append("❌ Limited data found - definitely needs JavaScript rendering (Playwright)")
        
        if results['market_indicators']:
            results['recommendations'].append("✅ Market-related CSS classes found - DOM selectors viable")
        
        if results['json_data']:
            results['recommendations'].append("✅ JSON data in script tags - structured data extraction possible")
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def analyze_playwright_requirements(results: Dict) -> List[str]:
    """Analyze if Playwright is required for successful scraping"""
    
    requirements = []
    
    if not results.get('page_loaded'):
        requirements.append("❌ Page failed to load - check connectivity")
        return requirements
    
    fighters_found = len(results.get('fighters_found', []))
    betting_patterns = results.get('betting_patterns', {})
    
    if fighters_found >= 4 and any(betting_patterns.values()):
        requirements.append("✅ Static extraction might work - consider requests + BeautifulSoup first")
        requirements.append("✅ Playwright recommended for reliability and completeness")
    elif fighters_found >= 2:
        requirements.append("⚠️ Partial data in static HTML - Playwright likely required")
        requirements.append("⚠️ Consider hybrid approach: static check first, then Playwright")
    else:
        requirements.append("❌ Minimal data in static HTML - Playwright definitely required")
        requirements.append("❌ Page heavily JavaScript-dependent")
    
    if results.get('json_data'):
        requirements.append("✅ Script tag data available - direct JSON extraction possible")
    
    return requirements

def suggest_extraction_strategy(results: Dict) -> List[str]:
    """Suggest optimal extraction strategy based on analysis"""
    
    strategies = []
    
    # Based on current scraper analysis
    strategies.append("🔧 Current scraper improvements:")
    
    if results.get('betting_patterns', {}).get('cents'):
        strategies.append("  ✅ Current cent pattern matching should work")
    else:
        strategies.append("  ⚠️ May need to adjust pattern matching")
    
    if len(results.get('fighters_found', [])) >= 4:
        strategies.append("  ✅ Fighter name detection working well")
    else:
        strategies.append("  ⚠️ May need better fighter name fallbacks")
    
    # Specific selector recommendations
    strategies.append("🎯 Recommended selector improvements:")
    
    market_classes = results.get('market_indicators', [])
    if market_classes:
        strategies.append(f"  • Try CSS classes found: {market_classes[:2]}")
    
    strategies.append("  • Add data-testid selectors")
    strategies.append("  • Look for aria-label attributes")
    strategies.append("  • Search for button[class*='buy'] elements")
    
    # Wait strategies
    strategies.append("⏱️ Timing adjustments:")
    strategies.append("  • Increase wait time from 7s to 10-15s")
    strategies.append("  • Add wait for specific market elements")
    strategies.append("  • Use wait_for_function for dynamic content")
    
    return strategies

if __name__ == "__main__":
    url = "https://polymarket.com/event/ufc-fight-night-whittaker-vs-de-ridder?tid=1753515605835"
    
    print("🧪 Testing Polymarket Extraction Approaches")
    print("=" * 60)
    print(f"🌐 Target URL: {url}")
    print()
    
    # Test static extraction
    print("1. Testing static HTML extraction...")
    results = test_static_extraction(url)
    
    print(f"   Page loaded: {'✅' if results['page_loaded'] else '❌'}")
    print(f"   Fighters found: {results['fighters_found']}")
    print(f"   Betting patterns: {results['betting_patterns']}")
    print(f"   Market indicators: {len(results['market_indicators'])} found")
    print(f"   JSON data: {len(results['json_data'])} script tags")
    
    if 'error' in results:
        print(f"   Error: {results['error']}")
    
    print()
    
    # Analyze requirements
    print("2. Playwright Requirements Analysis:")
    requirements = analyze_playwright_requirements(results)
    for req in requirements:
        print(f"   {req}")
    
    print()
    
    # Suggest strategy
    print("3. Extraction Strategy Recommendations:")
    strategies = suggest_extraction_strategy(results)
    for strategy in strategies:
        print(f"   {strategy}")
    
    print()
    print("4. Next Steps:")
    print("   • Install Playwright: pip install playwright")
    print("   • Install browsers: playwright install")
    print("   • Run your existing debug script")
    print("   • Test current scraper with increased timeouts")
    print("   • Consider implementing fallback to static extraction")
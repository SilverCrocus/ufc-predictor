#!/usr/bin/env python3
"""
Debug Polymarket Selectors
=========================

This script loads the Polymarket page with Playwright and inspects 
the actual DOM structure to identify the correct selectors for scraping.
"""

import asyncio
from playwright.async_api import async_playwright

async def debug_polymarket_page():
    """Debug the actual DOM structure of the Polymarket page"""
    
    url = "https://polymarket.com/event/ufc-fight-night-whittaker-vs-de-ridder?tid=1753515605835"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Show browser for debugging
        page = await browser.new_page()
        
        # Set reasonable viewport and headers
        await page.set_viewport_size({"width": 1920, "height": 1080})
        
        print(f"🌐 Loading: {url}")
        await page.goto(url, wait_until='domcontentloaded', timeout=45000)
        
        # Wait for page to load completely
        print("⏳ Waiting for page to render...")
        await page.wait_for_timeout(10000)
        
        # Try to take a screenshot for visual debugging
        await page.screenshot(path='polymarket_debug.png')
        print("📸 Screenshot saved as polymarket_debug.png")
        
        # Get all elements that might contain market data
        print("\n🔍 ANALYZING DOM STRUCTURE...")
        print("=" * 50)
        
        # Check for various potential market selectors
        selectors_to_test = [
            'div[class*="market"]',
            'div[class*="Market"]', 
            'div[class*="prediction"]',
            'div[class*="Prediction"]',
            'div[class*="bet"]',
            'div[class*="Bet"]',
            'div[class*="outcome"]',
            'div[class*="Outcome"]',
            '[data-testid*="market"]',
            '[data-testid*="prediction"]',
            '[data-testid*="bet"]',
            'button[class*="market"]',
            'button[class*="Market"]',
            'div[class*="card"]',
            'div[class*="Card"]',
            'article',
            'section',
            # Common text patterns
            'div:has-text("vs")',
            'div:has-text("Whittaker")',
            'div:has-text("de Ridder")',
            'div:has-text("%")',
            'div:has-text("¢")',
        ]
        
        found_elements = {}
        
        for selector in selectors_to_test:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    found_elements[selector] = len(elements)
                    print(f"✅ {selector}: {len(elements)} elements")
                else:
                    print(f"❌ {selector}: 0 elements")
            except Exception as e:
                print(f"🚫 {selector}: Error - {e}")
        
        # Get page text content to see what's actually there
        print(f"\n📄 PAGE TEXT CONTENT (first 2000 chars):")
        print("=" * 50)
        page_text = await page.text_content('body')
        print(page_text[:2000])
        
        # Look for specific fighter names
        print(f"\n🥊 FIGHTER NAME SEARCH:")
        print("=" * 50)
        fighters = ["Whittaker", "de Ridder", "Yan", "McGhee", "Magomedov", "Barriault"]
        for fighter in fighters:
            if fighter.lower() in page_text.lower():
                print(f"✅ Found: {fighter}")
            else:
                print(f"❌ Missing: {fighter}")
        
        # Check for probability indicators
        print(f"\n📊 PROBABILITY INDICATORS:")
        print("=" * 50)
        import re
        percent_matches = re.findall(r'\d+%', page_text)
        cent_matches = re.findall(r'\d+¢', page_text)
        decimal_matches = re.findall(r'0\.\d+', page_text)
        
        print(f"Percentage values: {percent_matches[:10]}")  # First 10
        print(f"Cent values: {cent_matches[:10]}")
        print(f"Decimal values: {decimal_matches[:10]}")
        
        # Get the actual HTML structure of promising elements
        if found_elements:
            print(f"\n🔧 DETAILED ELEMENT ANALYSIS:")
            print("=" * 50)
            
            # Analyze the most promising selector
            best_selector = max(found_elements.items(), key=lambda x: x[1])
            selector, count = best_selector
            
            print(f"Analyzing top selector: {selector} ({count} elements)")
            
            elements = await page.query_selector_all(selector)
            for i, element in enumerate(elements[:3]):  # First 3 elements
                try:
                    text = await element.text_content()
                    html = await element.inner_html()
                    print(f"\n--- Element {i+1} ---")
                    print(f"Text: {text[:200]}")
                    print(f"HTML: {html[:300]}")
                except:
                    print(f"Could not analyze element {i+1}")
        
        # Keep browser open for manual inspection
        print(f"\n⏸️  Browser kept open for manual inspection.")
        print("    Press Enter to close and continue...")
        input()
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(debug_polymarket_page())
#!/usr/bin/env python3
"""
Extract Polymarket DOM Structure
===============================

This script analyzes the rendered Polymarket page to find the exact 
selectors needed for successful scraping.
"""

import asyncio
from playwright.async_api import async_playwright
import re

async def extract_polymarket_structure():
    """Extract the actual DOM structure for building better selectors"""
    
    url = "https://polymarket.com/event/ufc-fight-night-whittaker-vs-de-ridder?tid=1753515605835"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        await page.set_viewport_size({"width": 1920, "height": 1080})
        
        print(f"🌐 Loading: {url}")
        await page.goto(url, wait_until='domcontentloaded', timeout=45000)
        
        print("⏳ Waiting for page to render...")
        await page.wait_for_timeout(10000)
        
        # Extract detailed structure for elements containing cent values
        print("\n🔍 ANALYZING CENT VALUE ELEMENTS...")
        print("=" * 50)
        
        cent_elements = await page.query_selector_all('div:has-text("¢")')
        print(f"Found {len(cent_elements)} elements with cent values")
        
        fight_data = []
        
        for i, element in enumerate(cent_elements[:20]):  # First 20 elements
            try:
                text = await element.text_content()
                html = await element.inner_html()
                
                # Look for fighter names and cent values in this element
                if any(fighter in text for fighter in ["Whittaker", "de Ridder", "Yan", "McGhee", "Magomedov", "Barriault"]):
                    print(f"\n--- FIGHT ELEMENT {i+1} ---")
                    print(f"Text: {text[:500]}")
                    print(f"HTML snippet: {html[:300]}")
                    
                    # Extract class names
                    class_attr = await element.get_attribute('class')
                    print(f"Classes: {class_attr}")
                    
                    # Check parent element
                    parent = await element.query_selector('xpath=..')
                    if parent:
                        parent_class = await parent.get_attribute('class')
                        parent_text = await parent.text_content()
                        print(f"Parent classes: {parent_class}")
                        print(f"Parent text: {parent_text[:200]}")
                    
                    fight_data.append({
                        'text': text,
                        'html': html,
                        'classes': class_attr
                    })
                    
            except Exception as e:
                print(f"Error analyzing element {i+1}: {e}")
        
        # Look for button elements that might contain betting actions
        print("\n\n🎯 ANALYZING BUTTON ELEMENTS...")
        print("=" * 50)
        
        button_elements = await page.query_selector_all('button')
        
        for i, button in enumerate(button_elements[:30]):  # First 30 buttons
            try:
                text = await button.text_content()
                
                # Look for betting-related buttons
                if any(keyword in text.lower() for keyword in ["buy", "bet", "¢", "%"]) and text.strip():
                    print(f"\n--- BUTTON {i+1} ---")
                    print(f"Text: {text}")
                    
                    class_attr = await button.get_attribute('class')
                    print(f"Classes: {class_attr}")
                    
                    html = await button.inner_html()
                    print(f"HTML: {html[:200]}")
                    
            except Exception as e:
                print(f"Error analyzing button {i+1}: {e}")
        
        # Try to find the actual market containers using JavaScript
        print("\n\n🧠 JAVASCRIPT EXTRACTION...")
        print("=" * 50)
        
        market_data = await page.evaluate('''
            () => {
                const results = [];
                
                // Look for elements containing fighter names and prices
                const fighters = ["Whittaker", "de Ridder", "Yan", "McGhee", "Magomedov", "Barriault"];
                
                // Search through all elements
                const allElements = document.querySelectorAll('*');
                
                for (const element of allElements) {
                    const text = element.textContent || '';
                    
                    // If element contains a fighter name and a cent value
                    if (fighters.some(fighter => text.includes(fighter)) && text.includes('¢')) {
                        
                        // Extract cent values from this element's text
                        const centMatches = text.match(/(\d+)¢/g) || [];
                        
                        // Look for vs pattern
                        const vsMatch = text.match(/([A-Za-z\s]+)\s+vs\.?\s+([A-Za-z\s]+)/);
                        
                        if (centMatches.length > 0) {
                            results.push({
                                text: text.substring(0, 300),
                                centValues: centMatches,
                                vsMatch: vsMatch ? [vsMatch[1].trim(), vsMatch[2].trim()] : null,
                                tagName: element.tagName,
                                className: element.className,
                                id: element.id,
                                querySelector: element.outerHTML.substring(0, 150)
                            });
                        }
                    }
                }
                
                return results;
            }
        ''')
        
        print(f"Found {len(market_data)} elements with fighter names and prices:")
        
        for i, data in enumerate(market_data[:10]):  # First 10 results
            print(f"\n--- MARKET ELEMENT {i+1} ---")
            print(f"Tag: {data['tagName']}")
            print(f"Class: {data['className']}")
            print(f"Cent Values: {data['centValues']}")
            print(f"VS Match: {data['vsMatch']}")
            print(f"Text: {data['text']}")
            print(f"HTML: {data['querySelector']}")
        
        await browser.close()
        
        return fight_data, market_data

if __name__ == "__main__":
    asyncio.run(extract_polymarket_structure())
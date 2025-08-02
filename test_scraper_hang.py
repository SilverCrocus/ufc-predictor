#!/usr/bin/env python3
"""
Minimal scraper test to isolate hanging issue
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import string
import time
import os
from pathlib import Path

print("Starting minimal scraper test...")

# Configuration
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
BASE_URL = "http://ufcstats.com/statistics/fighters"

print("Configuration loaded...")

def test_get_fighter_urls():
    """Test getting fighter URLs for just one letter"""
    print("Testing fighter URL collection...")
    
    fighter_urls = set()
    
    # Test just one letter
    letter = 'a'
    url = f"{BASE_URL}?char={letter}&page=all"
    print(f"Requesting: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        print(f"Response received: {response.status_code}")
    except Exception as e:
        print(f"Request failed: {e}")
        return []

    print("Parsing response...")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all fighter links on the page
    fighter_links = soup.select('tr.b-statistics__table-row td:first-child a')
    print(f"Found {len(fighter_links)} fighter links")
    
    if not fighter_links:
        print("No fighter links found!")
        return []
        
    for link in fighter_links[:5]:  # Just test first 5
        if link.has_attr('href'):
            fighter_urls.add(link['href'])
            print(f"Added URL: {link['href']}")
    
    print(f"Collected {len(fighter_urls)} URLs")
    return list(fighter_urls)

def test_scrape_single_fighter(fighter_url):
    """Test scraping a single fighter"""
    print(f"Testing single fighter scrape: {fighter_url}")
    
    try:
        response = requests.get(fighter_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        print(f"Fighter page response: {response.status_code}")
    except Exception as e:
        print(f"Fighter request failed: {e}")
        return None, None
    
    print("Parsing fighter page...")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get fighter name
    name_element = soup.select_one('span.b-content__title-highlight')
    fighter_name = name_element.text.strip() if name_element else "Unknown"
    print(f"Fighter name: {fighter_name}")
    
    return {"Name": fighter_name}, []

def main():
    print("=== MINIMAL SCRAPER TEST ===")
    
    # Test 1: Get fighter URLs
    print("\n--- Test 1: Getting fighter URLs ---")
    fighter_urls = test_get_fighter_urls()
    
    if not fighter_urls:
        print("No URLs collected, stopping test")
        return
    
    # Test 2: Scrape one fighter
    print("\n--- Test 2: Scraping single fighter ---")
    test_url = fighter_urls[0]
    details, history = test_scrape_single_fighter(test_url)
    
    if details:
        print(f"Successfully scraped: {details}")
    else:
        print("Failed to scrape fighter")
    
    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    main()
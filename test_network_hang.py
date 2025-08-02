#!/usr/bin/env python3
"""
Test script to isolate network hanging issue
"""
import requests
import time
from bs4 import BeautifulSoup

print("Starting network test...")

# Test 1: Basic HTTP request
try:
    print("Test 1: Making basic HTTP request...")
    response = requests.get("http://httpbin.org/get", timeout=5)
    print(f"Basic request successful: {response.status_code}")
except Exception as e:
    print(f"Basic request failed: {e}")

# Test 2: UFC Stats request (actual target)
try:
    print("Test 2: Making UFC Stats request...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url = "http://ufcstats.com/statistics/fighters"
    response = requests.get(url, headers=headers, timeout=10)
    print(f"UFC Stats request successful: {response.status_code}")
    
    # Test parsing
    soup = BeautifulSoup(response.text, 'html.parser')
    print(f"BeautifulSoup parsing successful, title: {soup.title.text if soup.title else 'No title'}")
    
except Exception as e:
    print(f"UFC Stats request failed: {e}")

# Test 3: Try with specific letter page
try:
    print("Test 3: Making specific letter request...")
    url = "http://ufcstats.com/statistics/fighters?char=a&page=all"
    response = requests.get(url, headers=headers, timeout=10)
    print(f"Letter 'a' request successful: {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    fighter_links = soup.select('tr.b-statistics__table-row td:first-child a')
    print(f"Found {len(fighter_links)} fighter links")
    
except Exception as e:
    print(f"Letter request failed: {e}")

print("Network test completed.")
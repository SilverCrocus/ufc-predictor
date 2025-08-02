#!/usr/bin/env python3
"""
Test the specific imports causing hanging in optimized scraper
"""
import sys
from pathlib import Path

print("Testing optimized scraper imports...")

# Add path like the optimized scraper does
sys.path.append(str(Path(__file__).parent))

print("Path added...")

# Test basic imports first
try:
    import asyncio
    print("✓ asyncio imported")
    import aiohttp
    print("✓ aiohttp imported")
    import aiofiles
    print("✓ aiofiles imported")
    import pandas as pd
    print("✓ pandas imported")
    import requests
    print("✓ requests imported")
    from bs4 import BeautifulSoup
    print("✓ BeautifulSoup imported")
    import string
    print("✓ string imported")
    import time
    print("✓ time imported")
    import os
    print("✓ os imported")
    import json
    print("✓ json imported")
    import random
    print("✓ random imported")
    from pathlib import Path
    print("✓ pathlib imported")
    from typing import List, Dict, Tuple, Optional
    print("✓ typing imported")
    from concurrent.futures import ThreadPoolExecutor
    print("✓ ThreadPoolExecutor imported")
    from dataclasses import dataclass
    print("✓ dataclass imported")
    from datetime import datetime
    print("✓ datetime imported")
    import logging
    print("✓ logging imported")
except Exception as e:
    print(f"✗ Basic import failed: {e}")
    sys.exit(1)

print("\nTesting enhanced component imports...")

# Test the problematic imports
try:
    from src.fast_odds_fetcher import FastOddsFetcher
    print("✓ FastOddsFetcher imported")
except Exception as e:
    print(f"✗ FastOddsFetcher import failed: {e}")

try:
    from src.optimized_feature_engineering import OptimizedFeatureEngineering
    print("✓ OptimizedFeatureEngineering imported")
except Exception as e:
    print(f"✗ OptimizedFeatureEngineering import failed: {e}")

try:
    from src.logging_config import setup_logging
    print("✓ setup_logging imported")
except Exception as e:
    print(f"✗ setup_logging import failed: {e}")

print("\nTesting configuration creation...")

# Test creating the configuration
try:
    from dataclasses import dataclass
    
    @dataclass
    class ScrapingConfig:
        FIGHTER_DELAY: float = 0.5
        INDEX_DELAY: float = 0.3
        MAX_CONCURRENT_FIGHTERS: int = 8
        MAX_CONCURRENT_LETTERS: int = 4
        BATCH_SIZE: int = 200
        MAX_RETRIES: int = 3
        RETRY_DELAY: float = 1.0
        REQUEST_TIMEOUT: int = 30
        REQUESTS_PER_SECOND: float = 2.0
        USER_AGENTS: List[str] = None
        
        def __post_init__(self):
            if self.USER_AGENTS is None:
                self.USER_AGENTS = [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                ]
    
    config = ScrapingConfig()
    print("✓ Configuration created")
    
except Exception as e:
    print(f"✗ Configuration creation failed: {e}")

print("\nTesting class instantiation...")

# Test the key problematic area - the class instantiation
try:
    # Simulate the problem area
    BASE_URL = "http://ufcstats.com/statistics/fighters"
    SNAPSHOT_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')
    SNAPSHOT_DATETIME = pd.to_datetime('today').strftime('%Y-%m-%d_%H-%M')
    
    print(f"✓ Constants set: {SNAPSHOT_DATE}")
    
    class TestOptimizedUFCStatsScraper:
        def __init__(self, config = None):
            self.config = config or ScrapingConfig()
            self.session = None
            self.semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_FIGHTERS)
            self.rate_limiter = asyncio.Semaphore(int(self.config.REQUESTS_PER_SECOND))
            print("✓ Scraper class initialized")
    
    scraper = TestOptimizedUFCStatsScraper()
    print("✓ Scraper instance created")
    
except Exception as e:
    print(f"✗ Class instantiation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== IMPORT TEST COMPLETED ===")
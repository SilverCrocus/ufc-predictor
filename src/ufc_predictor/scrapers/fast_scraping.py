# FILE: fast_scraping.py
#
# PURPOSE:
# Optimized UFC scraper that uses only existing dependencies (requests, beautifulsoup4, pandas)
# Implements Phase 1 optimization (reduced delays) for immediate 70% performance improvement
#
# This version works with your existing requirements.txt and provides significant speed gains
# without requiring new async dependencies.

import pandas as pd
import requests
from bs4 import BeautifulSoup
import string
import time
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import threading
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import ELO system for automatic updates
try:
    from src.ufc_predictor.utils.ufc_elo_system import UFCELOSystem
    from src.ufc_predictor.utils.elo_historical_processor import UFCELOHistoricalProcessor
    ELO_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        from ufc_predictor.utils.ufc_elo_system import UFCELOSystem
        from ufc_predictor.utils.elo_historical_processor import UFCELOHistoricalProcessor
        ELO_AVAILABLE = True
    except ImportError:
        print("⚠️ ELO system not available - ELO ratings will not be updated")
        ELO_AVAILABLE = False

# --- Configuration ---
class FastScrapingConfig:
    """Optimized scraping configuration using existing dependencies"""
    
    # Phase 1: Reduced delays (70% improvement)
    FIGHTER_DELAY: float = 0.5    # Reduced from 1.5s (70% improvement)
    INDEX_DELAY: float = 0.3      # Reduced from 1s
    
    # Threading configuration
    MAX_WORKER_THREADS: int = 8   # Concurrent threads
    BATCH_SIZE: int = 200         # Process fighters in batches
    
    # Retry and error handling
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    REQUEST_TIMEOUT: int = 30
    
    # Rate limiting
    REQUESTS_PER_SECOND: float = 2.0
    
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]

# --- Global Configuration ---
CONFIG = FastScrapingConfig()
BASE_URL = "http://ufcstats.com/statistics/fighters"
SNAPSHOT_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')
SNAPSHOT_DATETIME = pd.to_datetime('today').strftime('%Y-%m-%d_%H-%M')

# Thread-safe rate limiter
class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.last_request_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            min_interval = 1.0 / self.requests_per_second
            
            if time_since_last_request < min_interval:
                sleep_time = min_interval - time_since_last_request
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()

# Global rate limiter
rate_limiter = RateLimiter(CONFIG.REQUESTS_PER_SECOND)

# --- Fast Scraper Implementation ---

class FastUFCStatsScraper:
    """High-performance UFC scraper using threads and optimized delays"""
    
    def __init__(self, config: FastScrapingConfig = None):
        self.config = config or CONFIG
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(self.config.USER_AGENTS)})
        
        # Statistics tracking
        self.stats = {
            'fighters_scraped': 0,
            'fights_scraped': 0,
            'requests_made': 0,
            'errors_encountered': 0,
            'start_time': None,
            'end_time': None
        }
        self.stats_lock = threading.Lock()
    
    def update_stats(self, fighters: int = 0, fights: int = 0, requests: int = 0, errors: int = 0):
        """Thread-safe stats update"""
        with self.stats_lock:
            self.stats['fighters_scraped'] += fighters
            self.stats['fights_scraped'] += fights
            self.stats['requests_made'] += requests
            self.stats['errors_encountered'] += errors
    
    def make_request(self, url: str, delay: float = None) -> Optional[str]:
        """Make rate-limited HTTP request with retry logic"""
        rate_limiter.wait_if_needed()
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                self.update_stats(requests=1)
                
                response = self.session.get(url, timeout=self.config.REQUEST_TIMEOUT)
                
                if response.status_code == 200:
                    # Respectful delay
                    actual_delay = delay or self.config.FIGHTER_DELAY
                    time.sleep(actual_delay + random.uniform(0.1, 0.3))
                    return response.text
                    
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt * self.config.RETRY_DELAY
                    print(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                else:
                    print(f"HTTP {response.status_code} for {url}")
                    
            except Exception as e:
                self.update_stats(errors=1)
                print(f"Request failed (attempt {attempt+1}): {e}")
                
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY * (attempt + 1))
        
        return None
    
    def get_fighter_urls_for_letter(self, letter: str) -> List[str]:
        """Get all fighter URLs for a specific letter"""
        url = f"{BASE_URL}?char={letter}&page=all"
        content = self.make_request(url, self.config.INDEX_DELAY)
        
        if not content:
            return []
            
        soup = BeautifulSoup(content, 'html.parser')
        fighter_links = soup.select('tr.b-statistics__table-row td:first-child a')
        
        urls = []
        for link in fighter_links:
            if link.has_attr('href'):
                urls.append(link['href'])
        
        print(f"📋 Letter '{letter.upper()}': {len(urls)} fighters")
        return urls
    
    def get_all_fighter_urls_threaded(self) -> List[str]:
        """Get all fighter URLs using threaded requests for A-Z letters"""
        print("🔍 Discovering fighter URLs with threading...")
        
        letters = list(string.ascii_lowercase)
        all_urls = []
        
        # Use threading for letter discovery (moderate concurrency)
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_letter = {executor.submit(self.get_fighter_urls_for_letter, letter): letter 
                               for letter in letters}
            
            for future in as_completed(future_to_letter):
                letter = future_to_letter[future]
                try:
                    urls = future.result()
                    all_urls.extend(urls)
                except Exception as e:
                    print(f"Letter '{letter}' failed: {e}")
        
        print(f"✅ Discovered {len(all_urls)} fighter URLs total")
        return all_urls
    
    def scrape_fighter_data(self, fighter_url: str) -> Tuple[Optional[Dict], Optional[List]]:
        """Scrape individual fighter data"""
        content = self.make_request(fighter_url)
        if not content:
            return None, None
        
        # Parse fighter details and fight history
        soup = BeautifulSoup(content, 'html.parser')
        
        # Fighter details
        fighter_details = {'fighter_url': fighter_url}
        
        # Name
        name_element = soup.select_one('span.b-content__title-highlight')
        fighter_details['Name'] = name_element.text.strip() if name_element else None
        
        # Record
        record_element = soup.select_one('span.b-content__title-record')
        if record_element:
            fighter_details['Record'] = record_element.get_text(strip=True).replace('Record:', '').strip()
        
        # Other details
        detail_elements = soup.select('ul.b-list__box-list li.b-list__box-list-item')
        for item in detail_elements:
            text_content = item.get_text(separator=":", strip=True)
            if ":" in text_content:
                key, value = text_content.split(":", 1)
                fighter_details[key.strip()] = value.strip()
        
        # Fight history
        fight_history = []
        history_table = soup.select_one('table.b-fight-details__table')
        if history_table:
            rows = history_table.select('tr.b-fight-details__table-row[onclick]')
            for row in rows:
                cols = row.select('td')
                if len(cols) > 1:
                    fighters = cols[1].select('a')
                    fighter_name = fighters[0].get_text(strip=True) if len(fighters) > 0 else None
                    opponent_name = fighters[1].get_text(strip=True) if len(fighters) > 1 else None
                    
                    if opponent_name:
                        fight = {
                            'Outcome': cols[0].get_text(strip=True),
                            'Fighter': fighter_name,
                            'Opponent': opponent_name,
                            'Event': cols[6].get_text(strip=True),
                            'Method': cols[7].get_text(strip=True),
                            'Round': cols[8].get_text(strip=True),
                            'Time': cols[9].get_text(strip=True),
                            'fighter_url': fighter_url
                        }
                        fight_history.append(fight)
        
        self.update_stats(fighters=1, fights=len(fight_history))
        return fighter_details, fight_history
    
    def scrape_fighters_batch_threaded(self, fighter_urls: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Process fighters using thread pool for concurrency"""
        print(f"🏃 Scraping {len(fighter_urls)} fighters with threading (batch size: {self.config.BATCH_SIZE})")
        
        all_fighter_details = []
        all_fight_histories = []
        
        # Process in batches to manage memory and respect server limits
        for i in range(0, len(fighter_urls), self.config.BATCH_SIZE):
            batch = fighter_urls[i:i + self.config.BATCH_SIZE]
            batch_num = i // self.config.BATCH_SIZE + 1
            total_batches = (len(fighter_urls) + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
            
            print(f"📦 Processing batch {batch_num}/{total_batches}: {len(batch)} fighters")
            
            start_time = time.time()
            
            # Use ThreadPoolExecutor for this batch
            with ThreadPoolExecutor(max_workers=self.config.MAX_WORKER_THREADS) as executor:
                future_to_url = {executor.submit(self.scrape_fighter_data, url): url 
                                for url in batch}
                
                batch_fighters = 0
                batch_fights = 0
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        details, history = future.result()
                        if details:
                            all_fighter_details.append(details)
                            batch_fighters += 1
                        if history:
                            all_fight_histories.extend(history)
                            batch_fights += len(history)
                    except Exception as e:
                        print(f"Fighter {url} failed: {e}")
                        self.update_stats(errors=1)
            
            batch_time = time.time() - start_time
            print(f"✅ Batch {batch_num} complete: {batch_fighters} fighters, {batch_fights} fights ({batch_time:.1f}s)")
            
            # Brief pause between batches
            if i + self.config.BATCH_SIZE < len(fighter_urls):
                time.sleep(1.0)
        
        return all_fighter_details, all_fight_histories
    
    def basic_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic feature engineering using existing pandas functionality"""
        print("⚙️ Performing basic feature engineering...")
        
        # Parse height to inches
        if 'Height' in df.columns:
            def parse_height(height_str):
                if pd.isna(height_str):
                    return None
                try:
                    parts = height_str.replace('"', '').split("' ")
                    return int(parts[0]) * 12 + int(parts[1])
                except:
                    return None
            
            df['Height_inches'] = df['Height'].apply(parse_height)
        
        # Parse weight to lbs
        if 'Weight' in df.columns:
            df['Weight_lbs'] = pd.to_numeric(
                df['Weight'].str.replace(' lbs.', ''), errors='coerce'
            )
        
        # Parse reach to inches
        if 'Reach' in df.columns:
            df['Reach_inches'] = pd.to_numeric(
                df['Reach'].str.replace('"', ''), errors='coerce'
            )
        
        # Parse percentage columns
        percent_cols = ['Str. Acc.', 'Str. Def.', 'TD Acc.', 'TD Def.']
        for col in percent_cols:
            if col in df.columns:
                df[col + '_numeric'] = pd.to_numeric(
                    df[col].str.replace('%', ''), errors='coerce'
                ) / 100.0
        
        # Parse numeric columns
        numeric_cols = ['SLpM', 'SApM', 'TD Avg.', 'Sub. Avg.']
        for col in numeric_cols:
            if col in df.columns:
                df[col + '_numeric'] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✅ Feature engineering complete: {df.shape[0]} fighters, {df.shape[1]} features")
        return df
    
    def update_elo_ratings(self, fights_df: pd.DataFrame) -> bool:
        """Update ELO ratings based on newly scraped fight data"""
        if not ELO_AVAILABLE:
            print("⚠️ ELO system not available - skipping ELO update")
            return False
        
        try:
            print("🎯 Updating ELO ratings with latest fight results...")
            
            # Initialize ELO processor
            project_root = Path(__file__).parent.parent.parent.parent
            processor = UFCELOHistoricalProcessor(
                data_dir=str(project_root / "data"),
                output_dir=str(project_root)
            )
            
            # Build ELO ratings from complete fight history
            elo_system = processor.build_from_fight_history(fights_df)
            
            # Save updated ELO ratings
            elo_ratings_path = project_root / "ufc_fighter_elo_ratings.csv"
            processor.save_elo_ratings(elo_system, str(elo_ratings_path))
            
            # Get statistics
            active_fighters = len([f for f in elo_system.fighters.values() if f.is_active])
            total_fighters = len(elo_system.fighters)
            
            print(f"✅ ELO ratings updated successfully!")
            print(f"   - Total fighters: {total_fighters}")
            print(f"   - Active fighters: {active_fighters}")
            print(f"   - Ratings saved to: {elo_ratings_path}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ ELO update failed: {e}")
            print("   Continuing without ELO update...")
            return False
    
    def save_data(self, fighter_details: List[Dict], fight_histories: List[Dict], 
                  processed_fighters: pd.DataFrame) -> Dict[str, Path]:
        """Save data efficiently"""
        print("💾 Saving data...")
        
        # Setup directories
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create versioned directory
        version_dir = data_dir / f"scrape_{SNAPSHOT_DATETIME}"
        version_dir.mkdir(exist_ok=True)
        
        # Convert to DataFrames
        fighters_df = pd.DataFrame(fighter_details)
        fights_df = pd.DataFrame(fight_histories)
        
        # Save files
        files_saved = {}
        
        # Raw data
        fighters_raw_path = version_dir / f'ufc_fighters_raw_{SNAPSHOT_DATE}.csv'
        fights_raw_path = version_dir / f'ufc_fights_{SNAPSHOT_DATE}.csv'
        
        fighters_df.to_csv(fighters_raw_path, index=False)
        fights_df.to_csv(fights_raw_path, index=False)
        files_saved['fighters_raw'] = fighters_raw_path
        files_saved['fights_raw'] = fights_raw_path
        
        # Processed data
        processed_path = version_dir / f'ufc_fighters_processed_{SNAPSHOT_DATE}.csv'
        processed_fighters.to_csv(processed_path, index=False)
        files_saved['fighters_processed'] = processed_path
        
        # Also save to main data directory for easy access
        main_fighters_path = data_dir / 'ufc_fighters_raw.csv'
        main_fights_path = data_dir / 'ufc_fights.csv'
        fighters_df.to_csv(main_fighters_path, index=False)
        fights_df.to_csv(main_fights_path, index=False)
        
        # Update ELO ratings with new fight data
        self.update_elo_ratings(fights_df)
        
        # Save metadata
        metadata = {
            'scrape_date': SNAPSHOT_DATE,
            'scrape_datetime': SNAPSHOT_DATETIME,
            'performance_stats': self.stats,
            'config_used': {
                'fighter_delay': self.config.FIGHTER_DELAY,
                'index_delay': self.config.INDEX_DELAY,
                'max_threads': self.config.MAX_WORKER_THREADS,
                'batch_size': self.config.BATCH_SIZE
            },
            'files_created': [str(p.name) for p in files_saved.values()],
            'elo_updated': ELO_AVAILABLE
        }
        
        metadata_path = version_dir / f'scrape_metadata_{SNAPSHOT_DATE}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        files_saved['metadata'] = metadata_path
        
        print(f"✅ Data saved to: {version_dir}")
        return files_saved
    
    def run_fast_scrape(self) -> Dict:
        """Execute complete fast scraping workflow"""
        self.stats['start_time'] = time.time()
        
        print("🚀 STARTING FAST UFC SCRAPING")
        print(f"⚙️ Config: {self.config.FIGHTER_DELAY}s delays, {self.config.MAX_WORKER_THREADS} threads")
        
        try:
            # Get all fighter URLs (with threading for letters)
            fighter_urls = self.get_all_fighter_urls_threaded()
            
            # Scrape all fighters (with threading for fighters)  
            fighter_details, fight_histories = self.scrape_fighters_batch_threaded(fighter_urls)
            
            # Basic feature engineering
            fighters_df = pd.DataFrame(fighter_details)
            processed_fighters = self.basic_feature_engineering(fighters_df)
            
            # Save data
            saved_files = self.save_data(fighter_details, fight_histories, processed_fighters)
            
            # Calculate performance statistics
            self.stats['end_time'] = time.time()
            total_time = self.stats['end_time'] - self.stats['start_time']
            
            # Estimate improvement over original
            original_estimated_time = len(fighter_urls) * 1.5 + 26 * 1.0 + 30  # Original delays + overhead
            improvement = max(0, (1 - total_time / original_estimated_time) * 100)
            
            results = {
                'success': True,
                'total_time_minutes': total_time / 60,
                'total_time_seconds': total_time,
                'performance_stats': self.stats,
                'data_summary': {
                    'fighters_scraped': len(fighter_details),
                    'fights_scraped': len(fight_histories),
                    'features_created': processed_fighters.shape[1]
                },
                'saved_files': saved_files,
                'performance_improvement': {
                    'estimated_original_time_minutes': original_estimated_time / 60,
                    'improvement_percentage': improvement,
                    'time_saved_minutes': (original_estimated_time - total_time) / 60
                }
            }
            
            print("\n" + "="*60)
            print("🎉 FAST UFC SCRAPING COMPLETE!")
            print("="*60)
            print(f"⏱️ Total time: {total_time/60:.1f} minutes")
            print(f"📈 Performance improvement: {improvement:.1f}%")
            print(f"💾 Time saved: {(original_estimated_time - total_time)/60:.1f} minutes")
            print(f"🥊 Scraped: {len(fighter_details)} fighters")
            print(f"🥋 Scraped: {len(fight_histories)} fights")
            print(f"⚙️ Created: {processed_fighters.shape[1]} features")
            print(f"🌐 Requests: {self.stats['requests_made']}")
            print(f"⚠️ Errors: {self.stats['errors_encountered']}")
            print("="*60)
            
            return results
            
        except Exception as e:
            print(f"❌ Fast scraping failed: {e}")
            raise

def main():
    """Main execution function for fast scraping"""
    
    config = FastScrapingConfig()
    scraper = FastUFCStatsScraper(config)
    
    try:
        results = scraper.run_fast_scrape()
        return results
    except KeyboardInterrupt:
        print("\n⏹️ Scraping interrupted by user")
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        raise

if __name__ == "__main__":
    results = main()